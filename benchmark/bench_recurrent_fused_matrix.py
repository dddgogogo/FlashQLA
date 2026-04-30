#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from flash_qla.ops.gated_delta_rule.recurrent_fused import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update as qla_update,
)


PRESETS: dict[str, dict[str, str]] = {
    "smoke": {
        "modes": "decode,verify,tree",
        "decode_batches": "1,4,16",
        "verify_batches": "1,4",
        "verify_tokens": "8,16",
        "parent_patterns": "binary,random",
        "block_dv": "auto",
        "state_indices": "identity",
        "qk_l2norm": "true",
    },
    "standard": {
        "modes": "decode,verify,tree",
        "decode_batches": "1,2,3,4,8,16,32,64,128",
        "verify_batches": "1,2,4,8",
        "verify_tokens": "2,4,8,16,32",
        "parent_patterns": "chain,binary,star,random",
        "block_dv": "auto",
        "state_indices": "identity",
        "qk_l2norm": "true",
    },
    "exhaustive": {
        "modes": "decode,verify,tree",
        "decode_batches": "1,2,3,4,8,16,32,64,128",
        "verify_batches": "1,2,4,8,16",
        "verify_tokens": "2,4,8,16,32,64",
        "parent_patterns": "chain,binary,star,random",
        "block_dv": "auto,4,8",
        "state_indices": "identity,reverse,negative_last",
        "qk_l2norm": "true,false",
    },
}


@dataclass(frozen=True)
class BenchCase:
    mode: str
    batch: int
    tokens_per_seq: int
    parent_pattern: str
    block_dv: int | None
    state_indices: str
    qk_l2norm: bool


@dataclass
class BenchResult:
    case_id: int
    mode: str
    parent_pattern: str
    batch: int
    tokens_per_seq: int
    block_dv: str
    state_indices: str
    qk_l2norm: bool
    sglang_ms: float
    flashqla_ms: float
    speedup: float
    o_diff: float
    state_diff: float
    inter_diff: float | None
    passed: bool
    error: str


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def int_list(value: str) -> list[int]:
    return [int(item) for item in csv_list(value)]


def bool_list(value: str) -> list[bool]:
    mapping = {"1": True, "true": True, "yes": True, "0": False, "false": False, "no": False}
    result = []
    for item in csv_list(value.lower()):
        if item not in mapping:
            raise ValueError(f"Invalid boolean list item: {item}")
        result.append(mapping[item])
    return result


def block_list(value: str) -> list[int | None]:
    blocks: list[int | None] = []
    for item in csv_list(value.lower()):
        blocks.append(None if item == "auto" else int(item))
    return blocks


def block_label(block_dv: int | None) -> str:
    return "auto" if block_dv is None else str(block_dv)


def find_sglang_fla_path(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("SGLANG_FLA_PATH"):
        candidates.append(Path(os.environ["SGLANG_FLA_PATH"]))
    candidates.extend(
        [
            REPO_ROOT.parent / "Ling-RL/crates/kernel-jit/python/sglang-fla",
            Path("/home/spark/ai-tool/src/Ling-RL/crates/kernel-jit/python/sglang-fla"),
            Path("/home/gpu/Ling-RL/crates/kernel-jit/python/sglang-fla"),
            Path("/home/gpu/ai-tool/src/Ling-RL/crates/kernel-jit/python/sglang-fla"),
        ]
    )
    for path in candidates:
        path = path.expanduser().resolve()
        if (path / "fused_sigmoid_gating_recurrent.py").is_file():
            return path
    raise FileNotFoundError("Could not find fused_sigmoid_gating_recurrent.py; pass --sglang-fla-path")


def make_inputs(
    batch: int,
    tokens_per_seq: int,
    h: int,
    hv: int,
    kdim: int,
    vdim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    total = batch * tokens_per_seq
    dev = torch.device("cuda")
    q = torch.randn(1, total, h, kdim, device=dev, dtype=dtype)
    k = torch.randn(1, total, h, kdim, device=dev, dtype=dtype)
    v = torch.randn(1, total, hv, vdim, device=dev, dtype=dtype)
    a = torch.randn(1, total, hv, device=dev, dtype=dtype)
    b = torch.randn(1, total, hv, device=dev, dtype=dtype)
    a_log = torch.randn(hv, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(hv, device=dev, dtype=dtype)
    state = torch.randn(batch, hv, vdim, kdim, device=dev, dtype=torch.float32)
    cu_seqlens = torch.arange(0, total + 1, tokens_per_seq, dtype=torch.int32, device=dev)
    return q, k, v, a, b, a_log, dt_bias, state, cu_seqlens


def make_state_indices(batch: int, mode: str) -> torch.Tensor:
    indices = torch.arange(batch, dtype=torch.int32, device="cuda")
    if mode == "identity":
        return indices
    if mode == "reverse":
        return torch.flip(indices, dims=(0,)).contiguous()
    if mode == "negative_last":
        if batch > 0:
            indices[-1] = -1
        return indices
    raise ValueError(f"Unknown state index mode: {mode}")


def make_parent(batch: int, tokens_per_seq: int, pattern: str, seed: int) -> torch.Tensor | None:
    if pattern == "none":
        return None
    parent = torch.zeros(batch, tokens_per_seq, dtype=torch.int32, device="cuda")
    if tokens_per_seq <= 1:
        return parent
    steps = torch.arange(1, tokens_per_seq, dtype=torch.int32, device="cuda")
    if pattern == "chain":
        parent[:, 1:] = steps - 1
    elif pattern == "binary":
        parent[:, 1:] = (steps - 1) // 2
    elif pattern == "star":
        parent[:, 1:] = 0
    elif pattern == "random":
        batch_offsets = torch.arange(batch, dtype=torch.int32, device="cuda")[:, None] * 13
        step_offsets = steps[None, :] * 17 + seed
        parent[:, 1:] = (batch_offsets + step_offsets) % steps[None, :]
    else:
        raise ValueError(f"Unknown parent pattern: {pattern}")
    return parent


def case_seed(base_seed: int, case: BenchCase) -> int:
    mode_id = {"decode": 1, "verify": 2, "tree": 3}[case.mode]
    parent_id = {"none": 0, "chain": 1, "binary": 2, "star": 3, "random": 4}[case.parent_pattern]
    state_id = {"identity": 1, "reverse": 2, "negative_last": 3}[case.state_indices]
    qk_id = 1 if case.qk_l2norm else 0
    return (
        base_seed
        + mode_id * 1_000_003
        + parent_id * 100_003
        + state_id * 10_007
        + case.batch * 503
        + case.tokens_per_seq * 53
        + qk_id * 7
    )


def bench_cuda_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def build_cases(args: argparse.Namespace) -> list[BenchCase]:
    cases: list[BenchCase] = []
    modes = csv_list(args.modes)
    blocks = block_list(args.block_dv)
    state_modes = csv_list(args.state_indices)
    qk_modes = bool_list(args.qk_l2norm)

    if "decode" in modes:
        for batch in int_list(args.decode_batches):
            for block in blocks:
                for state_mode in state_modes:
                    for qk_l2norm in qk_modes:
                        cases.append(
                            BenchCase("decode", batch, 1, "none", block, state_mode, qk_l2norm)
                        )

    verify_tokens = int_list(args.verify_tokens)
    verify_batches = int_list(args.verify_batches)
    if "verify" in modes:
        for tokens in verify_tokens:
            for batch in verify_batches:
                for block in blocks:
                    for state_mode in state_modes:
                        for qk_l2norm in qk_modes:
                            cases.append(
                                BenchCase("verify", batch, tokens, "none", block, state_mode, qk_l2norm)
                            )

    if "tree" in modes:
        for pattern in csv_list(args.parent_patterns):
            for tokens in verify_tokens:
                for batch in verify_batches:
                    for block in blocks:
                        for state_mode in state_modes:
                            for qk_l2norm in qk_modes:
                                cases.append(
                                    BenchCase(
                                        "tree", batch, tokens, pattern, block, state_mode, qk_l2norm
                                    )
                                )

    return cases


def run_case(
    case_id: int,
    case: BenchCase,
    sglang_update,
    args: argparse.Namespace,
) -> BenchResult:
    seed = case_seed(args.seed, case)
    torch.manual_seed(seed)
    dtype = getattr(torch, args.dtype)
    q, k, v, a, b, a_log, dt_bias, state0, cu_seqlens = make_inputs(
        case.batch,
        case.tokens_per_seq,
        args.h,
        args.hv,
        args.kdim,
        args.vdim,
        dtype,
    )
    indices = make_state_indices(case.batch, case.state_indices)
    scale = args.kdim**-0.5
    s_state = state0.clone()
    q_state = state0.clone()

    verify = case.mode in ("verify", "tree")
    s_kwargs: dict[str, Any] = {}
    q_kwargs: dict[str, Any] = {}
    if verify:
        s_inter = torch.empty(
            case.batch,
            case.tokens_per_seq,
            args.hv,
            args.vdim,
            args.kdim,
            device="cuda",
            dtype=torch.float32,
        )
        q_inter = torch.empty_like(s_inter)
        s_kwargs.update(
            disable_state_update=True,
            intermediate_states_buffer=s_inter,
            intermediate_state_indices=indices,
            cache_steps=case.tokens_per_seq,
        )
        q_kwargs.update(
            disable_state_update=True,
            intermediate_states_buffer=q_inter,
            intermediate_state_indices=indices,
            cache_steps=case.tokens_per_seq,
        )
        parent = make_parent(case.batch, case.tokens_per_seq, case.parent_pattern, seed)
        if parent is not None:
            s_kwargs["retrieve_parent_token"] = parent
            q_kwargs["retrieve_parent_token"] = parent

    s_out = sglang_update(
        a_log,
        a,
        dt_bias,
        1.0,
        20.0,
        q,
        k,
        v,
        b,
        s_state,
        indices,
        scale=scale,
        use_qk_l2norm_in_kernel=case.qk_l2norm,
        cu_seqlens=cu_seqlens,
        **s_kwargs,
    )
    q_out = qla_update(
        a_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        q_state,
        indices,
        scale=scale,
        use_qk_l2norm_in_kernel=case.qk_l2norm,
        cu_seqlens=cu_seqlens,
        block_dv=case.block_dv,
        assume_regular=True,
        **q_kwargs,
    )
    torch.cuda.synchronize()
    o_diff = (s_out.float() - q_out.float()).abs().max().item()
    state_diff = (s_state.float() - q_state.float()).abs().max().item()
    inter_diff = None
    if verify:
        inter_diff = (
            s_kwargs["intermediate_states_buffer"].float()
            - q_kwargs["intermediate_states_buffer"].float()
        ).abs().max().item()

    def s_call():
        sglang_update(
            a_log,
            a,
            dt_bias,
            1.0,
            20.0,
            q,
            k,
            v,
            b,
            s_state,
            indices,
            scale=scale,
            use_qk_l2norm_in_kernel=case.qk_l2norm,
            cu_seqlens=cu_seqlens,
            **s_kwargs,
        )

    def q_call():
        qla_update(
            a_log,
            a,
            dt_bias,
            q,
            k,
            v,
            b,
            q_state,
            indices,
            scale=scale,
            use_qk_l2norm_in_kernel=case.qk_l2norm,
            cu_seqlens=cu_seqlens,
            block_dv=case.block_dv,
            assume_regular=True,
            **q_kwargs,
        )

    s_ms = bench_cuda_ms(s_call, args.warmup, args.iters)
    q_ms = bench_cuda_ms(q_call, args.warmup, args.iters)
    speedup = s_ms / q_ms if q_ms > 0 else math.inf
    passed = (
        o_diff <= args.max_o_diff
        and state_diff <= args.max_state_diff
        and (inter_diff is None or inter_diff <= args.max_inter_diff)
    )
    return BenchResult(
        case_id=case_id,
        mode=case.mode,
        parent_pattern=case.parent_pattern,
        batch=case.batch,
        tokens_per_seq=case.tokens_per_seq,
        block_dv=block_label(case.block_dv),
        state_indices=case.state_indices,
        qk_l2norm=case.qk_l2norm,
        sglang_ms=s_ms,
        flashqla_ms=q_ms,
        speedup=speedup,
        o_diff=o_diff,
        state_diff=state_diff,
        inter_diff=inter_diff,
        passed=passed,
        error="",
    )


def failed_result(case_id: int, case: BenchCase, error: Exception) -> BenchResult:
    return BenchResult(
        case_id=case_id,
        mode=case.mode,
        parent_pattern=case.parent_pattern,
        batch=case.batch,
        tokens_per_seq=case.tokens_per_seq,
        block_dv=block_label(case.block_dv),
        state_indices=case.state_indices,
        qk_l2norm=case.qk_l2norm,
        sglang_ms=math.nan,
        flashqla_ms=math.nan,
        speedup=math.nan,
        o_diff=math.nan,
        state_diff=math.nan,
        inter_diff=math.nan,
        passed=False,
        error=f"{type(error).__name__}: {error}",
    )


def print_result(result: BenchResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    parent = "" if result.parent_pattern == "none" else f" parent={result.parent_pattern}"
    err = "" if not result.error else f" error={result.error}"
    print(
        f"{status:4s} #{result.case_id:04d} {result.mode:6s}{parent:14s} "
        f"B={result.batch:3d} T={result.tokens_per_seq:2d} block={result.block_dv:>4s} "
        f"idx={result.state_indices:13s} qk_norm={int(result.qk_l2norm)} "
        f"SGLang={result.sglang_ms:.4f} ms FlashQLA={result.flashqla_ms:.4f} ms "
        f"speedup={result.speedup:.3f}x o={result.o_diff:.3g} "
        f"state={result.state_diff:.3g} inter={result.inter_diff if result.inter_diff is not None else 'n/a'}"
        f"{err}",
        flush=True,
    )


def write_csv(path: Path, results: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_json(path: Path, metadata: dict[str, Any], results: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": [asdict(result) for result in results]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def apply_preset(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matrix benchmark for FlashQLA recurrent fused GDN.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="standard")
    parser.add_argument("--sglang-fla-path")
    parser.add_argument("--modes", help="Comma list: decode,verify,tree")
    parser.add_argument("--decode-batches")
    parser.add_argument("--verify-batches")
    parser.add_argument("--verify-tokens")
    parser.add_argument("--parent-patterns", help="Comma list: chain,binary,star,random")
    parser.add_argument("--block-dv", help="Comma list of auto/int block sizes, e.g. auto,4,8")
    parser.add_argument("--state-indices", help="Comma list: identity,reverse,negative_last")
    parser.add_argument("--qk-l2norm", help="Comma list of booleans")
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--hv", type=int, default=48)
    parser.add_argument("--kdim", type=int, default=128)
    parser.add_argument("--vdim", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--max-o-diff", type=float, default=3e-3)
    parser.add_argument("--max-state-diff", type=float, default=1e-4)
    parser.add_argument("--max-inter-diff", type=float, default=1e-4)
    parser.add_argument("--csv", type=Path, help="Write per-case results to CSV")
    parser.add_argument("--json", type=Path, help="Write metadata and results to JSON")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--empty-cache-every", type=int, default=0)
    args = parser.parse_args()
    return apply_preset(parser, args)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    sglang_path = find_sglang_fla_path(args.sglang_fla_path)
    sys.path.insert(0, str(sglang_path))
    from fused_sigmoid_gating_recurrent import (  # noqa: WPS433
        fused_sigmoid_gating_delta_rule_update as sglang_update,
    )

    cases = build_cases(args)
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "sglang_fla_path": str(sglang_path),
        "preset": args.preset,
        "h": args.h,
        "hv": args.hv,
        "kdim": args.kdim,
        "vdim": args.vdim,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "num_cases": len(cases),
    }

    print(
        f"device={metadata['device']} capability={metadata['capability']} "
        f"torch={metadata['torch']}"
    )
    print(f"shape H={args.h} HV={args.hv} K={args.kdim} V={args.vdim} dtype={args.dtype}")
    print(f"preset={args.preset} cases={len(cases)} sglang_fla_path={sglang_path}")

    if args.list_cases:
        for i, case in enumerate(cases):
            print(f"#{i:04d} {case}")
        return 0

    results: list[BenchResult] = []
    for case_id, case in enumerate(cases):
        try:
            result = run_case(case_id, case, sglang_update, args)
        except Exception as exc:  # Keep long matrices useful after one bad case.
            result = failed_result(case_id, case, exc)
        results.append(result)
        print_result(result)
        if args.empty_cache_every and (case_id + 1) % args.empty_cache_every == 0:
            torch.cuda.empty_cache()

    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    valid_speedups = [result.speedup for result in results if result.passed and math.isfinite(result.speedup)]
    min_speedup = min(valid_speedups) if valid_speedups else math.nan
    geomean = (
        math.exp(sum(math.log(speedup) for speedup in valid_speedups) / len(valid_speedups))
        if valid_speedups
        else math.nan
    )
    print(
        f"summary cases={len(results)} passed={passed} failed={failed} "
        f"min_speedup={min_speedup:.3f}x geomean_speedup={geomean:.3f}x"
    )

    if args.csv:
        write_csv(args.csv, results)
        print(f"wrote_csv={args.csv}")
    if args.json:
        write_json(args.json, metadata, results)
        print(f"wrote_json={args.json}")

    return 1 if failed and args.fail_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
