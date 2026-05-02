#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""FlashQLA gated-delta-rule precision and benchmark runner.

The default case set is intentionally small enough for a local smoke run. Use
``--cases full`` and explicit speedup thresholds on the sm12x benchmark host for
performance gating.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable


def _add_local_import_paths() -> None:
    tests_dir = Path(__file__).resolve().parent
    flashqla_root = tests_dir.parent
    third_party = flashqla_root.parent
    for path in (
        tests_dir,
        flashqla_root,
        third_party / "flash-linear-attention",
    ):
        if path.exists():
            sys.path.insert(0, str(path))


_add_local_import_paths()
os.environ.setdefault("FLA_TILELANG", "0")

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_bwd as fla_bwd,
)
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd as fla_fwd,
)
from flash_qla import chunk_gated_delta_rule_bwd as qla_bwd
from flash_qla import chunk_gated_delta_rule_fwd as qla_fwd
from flash_qla.ops.gated_delta_rule.chunk.arch import is_sm12x
from flash_qla.utils import l2norm
from ref_gdr import chunk_gated_delta_rule_bwd as ref_bwd
from ref_gdr import chunk_gated_delta_rule_fwd as ref_fwd


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    num_tokens: int
    num_k_heads: int
    num_v_heads: int
    head_dim_k: int = 128
    head_dim_v: int = 128
    use_h0: bool = True
    varlen: bool = False
    cu_seqlens: tuple[int, ...] | None = None
    gate_scale: float = 1.0
    swa_ratio: float = 1.0
    auto_cp: bool = False
    min_fwd_speedup: float = 1.0
    min_bwd_speedup: float = 1.0


@dataclass
class TensorMetrics:
    max_abs: float
    mean_abs: float
    rms: float
    max_rel: float


@dataclass
class CaseResult:
    case: str
    fwd_ref: str
    bwd_ref: str
    out_max_abs: float
    out_mean_abs: float
    out_rms: float
    out_max_rel: float
    state_max_abs: float
    state_mean_abs: float
    state_rms: float
    state_max_rel: float
    dq_max_abs: float | None
    dk_max_abs: float | None
    dv_max_abs: float | None
    db_max_abs: float | None
    dg_max_abs: float | None
    dh0_max_abs: float | None
    fla_fwd_ms: float | None
    qla_fwd_ms: float | None
    fwd_speedup: float | None
    fla_bwd_ms: float | None
    qla_bwd_ms: float | None
    bwd_speedup: float | None
    passed: bool


SMOKE_CASES = (
    Case("short_tail", 1, 17, 16, 16, use_h0=True),
    Case(
        "tail_gate_stress",
        1,
        48,
        16,
        16,
        use_h0=True,
        gate_scale=128.0,
        min_fwd_speedup=0.0,
        min_bwd_speedup=0.0,
    ),
    Case("one_chunk", 1, 64, 16, 16, use_h0=True),
    Case("tail_2chunk", 1, 96, 16, 16, use_h0=True),
    Case("gva_tail", 1, 96, 8, 16, use_h0=True),
    Case("batch2", 2, 128, 16, 16, use_h0=True),
    Case(
        "varlen_mix",
        1,
        160,
        16,
        16,
        use_h0=True,
        varlen=True,
        cu_seqlens=(0, 17, 96, 160),
        min_bwd_speedup=0.95,
    ),
)

FULL_EXTRA_CASES = (
    Case("multi_chunk", 1, 256, 16, 16, use_h0=True),
    Case("multi_chunk_gva", 1, 256, 8, 16, use_h0=True),
    Case("no_h0", 1, 128, 16, 16, use_h0=False),
    Case("swa_half", 1, 192, 16, 16, use_h0=True, swa_ratio=0.5),
    Case("gate_stress", 1, 128, 16, 16, use_h0=True, gate_scale=16.0),
)

SERVER_CASES = (
    # ling-llm Qwen3.5-27B GPTQ prompt-prefill shape:
    # chunk_prefill_size=4096, 48 GDN value heads, GQA ratio 48/16.
    Case(
        "server_27b_chunk",
        1,
        4096,
        16,
        48,
        use_h0=True,
        min_fwd_speedup=1.0,
        min_bwd_speedup=0.0,
    ),
)


def _case_set(name: str) -> tuple[Case, ...]:
    if name == "smoke":
        return SMOKE_CASES
    if name == "full":
        return SMOKE_CASES + FULL_EXTRA_CASES
    if name == "server":
        return SERVER_CASES
    raise ValueError(f"unknown case set: {name}")


def _resolve_cases(case_set: str, only: str | None) -> list[Case]:
    cases = list(_case_set(case_set))
    if only is None:
        return cases
    wanted = {x.strip() for x in only.split(",") if x.strip()}
    selected = [case for case in cases if case.name in wanted]
    missing = wanted - {case.name for case in selected}
    if missing:
        raise SystemExit(f"unknown case name(s): {', '.join(sorted(missing))}")
    return selected


def _make_inputs(
    case: Case,
    dtype: torch.dtype,
    beta_dtype: torch.dtype,
    device: torch.device,
    seed: int,
    transpose_state_layout: bool,
    use_dht: bool,
):
    torch.manual_seed(seed)
    q = l2norm(
        torch.randn(
            (
                case.batch_size,
                case.num_tokens,
                case.num_k_heads,
                case.head_dim_k,
            ),
            device=device,
            dtype=dtype,
        )
    )
    k = l2norm(
        torch.randn(
            (
                case.batch_size,
                case.num_tokens,
                case.num_k_heads,
                case.head_dim_k,
            ),
            device=device,
            dtype=dtype,
        )
    )
    v = torch.randn(
        (
            case.batch_size,
            case.num_tokens,
            case.num_v_heads,
            case.head_dim_v,
        ),
        device=device,
        dtype=dtype,
    )
    g = (
        F.logsigmoid(
            torch.randn(
                (case.batch_size, case.num_tokens, case.num_v_heads),
                device=device,
                dtype=torch.float32,
            )
        )
        * (case.gate_scale / 16.0)
    )
    beta = torch.randn(
        (case.batch_size, case.num_tokens, case.num_v_heads),
        device=device,
        dtype=torch.float32,
    ).sigmoid().to(beta_dtype)
    dg_mask = torch.ones(
        (case.batch_size, case.num_tokens, case.num_v_heads),
        device=device,
        dtype=torch.bool,
    )

    if case.swa_ratio < 1.0:
        active = max(1, math.ceil(case.swa_ratio * case.num_v_heads))
        mask = torch.zeros((case.num_v_heads,), dtype=torch.bool, device=device)
        mask[:active] = True
        mask = mask[torch.randperm(case.num_v_heads, device=device)]
        g[:, :, ~mask] = 0.0
        dg_mask &= mask.view(1, 1, case.num_v_heads)

    cu_seqlens = None
    h0_batch = case.batch_size
    if case.varlen:
        if case.cu_seqlens is None:
            raise ValueError(f"{case.name}: varlen case requires cu_seqlens")
        if case.batch_size != 1:
            raise ValueError(f"{case.name}: varlen FlashQLA inputs must be flattened")
        if case.cu_seqlens[-1] != case.num_tokens:
            raise ValueError(f"{case.name}: cu_seqlens must end at num_tokens")
        cu_seqlens = torch.tensor(case.cu_seqlens, device=device, dtype=torch.int32)
        h0_batch = len(case.cu_seqlens) - 1

    h0 = None
    if case.use_h0:
        state_shape = (
            (h0_batch, case.num_v_heads, case.head_dim_v, case.head_dim_k)
            if transpose_state_layout
            else (h0_batch, case.num_v_heads, case.head_dim_k, case.head_dim_v)
        )
        h0 = torch.randn(
            state_shape,
            device=device,
            dtype=torch.float32,
        )
    do = torch.randn_like(v)
    dht = None
    if use_dht:
        state_shape = (
            (h0_batch, case.num_v_heads, case.head_dim_v, case.head_dim_k)
            if transpose_state_layout
            else (h0_batch, case.num_v_heads, case.head_dim_k, case.head_dim_v)
        )
        dht = torch.randn(
            state_shape,
            device=device,
            dtype=torch.float32,
        ) / 8.0

    return q, k, v, g, beta, h0, do, dht, cu_seqlens, dg_mask


def _metrics(
    actual: torch.Tensor | None,
    expected: torch.Tensor | None,
    mask: torch.Tensor | None = None,
) -> TensorMetrics:
    if actual is None and expected is None:
        return TensorMetrics(0.0, 0.0, 0.0, 0.0)
    if actual is None or expected is None:
        return TensorMetrics(float("inf"), float("inf"), float("inf"), float("inf"))
    diff = (actual.float() - expected.float()).abs()
    ref = expected.float().abs()
    if mask is not None:
        diff = diff[mask]
        ref = ref[mask]
        if diff.numel() == 0:
            return TensorMetrics(0.0, 0.0, 0.0, 0.0)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rms = diff.square().mean().sqrt().item()
    max_rel = (diff / ref.clamp_min(1e-6)).max().item()
    return TensorMetrics(max_abs, mean_abs, rms, max_rel)


def _time_cuda(fn: Callable[[], object], warmup: int, iters: int) -> float:
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


def _qla_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    h0,
    cu_seqlens,
    output_h: bool,
    output_v_new: bool,
    auto_cp: bool,
    transpose_state_layout: bool,
):
    return qla_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        output_h=output_h,
        auto_cp=auto_cp,
        output_v_new=output_v_new,
        transpose_state_layout=transpose_state_layout,
    )


def _run_case(case: Case, args: argparse.Namespace, device: torch.device) -> CaseResult:
    dtype = getattr(torch, args.dtype)
    beta_dtype = getattr(torch, args.beta_dtype)
    transpose_state_layout = args.state_layout == "transposed"
    q, k, v, g, beta, h0, do, dht, cu_seqlens, dg_mask = _make_inputs(
        case,
        dtype=dtype,
        beta_dtype=beta_dtype,
        device=device,
        seed=args.seed,
        transpose_state_layout=transpose_state_layout,
        use_dht=args.use_dht,
    )
    scale = case.head_dim_k ** -0.5
    qla_output_v_new = is_sm12x()

    fwd_ref_name = "fla"
    with torch.no_grad():
        try:
            if args.force_ref_fwd:
                raise RuntimeError("forced ref_gdr fwd")
            g_fla, o_fla, a_fla, state_fla, _, _ = fla_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=h0,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                transpose_state_layout=transpose_state_layout,
            )
        except Exception as exc:
            if not args.allow_ref_fwd_fallback:
                raise RuntimeError(
                    f"{case.name}: FLA fwd reference failed; rerun with "
                    "--allow-ref-fwd-fallback to compare against ref_gdr"
                ) from exc
            ref_h0 = (
                h0.transpose(-1, -2).contiguous()
                if transpose_state_layout and h0 is not None
                else h0
            )
            g_fla, o_fla, a_fla, _, state_fla = ref_fwd(
                q=q.to(torch.float32, copy=True),
                k=k.to(torch.float32, copy=True),
                v=v.to(torch.float32, copy=True),
                g=g.to(torch.float32, copy=True),
                beta=beta.to(torch.float32, copy=True),
                scale=scale,
                initial_state=ref_h0,
                cu_seqlens=cu_seqlens,
            )
            if transpose_state_layout:
                state_fla = state_fla.transpose(-1, -2).contiguous()
            fwd_ref_name = "ref_gdr"

        qla_result = _qla_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            h0,
            cu_seqlens,
            output_h=not args.skip_bwd,
            output_v_new=(not args.skip_bwd) and qla_output_v_new,
            auto_cp=case.auto_cp,
            transpose_state_layout=transpose_state_layout,
        )
        if (not args.skip_bwd) and qla_output_v_new:
            g_qla, a_qla, o_qla, h_qla, state_qla, v_new_qla = qla_result
        else:
            g_qla, a_qla, o_qla, h_qla, state_qla = qla_result
            v_new_qla = None

    out = _metrics(o_qla, o_fla)
    state = _metrics(state_qla, state_fla)

    bwd_ref_name = "skipped"
    dq_m = dk_m = dv_m = db_m = dg_m = dh0_m = TensorMetrics(0.0, 0.0, 0.0, 0.0)
    fla_bwd_ms = qla_bwd_ms = bwd_speedup = None
    if not args.skip_bwd:
        with torch.no_grad():
            try:
                if fwd_ref_name != "fla":
                    raise RuntimeError("FLA fwd reference unavailable")
                dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref, _, _ = fla_bwd(
                    q,
                    k,
                    v,
                    g_fla,
                    beta,
                    a_fla,
                    scale,
                    h0,
                    do,
                    dht,
                    cu_seqlens,
                    transpose_state_layout=transpose_state_layout,
                )
                bwd_ref_name = "fla"
            except Exception as exc:
                if not args.allow_ref_bwd_fallback:
                    raise RuntimeError(
                        f"{case.name}: FLA bwd reference failed; rerun with "
                        "--allow-ref-bwd-fallback to compare against ref_gdr"
                    ) from exc
                ref_h0 = (
                    h0.transpose(-1, -2).contiguous()
                    if transpose_state_layout and h0 is not None
                    else h0
                )
                ref_dht = (
                    dht.transpose(-1, -2).contiguous()
                    if transpose_state_layout and dht is not None
                    else dht
                )
                g_ref, _, a_ref, _, _ = ref_fwd(
                    q=q.to(torch.float32, copy=True),
                    k=k.to(torch.float32, copy=True),
                    v=v.to(torch.float32, copy=True),
                    g=g.to(torch.float32, copy=True),
                    beta=beta.to(torch.float32, copy=True),
                    scale=scale,
                    initial_state=ref_h0,
                    cu_seqlens=cu_seqlens,
                )
                dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref = ref_bwd(
                    q.to(torch.float32, copy=True),
                    k.to(torch.float32, copy=True),
                    v.to(torch.float32, copy=True),
                    g_ref,
                    beta.to(torch.float32, copy=True),
                    a_ref.to(torch.float32, copy=True),
                    scale,
                    ref_h0,
                    do.to(torch.float32, copy=True),
                    ref_dht,
                    cu_seqlens,
                )
                if transpose_state_layout:
                    dh0_ref = dh0_ref.transpose(-1, -2).contiguous()
                bwd_ref_name = "ref_gdr"

            dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla = qla_bwd(
                q,
                k,
                v,
                g_qla,
                beta,
                a_qla,
                do,
                dht,
                scale,
                h0,
                cu_seqlens,
                h=h_qla,
                v_new=v_new_qla,
                transpose_state_layout=transpose_state_layout,
            )

        dq_m = _metrics(dq_qla, dq_ref)
        dk_m = _metrics(dk_qla, dk_ref)
        dv_m = _metrics(dv_qla, dv_ref)
        db_m = _metrics(db_qla, db_ref)
        dg_m = _metrics(dg_qla, dg_ref, dg_mask)
        dh0_m = _metrics(dh0_qla, dh0_ref)

    fla_fwd_ms = qla_fwd_ms = fwd_speedup = None
    if not args.no_bench:
        with torch.no_grad():
            try:
                if args.force_ref_fwd:
                    raise RuntimeError("forced ref_gdr fwd")
                fla_fwd_ms = _time_cuda(
                    lambda: fla_fwd(
                        q=q,
                        k=k,
                        v=v,
                        g=g,
                        beta=beta,
                        scale=scale,
                        initial_state=h0,
                        output_final_state=True,
                        cu_seqlens=cu_seqlens,
                        transpose_state_layout=transpose_state_layout,
                    ),
                    args.warmup,
                    args.iters,
                )
            except Exception:
                if not args.allow_ref_fwd_fallback:
                    raise
                fla_fwd_ms = None

            qla_fwd_ms = _time_cuda(
                lambda: _qla_fwd(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    scale,
                    h0,
                    cu_seqlens,
                    output_h=False,
                    output_v_new=False,
                    auto_cp=case.auto_cp,
                    transpose_state_layout=transpose_state_layout,
                ),
                args.warmup,
                args.iters,
            )
            if fla_fwd_ms is not None:
                fwd_speedup = fla_fwd_ms / qla_fwd_ms

            if not args.skip_bwd:
                if bwd_ref_name == "fla":
                    fla_bwd_ms = _time_cuda(
                        lambda: fla_bwd(
                            q,
                            k,
                            v,
                            g_fla,
                            beta,
                            a_fla,
                            scale,
                            h0,
                            do,
                            dht,
                            cu_seqlens,
                            transpose_state_layout=transpose_state_layout,
                        ),
                        args.warmup,
                        args.iters,
                    )
                qla_bwd_ms = _time_cuda(
                    lambda: qla_bwd(
                        q,
                        k,
                        v,
                        g_qla,
                        beta,
                        a_qla,
                        do,
                        dht,
                        scale,
                        h0,
                        cu_seqlens,
                        h=h_qla,
                        v_new=v_new_qla,
                        transpose_state_layout=transpose_state_layout,
                    ),
                    args.warmup,
                    args.iters,
                )
                if fla_bwd_ms is not None:
                    bwd_speedup = fla_bwd_ms / qla_bwd_ms

    passed = True
    passed &= out.max_abs <= args.out_atol or out.max_rel <= args.out_rtol
    passed &= state.max_abs <= args.state_atol or state.max_rel <= args.state_rtol
    if not args.skip_bwd:
        for metric in (dq_m, dk_m, dv_m, db_m, dg_m, dh0_m):
            passed &= metric.max_abs <= args.grad_atol or metric.max_rel <= args.grad_rtol
    min_fwd_speedup = (
        args.min_fwd_speedup
        if args.min_fwd_speedup is not None
        else case.min_fwd_speedup
    )
    min_bwd_speedup = (
        args.min_bwd_speedup
        if args.min_bwd_speedup is not None
        else case.min_bwd_speedup
    )
    if fwd_speedup is not None and min_fwd_speedup > 0:
        passed &= fwd_speedup >= min_fwd_speedup
    if bwd_speedup is not None and min_bwd_speedup > 0:
        passed &= bwd_speedup >= min_bwd_speedup

    return CaseResult(
        case=case.name,
        fwd_ref=fwd_ref_name,
        bwd_ref=bwd_ref_name,
        out_max_abs=out.max_abs,
        out_mean_abs=out.mean_abs,
        out_rms=out.rms,
        out_max_rel=out.max_rel,
        state_max_abs=state.max_abs,
        state_mean_abs=state.mean_abs,
        state_rms=state.rms,
        state_max_rel=state.max_rel,
        dq_max_abs=dq_m.max_abs if not args.skip_bwd else None,
        dk_max_abs=dk_m.max_abs if not args.skip_bwd else None,
        dv_max_abs=dv_m.max_abs if not args.skip_bwd else None,
        db_max_abs=db_m.max_abs if not args.skip_bwd else None,
        dg_max_abs=dg_m.max_abs if not args.skip_bwd else None,
        dh0_max_abs=dh0_m.max_abs if not args.skip_bwd else None,
        fla_fwd_ms=fla_fwd_ms,
        qla_fwd_ms=qla_fwd_ms,
        fwd_speedup=fwd_speedup,
        fla_bwd_ms=fla_bwd_ms,
        qla_bwd_ms=qla_bwd_ms,
        bwd_speedup=bwd_speedup,
        passed=passed,
    )


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "-"
    if math.isinf(x) or math.isnan(x):
        return str(x)
    return f"{x:.{digits}g}"


def _print_result(result: CaseResult) -> None:
    print(
        " ".join(
            (
                f"{result.case:>16}",
                f"fwd_ref={result.fwd_ref}",
                f"bwd_ref={result.bwd_ref}",
                f"out={_fmt(result.out_max_abs)}",
                f"state={_fmt(result.state_max_abs)}",
                f"dq={_fmt(result.dq_max_abs)}",
                f"dk={_fmt(result.dk_max_abs)}",
                f"dv={_fmt(result.dv_max_abs)}",
                f"db={_fmt(result.db_max_abs)}",
                f"dg={_fmt(result.dg_max_abs)}",
                f"fla_fwd={_fmt(result.fla_fwd_ms)}ms",
                f"qla_fwd={_fmt(result.qla_fwd_ms)}ms",
                f"fwd_x={_fmt(result.fwd_speedup)}",
                f"qla_bwd={_fmt(result.qla_bwd_ms)}ms",
                f"bwd_x={_fmt(result.bwd_speedup)}",
                "PASS" if result.passed else "FAIL",
            )
        )
    )


def _write_csv(path: str, results: list[CaseResult]) -> None:
    names = [field.name for field in fields(CaseResult)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for result in results:
            writer.writerow({name: getattr(result, name) for name in names})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FlashQLA gated-delta-rule precision and benchmark runner"
    )
    parser.add_argument("--cases", choices=("smoke", "full", "server"), default="smoke")
    parser.add_argument("--only", help="comma-separated case names to run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--beta-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
        help="beta dtype; ling-llm inference uses bfloat16",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--state-layout",
        choices=("transposed", "standard"),
        default="transposed",
        help="state layout for h0/ht/dht; ling-llm uses transposed [H,V,K]",
    )
    parser.add_argument(
        "--use-dht",
        action="store_true",
        help="include final-state gradient in bwd checks; kernel-jit production bwd uses dht=None",
    )
    parser.add_argument("--skip-bwd", action="store_true")
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument(
        "--force-ref-fwd",
        action="store_true",
        help="use ref_gdr as the fwd precision reference and skip FLA fwd timing",
    )
    parser.add_argument("--allow-ref-fwd-fallback", action="store_true", default=True)
    parser.add_argument(
        "--no-ref-fwd-fallback",
        dest="allow_ref_fwd_fallback",
        action="store_false",
        help="fail instead of using ref_gdr when the Python FLA fwd reference cannot compile",
    )
    parser.add_argument("--allow-ref-bwd-fallback", action="store_true", default=True)
    parser.add_argument(
        "--no-ref-bwd-fallback",
        dest="allow_ref_bwd_fallback",
        action="store_false",
        help="fail instead of using ref_gdr when the Python FLA bwd reference cannot compile",
    )
    parser.add_argument("--out-atol", type=float, default=3e-2)
    parser.add_argument("--out-rtol", type=float, default=3e-2)
    parser.add_argument("--state-atol", type=float, default=1e-2)
    parser.add_argument("--state-rtol", type=float, default=3e-2)
    parser.add_argument("--grad-atol", type=float, default=5e-2)
    parser.add_argument("--grad-rtol", type=float, default=5e-2)
    parser.add_argument(
        "--min-fwd-speedup",
        type=float,
        default=None,
        help="override per-case minimum FLA/FlashQLA fwd speedup; set 0 to disable performance gating",
    )
    parser.add_argument(
        "--min-bwd-speedup",
        type=float,
        default=None,
        help="override per-case minimum FLA/FlashQLA bwd speedup; set 0 to disable performance gating",
    )
    parser.add_argument("--csv", help="optional CSV output path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for FlashQLA")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    cases = _resolve_cases(args.cases, args.only)
    print(
        f"device={device} sm12x={is_sm12x()} cases={len(cases)} "
        f"state_layout={args.state_layout} use_dht={args.use_dht} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print(
        "thresholds: "
        f"out<=({args.out_atol}, rtol {args.out_rtol}) "
        f"state<=({args.state_atol}, rtol {args.state_rtol}) "
        f"grad<=({args.grad_atol}, rtol {args.grad_rtol})"
    )
    if not args.no_bench:
        if args.min_fwd_speedup is None and args.min_bwd_speedup is None:
            print("perf gates: per-case defaults")
        else:
            print(
                "perf gates: "
                f"fwd>={args.min_fwd_speedup if args.min_fwd_speedup is not None else 'case'} "
                f"bwd>={args.min_bwd_speedup if args.min_bwd_speedup is not None else 'case'}"
            )

    results: list[CaseResult] = []
    t0 = time.perf_counter()
    for case in cases:
        result = _run_case(case, args, device)
        results.append(result)
        _print_result(result)

    if args.csv:
        _write_csv(args.csv, results)

    failed = [result.case for result in results if not result.passed]
    print(f"elapsed={time.perf_counter() - t0:.2f}s")
    if failed:
        print(f"failed cases: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
