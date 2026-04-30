#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch

from flash_qla.ops.gated_delta_rule.recurrent_fused import (
    fused_sigmoid_gating_delta_rule_update as qla_update,
)


def find_sglang_fla_path(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("SGLANG_FLA_PATH"):
        candidates.append(Path(os.environ["SGLANG_FLA_PATH"]))
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[1] / "Ling-RL" / "crates/kernel-jit/python/sglang-fla",
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


def make_inputs(batch: int, tokens_per_seq: int, h: int, hv: int, kdim: int, vdim: int):
    total = batch * tokens_per_seq
    dev = torch.device("cuda")
    q = torch.randn(1, total, h, kdim, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, total, h, kdim, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, total, hv, vdim, device=dev, dtype=torch.bfloat16)
    a = torch.randn(1, total, hv, device=dev, dtype=torch.bfloat16)
    b = torch.randn(1, total, hv, device=dev, dtype=torch.bfloat16)
    a_log = torch.randn(hv, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(hv, device=dev, dtype=torch.bfloat16)
    state = torch.randn(batch, hv, vdim, kdim, device=dev, dtype=torch.float32)
    indices = torch.arange(batch, dtype=torch.int32, device=dev)
    cu_seqlens = torch.arange(0, total + 1, tokens_per_seq, dtype=torch.int32, device=dev)
    return q, k, v, a, b, a_log, dt_bias, state, indices, cu_seqlens


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


def run_case(
    label: str,
    sglang_update,
    batch: int,
    tokens_per_seq: int,
    h: int,
    hv: int,
    kdim: int,
    vdim: int,
    warmup: int,
    iters: int,
    verify: bool,
    tree: bool,
    block_dv: int | None,
):
    torch.manual_seed(20260430 + batch * 17 + tokens_per_seq)
    q, k, v, a, b, a_log, dt_bias, state0, indices, cu_seqlens = make_inputs(
        batch, tokens_per_seq, h, hv, kdim, vdim
    )
    scale = kdim**-0.5

    s_state = state0.clone()
    q_state = state0.clone()
    s_kwargs = {}
    q_kwargs = {}
    if verify:
        s_inter = torch.empty(batch, tokens_per_seq, hv, vdim, kdim, device="cuda", dtype=torch.float32)
        q_inter = torch.empty_like(s_inter)
        s_kwargs.update(
            disable_state_update=True,
            intermediate_states_buffer=s_inter,
            intermediate_state_indices=indices,
            cache_steps=tokens_per_seq,
        )
        q_kwargs.update(
            disable_state_update=True,
            intermediate_states_buffer=q_inter,
            intermediate_state_indices=indices,
            cache_steps=tokens_per_seq,
        )
        if tree:
            parent = torch.zeros(batch, tokens_per_seq, dtype=torch.int32, device="cuda")
            if tokens_per_seq > 1:
                parent[:, 1:] = (
                    torch.arange(1, tokens_per_seq, dtype=torch.int32, device="cuda") - 1
                ) // 2
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
        use_qk_l2norm_in_kernel=True,
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
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        block_dv=block_dv,
        assume_regular=True,
        **q_kwargs,
    )
    torch.cuda.synchronize()
    o_diff = (s_out.float() - q_out.float()).abs().max().item()
    state_diff = (s_state.float() - q_state.float()).abs().max().item()
    inter_diff = None
    if verify:
        inter_diff = (s_kwargs["intermediate_states_buffer"].float() - q_kwargs["intermediate_states_buffer"].float()).abs().max().item()

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
            use_qk_l2norm_in_kernel=True,
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
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            block_dv=block_dv,
            assume_regular=True,
            **q_kwargs,
        )

    s_ms = bench_cuda_ms(s_call, warmup, iters)
    q_ms = bench_cuda_ms(q_call, warmup, iters)
    speed = s_ms / q_ms if q_ms > 0 else float("inf")
    diff = f"o_diff={o_diff:.3g} state_diff={state_diff:.3g}"
    if inter_diff is not None:
        diff += f" inter_diff={inter_diff:.3g}"
    print(
        f"{label:12s} B={batch:3d} T={tokens_per_seq:2d} "
        f"SGLang={s_ms:.4f} ms FlashQLA={q_ms:.4f} ms speedup={speed:.3f}x {diff}",
        flush=True,
    )
    return s_ms, q_ms, speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-fla-path")
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--hv", type=int, default=48)
    parser.add_argument("--kdim", type=int, default=128)
    parser.add_argument("--vdim", type=int, default=128)
    parser.add_argument("--decode-batches", default="1,2,3,4,8,16,64")
    parser.add_argument("--verify-batches", default="1,2,4,8")
    parser.add_argument("--verify-tokens", default="16,32")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--block-dv", type=int)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    sglang_path = find_sglang_fla_path(args.sglang_fla_path)
    sys.path.insert(0, str(sglang_path))
    from fused_sigmoid_gating_recurrent import (  # noqa: WPS433
        fused_sigmoid_gating_delta_rule_update as sglang_update,
    )

    print(f"device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
    print(f"shape H={args.h} HV={args.hv} K={args.kdim} V={args.vdim}")
    print(f"sglang_fla_path={sglang_path}")

    decode_batches = [int(x) for x in args.decode_batches.split(",") if x]
    for batch in decode_batches:
        run_case(
            "decode",
            sglang_update,
            batch,
            1,
            args.h,
            args.hv,
            args.kdim,
            args.vdim,
            args.warmup,
            args.iters,
            verify=False,
            tree=False,
            block_dv=args.block_dv,
        )

    if not args.skip_verify:
        verify_batches = [int(x) for x in args.verify_batches.split(",") if x]
        verify_tokens = [int(x) for x in args.verify_tokens.split(",") if x]
        for tokens in verify_tokens:
            for batch in verify_batches:
                run_case(
                    "verify",
                    sglang_update,
                    batch,
                    tokens,
                    args.h,
                    args.hv,
                    args.kdim,
                    args.vdim,
                    args.warmup,
                    args.iters,
                    verify=True,
                    tree=False,
                    block_dv=args.block_dv,
                )
        for tokens in verify_tokens:
            for batch in verify_batches:
                run_case(
                    "verify_tree",
                    sglang_update,
                    batch,
                    tokens,
                    args.h,
                    args.hv,
                    args.kdim,
                    args.vdim,
                    args.warmup,
                    args.iters,
                    verify=True,
                    tree=True,
                    block_dv=args.block_dv,
                )


if __name__ == "__main__":
    main()
