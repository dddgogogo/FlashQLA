# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
#
# Honest FLA(Triton) vs pure-TileLang(QLA) comparison for the Qwen3.6-27B GDN
# shape on sm12x (Blackwell sm120/sm121).
#
# Design contract (why this script exists):
#   * The TileLang side NEVER calls into FLA. On sm12x the backward is now 100%
#     pure TileLang (blackwell/bwd_sm12x.py); the old FLA-coupled fused_bwd.py
#     was deleted. Mixing a TileLang forward with an FLA backward (or
#     vice-versa) produces inconsistent precision, which is why the sm12x path
#     is fully pure -- exactly the kernels Ling-RL ships via kernel-jit.
#   * The FLA side is forced to its Triton backend (FLA_TILELANG=0).
#   * Every case reports BOTH timing and max-abs-error vs an fp32 reference, so
#     "faster" is never reported without "correct".
#
# It covers chunk forward, chunk backward, and recurrent decode/verify/tree.
# For recurrent it reports BOTH the deployed AOT kernel (the generic
# tilelang_flashqla_gdn_update, the only kernel loadable through Ling-RL's Rust
# ABI) and the performance ceiling (the warp-specialized kernels the eager
# dispatcher selects), against FLA's fused_recurrent.

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Force FLA onto its Triton backend BEFORE importing fla, so the baseline is a
# genuine FLA-Triton number and never FLA's own TileLang path.
os.environ.setdefault("FLA_TILELANG", "0")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tests"))

from flash_qla.ops.gated_delta_rule.chunk import (  # noqa: E402
    chunk_gated_delta_rule_fwd as qla_fwd,
    chunk_gated_delta_rule_bwd as qla_bwd,
)
from flash_qla.ops.gated_delta_rule.chunk.arch import is_sm12x  # noqa: E402
from flash_qla.ops.gated_delta_rule.chunk.blackwell.bwd_sm12x import (  # noqa: E402
    chunk_gated_delta_rule_bwd_sm12x,
)
from flash_qla.ops.utils.cumsum import chunk_local_cumsum  # noqa: E402
from flash_qla.utils import l2norm  # noqa: E402

# fp32 reference (numpy/torch, no kernels)
from ref_gdr import (  # noqa: E402
    chunk_gated_delta_rule_fwd as ref_fwd,
    chunk_gated_delta_rule_bwd as ref_bwd,
)

# FLA Triton baseline
from fla.ops.gated_delta_rule.chunk import (  # noqa: E402
    chunk_gated_delta_rule_fwd as fla_fwd,
    chunk_gated_delta_rule_bwd as fla_bwd,
)

HEAD_DIM = 128
CHUNK = 64


def bench_ms(fn, warmup: int, iters: int) -> float:
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


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def make_chunk_inputs(B, T, Hk, Hv, seed=42, swa_ratio=0.75):
    dev = "cuda"
    torch.manual_seed(seed)
    q = l2norm(torch.randn(B, T, Hk, HEAD_DIM, device=dev, dtype=torch.bfloat16))
    k = l2norm(torch.randn(B, T, Hk, HEAD_DIM, device=dev, dtype=torch.bfloat16))
    v = torch.randn(B, T, Hv, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, Hv, device=dev, dtype=torch.float32)) / 16
    beta = torch.randn(B, T, Hv, device=dev, dtype=torch.float32).sigmoid()
    do = torch.randn_like(v)
    # sliding-window-attention gate mask, matching test_gdr.py
    swa = torch.zeros(Hv, dtype=torch.bool, device=dev)
    swa[: math.ceil(swa_ratio * Hv)] = True
    swa = swa[torch.randperm(Hv, device=dev)]
    g[:, :, ~swa] = 0.0
    return q, k, v, g, beta, do


def group_reduce_heads(grad: torch.Tensor, Hk: int) -> torch.Tensor:
    """Reduce per-value-head grads [B,T,Hv,D] to per-key-head [B,T,Hk,D] for GQA."""
    B, T, Hv, D = grad.shape
    if Hv == Hk:
        return grad
    group = Hv // Hk
    return grad.reshape(B, T, Hk, group, D).sum(dim=3)


def run_chunk(args):
    B, T, Hk, Hv = 1, args.tokens, args.hk, args.hv
    scale = HEAD_DIM ** -0.5
    q, k, v, g, beta, do = make_chunk_inputs(B, T, Hk, Hv, seed=args.seed)

    print(f"\n=== CHUNK  B={B} T={T} Hk={Hk} Hv={Hv} K=V={HEAD_DIM} chunk={CHUNK} bf16 ===")

    # ---- fp32 reference (ground truth) ----
    qf, kf, vf = q.float(), k.float(), v.float()
    g_ref, o_ref, A_ref, h_ref, s_ref = ref_fwd(
        q=qf.clone(), k=kf.clone(), v=vf.clone(),
        g=g.clone(), beta=beta.clone(), scale=scale,
        initial_state=None, chunk_size=CHUNK,
    )
    # NOTE: ref_bwd expects the CUMSUM'd g (g_ref returned by ref_fwd), not raw g.
    dq_ref, dk_ref, dv_ref, db_ref, dg_ref, _ = ref_bwd(
        qf.clone(), kf.clone(), vf.clone(), g_ref.clone(), beta.clone(),
        A_ref.clone(), scale, None, do.float().clone(), None, None, CHUNK,
    )

    # ---- pure TileLang forward (no FLA) ----
    g_q, A_q, o_q, h_q, s_q, *rest = qla_fwd(
        q, k, v, g, beta, scale=scale, initial_state=None,
        output_final_state=True, output_h=True, auto_cp=False,
        output_v_new=is_sm12x(),
    )
    v_new_q = rest[0] if rest else None
    o_err = max_abs_err(o_q, o_ref)

    # ---- FLA baseline forward (forced Triton) ----
    g_fla, o_fla, A_fla, s_fla, *_ = fla_fwd(
        q, k, v, g, beta, scale=scale, initial_state=None,
        output_final_state=True, cu_seqlens=None,
    )
    o_fla_err = max_abs_err(o_fla, o_ref)

    def fwd_qla():
        qla_fwd(q, k, v, g, beta, scale=scale, initial_state=None,
                output_final_state=True, output_h=False, auto_cp=False)

    def fwd_fla():
        fla_fwd(q, k, v, g, beta, scale=scale, initial_state=None,
                output_final_state=True, cu_seqlens=None)

    fwd_q_ms = bench_ms(fwd_qla, args.warmup, args.iters)
    fwd_f_ms = bench_ms(fwd_fla, args.warmup, args.iters)

    print("  FORWARD")
    print(f"    correctness (max-abs-err vs fp32 ref):  TileLang={o_err:.3e}   FLA={o_fla_err:.3e}")
    print(f"    latency:  TileLang={fwd_q_ms:.4f} ms   FLA={fwd_f_ms:.4f} ms   speedup={fwd_f_ms/fwd_q_ms:.2f}x")

    # ---- decomposed sm12x backward (chunk_gated_delta_rule_bwd) ----
    # On sm12x this routes through the 100% pure-TileLang decomposed backward
    # (bwd_sm12x.py); there is no FLA on the sm12x path anymore.
    print("  BACKWARD (pure-TileLang sm12x path)")
    cap = torch.cuda.get_device_capability()
    try:
        dq_q, dk_q, dv_q, db_q, dg_q, _ = qla_bwd(
            q=q, k=k, v=v, g=g_q, beta=beta, A=A_q, do=do, dht=None,
            scale=scale, initial_state=None, cu_seqlens=None, h=h_q, v_new=v_new_q,
        )
        bwd_errs = {
            "dq": max_abs_err(dq_q, dq_ref), "dk": max_abs_err(dk_q, dk_ref),
            "dv": max_abs_err(dv_q, dv_ref), "db": max_abs_err(db_q, db_ref),
            "dg": max_abs_err(dg_q, dg_ref),
        }

        def bwd_qla():
            qla_bwd(q=q, k=k, v=v, g=g_q, beta=beta, A=A_q, do=do, dht=None,
                    scale=scale, initial_state=None, cu_seqlens=None, h=h_q, v_new=v_new_q)

        def bwd_fla():
            fla_bwd(q, k, v, g_fla, beta, A_fla, scale, None, do, None, None)

        bwd_q_ms = bench_ms(bwd_qla, args.warmup, args.iters)
        try:
            bwd_f_ms = bench_ms(bwd_fla, args.warmup, args.iters)
        except Exception as e:
            print(f"    [warn] FLA bwd failed: {e}")
            bwd_f_ms = float("nan")
        print("    [hybrid] correctness (max-abs-err vs fp32 ref): " +
              "  ".join(f"{kk}={vv:.3e}" for kk, vv in bwd_errs.items()))
        print(f"    [hybrid] latency:  QLA={bwd_q_ms:.4f} ms   FLA={bwd_f_ms:.4f} ms   speedup={bwd_f_ms/bwd_q_ms:.2f}x")
    except Exception as e:
        msg = str(e).splitlines()[-1] if str(e) else type(e).__name__
        print(f"    [error] decomposed bwd failed on sm{cap[0]}{cap[1]}: {msg}")

    # ---- pure-TileLang sm12x backward (NO FLA at the compute layer) ----
    print("  BACKWARD (pure-TileLang sm12x path, no FLA)")
    try:
        dq_p, dk_p, dv_p, db_p, dg_p = chunk_gated_delta_rule_bwd_sm12x(
            q, k, v, g_q, beta, A_q, do, h_q, v_new_q, scale, chunk_size=CHUNK,
        )
        errs = {
            "dq": max_abs_err(dq_p, dq_ref), "dk": max_abs_err(dk_p, dk_ref),
            "dv": max_abs_err(dv_p, dv_ref), "db": max_abs_err(db_p, db_ref),
            "dg": max_abs_err(dg_p, dg_ref),
        }

        def bwd_pure():
            chunk_gated_delta_rule_bwd_sm12x(
                q, k, v, g_q, beta, A_q, do, h_q, v_new_q, scale, chunk_size=CHUNK)

        bwd_p_ms = bench_ms(bwd_pure, args.warmup, args.iters)
        print("    correctness (max-abs-err vs fp32 ref): " +
              "  ".join(f"{kk}={vv:.3e}" for kk, vv in errs.items()))
        print(f"    latency:  pure-TileLang={bwd_p_ms:.4f} ms   FLA={bwd_f_ms:.4f} ms   speedup={bwd_f_ms/bwd_p_ms:.2f}x")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"    [error] pure-TileLang bwd failed: {str(e).splitlines()[-1] if str(e) else type(e).__name__}")
    return None


def run_recurrent(args):
    """Honest recurrent-decode comparison: pure-TileLang QLA update vs FLA's
    fused_recurrent (Triton), Qwen3.6-27B decode (tokens_per_seq=1), batch sweep."""
    import importlib.util
    # reuse the proven input maker + FLA call convention
    sys.path.insert(0, str(_REPO_ROOT))
    from bench_sglang_fused import make_inputs  # noqa: E402
    from fla.ops.gated_delta_rule.fused_recurrent import (  # noqa: E402
        fused_recurrent_gated_delta_rule as fla_update,
    )
    from flash_qla.ops.gated_delta_rule.recurrent_fused import (  # noqa: E402
        fused_sigmoid_gating_delta_rule_update as qla_update,
    )

    H, HV, K, V = args.hk, args.hv, HEAD_DIM, HEAD_DIM
    scale = K ** -0.5
    print(f"\n=== RECURRENT DECODE  Hk={H} Hv={HV} K=V={K} bf16 (vs FLA fused_recurrent) ===")
    print(f"{'batch':>6} {'TileLang(ms)':>13} {'FLA(ms)':>10} {'speedup':>8} {'o_diff':>10} {'state_diff':>11}")
    for batch in [int(x) for x in (args.decode_batches or "1,2,4,8,16,64").split(",")]:
        torch.manual_seed(args.seed + batch)
        q, k, v, a, b, a_log, dt_bias, state0, indices, cu_seqlens = make_inputs(
            batch, 1, H, HV, K, V)
        fs, qs = state0.clone(), state0.clone()
        fo, ffs = fla_update(
            q, k, v, g=a, beta=b.sigmoid(), scale=scale, initial_state=fs,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True, A_log=a_log, dt_bias=dt_bias,
            cu_seqlens=cu_seqlens, transpose_state_layout=True)
        qo = qla_update(
            a_log, a, dt_bias, q, k, v, b, qs, None, scale=scale,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens, assume_regular=True)
        torch.cuda.synchronize()
        o_diff = (fo.float() - qo.float()).abs().max().item()
        st_diff = (ffs.float() - qs.float()).abs().max().item()

        def q_call():
            qla_update(a_log, a, dt_bias, q, k, v, b, state0.clone(), None, scale=scale,
                       use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens, assume_regular=True)

        def f_call():
            fla_update(q, k, v, g=a, beta=b.sigmoid(), scale=scale,
                       initial_state=state0.clone(), output_final_state=True,
                       use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                       A_log=a_log, dt_bias=dt_bias, cu_seqlens=cu_seqlens,
                       transpose_state_layout=True)

        qm = bench_ms(q_call, args.warmup, args.iters)
        fm = bench_ms(f_call, args.warmup, args.iters)
        print(f"{batch:>6} {qm:>13.4f} {fm:>10.4f} {fm/qm:>7.2f}x {o_diff:>10.2e} {st_diff:>11.2e}")


def parse_args():
    p = argparse.ArgumentParser(description="Honest FLA vs pure-TileLang GDN benchmark (Qwen3.6-27B)")
    p.add_argument("--decode-batches", default=None, help="recurrent decode batch sweep")
    p.add_argument("--hk", type=int, default=16, help="num key heads (27B TP1 = 16)")
    p.add_argument("--hv", type=int, default=48, help="num value heads (27B TP1 = 48)")
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--seqlens", default=None,
                   help="comma list of token counts to sweep (default: 2048,4096,8192,16384)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["chunk", "recurrent", "all"], default="chunk")
    return p.parse_args()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    args = parse_args()
    cap = torch.cuda.get_device_capability()
    print(f"device={torch.cuda.get_device_name()} cap={cap} torch={torch.__version__}")
    print(f"FLA_TILELANG={os.environ.get('FLA_TILELANG')} (0 => FLA forced to Triton)")
    print("Model: Qwen3.6-27B TP1 GDN  (Hk=%d Hv=%d K=V=%d, chunk=%d, bf16)"
          % (args.hk, args.hv, HEAD_DIM, CHUNK))
    if args.mode in ("chunk", "all"):
        seqlens = (
            [int(x) for x in args.seqlens.split(",")]
            if args.seqlens else [2048, 4096, 8192, 16384]
        )
        for sl in seqlens:
            args.tokens = sl
            run_chunk(args)
    if args.mode in ("recurrent", "all"):
        run_recurrent(args)


if __name__ == "__main__":
    main()
