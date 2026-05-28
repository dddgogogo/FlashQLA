#!/usr/bin/env python3
import argparse

import torch
import torch.nn.functional as F
import triton

from bench_sglang_fused import bench_cuda_ms, make_inputs
from fla.ops.gated_delta_rule.fused_recurrent import (
    fused_recurrent_gated_delta_rule as fla_update,
    fused_recurrent_gated_delta_rule_fwd_kernel as fla_fwd_kernel,
)

from flash_qla.ops.gated_delta_rule.recurrent_fused import (
    fused_sigmoid_gating_delta_rule_update as qla_update,
)


def make_log_decay(a, a_log, dt_bias):
    return -torch.exp(a_log.float()).view(1, 1, -1) * F.softplus(
        a.float() + dt_bias.float().view(1, 1, -1)
    )


def run_fla_direct(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    initial_state,
    cu_seqlens,
    scale,
    use_qk_l2norm,
    use_gate_in_kernel,
    bv,
    num_warps,
):
    batch = initial_state.shape[0]
    h = q.shape[2]
    hv = v.shape[2]
    kdim = q.shape[-1]
    vdim = v.shape[-1]
    bk = triton.next_power_of_2(kdim)
    nv = triton.cdiv(vdim, bv)
    o = torch.empty_like(v)
    ht = q.new_empty(batch, hv, vdim, kdim, dtype=torch.float32)
    fla_fwd_kernel[(nv, batch * hv)](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=None,
        gv=None,
        beta=beta,
        A_log=a_log if use_gate_in_kernel else None,
        dt_bias=dt_bias if use_gate_in_kernel else None,
        o=o,
        h0=initial_state,
        ht=ht,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=1,
        H=h,
        HV=hv,
        K=kdim,
        V=vdim,
        BK=bk,
        BV=bv,
        IS_BETA_HEADWISE=True,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm,
        USE_EXP2=False,
        TRANSPOSE_STATE=True,
        num_warps=num_warps,
        num_stages=3,
    )
    return o, ht


def autotune_fla(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    initial_state,
    cu_seqlens,
    scale,
    use_qk_l2norm,
    use_gate_in_kernel,
    bv_candidates,
    warp_candidates,
    warmup,
    iters,
):
    best_ms = float("inf")
    best = (8, 1)
    for bv in bv_candidates:
        if bv > v.shape[-1]:
            continue
        for num_warps in warp_candidates:
            def call():
                return run_fla_direct(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    a_log,
                    dt_bias,
                    initial_state,
                    cu_seqlens,
                    scale,
                    use_qk_l2norm,
                    use_gate_in_kernel,
                    bv,
                    num_warps,
                )

            ms = bench_cuda_ms(call, warmup, iters)
            if ms < best_ms:
                best_ms = ms
                best = (bv, num_warps)
    return best, best_ms


def run_case(
    batch,
    h,
    hv,
    kdim,
    vdim,
    warmup,
    iters,
    beta_mode,
    gate_mode,
    qk_mode,
    fla_autotune,
    fla_bv_candidates,
    fla_warp_candidates,
    fla_autotune_warmup,
    fla_autotune_iters,
):
    torch.manual_seed(20260430 + batch * 17 + 1)
    q, k, v, a, b, a_log, dt_bias, state0, indices, cu_seqlens = make_inputs(
        batch, 1, h, hv, kdim, vdim
    )
    scale = kdim**-0.5
    use_qk_l2norm = qk_mode == "raw"
    if qk_mode == "precomputed":
        q = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
        k = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)
    beta = b.sigmoid()
    g = make_log_decay(a, a_log, dt_bias)
    qla_a = g if gate_mode == "precomputed" else a
    qla_a_is_log_decay = gate_mode == "precomputed"
    qla_b = beta if beta_mode == "precomputed" else b
    qla_b_is_beta = beta_mode == "precomputed"
    fla_beta = beta if beta_mode == "precomputed" else b.sigmoid()
    fla_gate_in_kernel = gate_mode == "raw"
    fla_g = a if gate_mode == "raw" else g

    fla_state = state0.clone()
    qla_state = state0.clone()
    fla_out, fla_final_state = fla_update(
        q,
        k,
        v,
        g=fla_g,
        beta=fla_beta,
        scale=scale,
        initial_state=fla_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        use_gate_in_kernel=fla_gate_in_kernel,
        A_log=a_log if fla_gate_in_kernel else None,
        dt_bias=dt_bias if fla_gate_in_kernel else None,
        cu_seqlens=cu_seqlens,
        transpose_state_layout=True,
    )
    qla_out = qla_update(
        a_log,
        qla_a,
        dt_bias,
        q,
        k,
        v,
        qla_b,
        qla_state,
        None,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        cu_seqlens=cu_seqlens,
        assume_regular=True,
        b_is_beta=qla_b_is_beta,
        a_is_log_decay=qla_a_is_log_decay,
    )
    torch.cuda.synchronize()
    o_diff = (fla_out.float() - qla_out.float()).abs().max().item()
    state_diff = (fla_final_state.float() - qla_state.float()).abs().max().item()

    bench_fla_state = state0.clone()
    bench_qla_state = state0.clone()

    selected_fla = None
    if fla_autotune:
        selected_fla, tune_ms = autotune_fla(
            q,
            k,
            v,
            fla_g,
            fla_beta,
            a_log,
            dt_bias,
            bench_fla_state,
            cu_seqlens,
            scale,
            use_qk_l2norm,
            fla_gate_in_kernel,
            fla_bv_candidates,
            fla_warp_candidates,
            fla_autotune_warmup,
            fla_autotune_iters,
        )
        print(
            f"  FLA autotune B={batch}: BV={selected_fla[0]} "
            f"warps={selected_fla[1]} tune_ms={tune_ms:.4f}",
            flush=True,
        )

    def fla_call():
        call_beta = beta if beta_mode == "precomputed" else b.sigmoid()
        if selected_fla is not None:
            return run_fla_direct(
                q,
                k,
                v,
                fla_g,
                call_beta,
                a_log,
                dt_bias,
                bench_fla_state,
                cu_seqlens,
                scale,
                use_qk_l2norm,
                fla_gate_in_kernel,
                selected_fla[0],
                selected_fla[1],
            )
        return fla_update(
            q,
            k,
            v,
            g=fla_g,
            beta=call_beta,
            scale=scale,
            initial_state=bench_fla_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            use_gate_in_kernel=fla_gate_in_kernel,
            A_log=a_log if fla_gate_in_kernel else None,
            dt_bias=dt_bias if fla_gate_in_kernel else None,
            cu_seqlens=cu_seqlens,
            transpose_state_layout=True,
        )

    def qla_call():
        return qla_update(
            a_log,
            qla_a,
            dt_bias,
            q,
            k,
            v,
            qla_b,
            bench_qla_state,
            None,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            cu_seqlens=cu_seqlens,
            assume_regular=True,
            b_is_beta=qla_b_is_beta,
            a_is_log_decay=qla_a_is_log_decay,
        )

    fla_ms = bench_cuda_ms(fla_call, warmup, iters)
    qla_ms = bench_cuda_ms(qla_call, warmup, iters)
    print(
        f"decode B={batch:3d} FLA={fla_ms:.4f} ms "
        f"FlashQLA={qla_ms:.4f} ms speedup={fla_ms / qla_ms:.3f}x "
        f"o_diff={o_diff:.3g} state_diff={state_diff:.3g}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--hv", type=int, default=48)
    parser.add_argument("--kdim", type=int, default=128)
    parser.add_argument("--vdim", type=int, default=128)
    parser.add_argument("--decode-batches", default="1,2,4,8,16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--fla-autotune", action="store_true")
    parser.add_argument("--fla-autotune-warmup", type=int, default=20)
    parser.add_argument("--fla-autotune-iters", type=int, default=100)
    parser.add_argument("--fla-bv-candidates", default="4,8,16,32,64,128")
    parser.add_argument("--fla-warp-candidates", default="1,2,4,8")
    parser.add_argument(
        "--beta-mode",
        choices=("raw", "precomputed"),
        default="raw",
        help="raw compares the full raw-b GDN semantic; precomputed gives both kernels beta directly.",
    )
    parser.add_argument(
        "--gate-mode",
        choices=("raw", "precomputed"),
        default="raw",
        help="raw lets both kernels compute decay internally; precomputed passes log-decay g directly.",
    )
    parser.add_argument(
        "--qk-mode",
        choices=("raw", "precomputed"),
        default="raw",
        help="raw lets both kernels normalize q/k internally; precomputed normalizes q/k before timing.",
    )
    args = parser.parse_args()

    print(f"device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
    print(f"shape H={args.h} HV={args.hv} K={args.kdim} V={args.vdim}")
    print(f"beta_mode={args.beta_mode}")
    print(f"gate_mode={args.gate_mode}")
    print(f"qk_mode={args.qk_mode}")
    if args.fla_autotune:
        print(
            f"fla_autotune=on bv={args.fla_bv_candidates} "
            f"warps={args.fla_warp_candidates}"
        )
    fla_bv_candidates = [int(x) for x in args.fla_bv_candidates.split(",") if x]
    fla_warp_candidates = [int(x) for x in args.fla_warp_candidates.split(",") if x]
    for batch in [int(x) for x in args.decode_batches.split(",") if x]:
        run_case(
            batch,
            args.h,
            args.hv,
            args.kdim,
            args.vdim,
            args.warmup,
            args.iters,
            args.beta_mode,
            args.gate_mode,
            args.qk_mode,
            args.fla_autotune,
            fla_bv_candidates,
            fla_warp_candidates,
            args.fla_autotune_warmup,
            args.fla_autotune_iters,
        )


if __name__ == "__main__":
    main()
