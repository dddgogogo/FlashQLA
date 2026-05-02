# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch

from flash_qla.utils import l2norm
from flash_qla.ops.utils import chunk_local_cumsum, group_reduce_vector
from . import tilelang_compat as _tilelang_compat  # noqa: F401
from .arch import is_sm12x, is_sm90

if is_sm90():
    from .hopper import fused_gdr_fwd, fused_gdr_bwd, fused_gdr_h, kkt_solve
elif is_sm12x():
    from .blackwell import (
        fused_gdr_fwd,
        fused_gdr_bwd,
        fused_gdr_h,
        is_fla2_bwd_available,
        kkt_solve,
    )
else:
    raise ValueError("FlashQLA now supports sm90 and sm12x only.")
from .cp_context import intra_card_cp_preprocess


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    auto_cp: bool = True,
    output_v_new: bool = False,
    transpose_state_layout: bool = False,
):
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    kkt_kwargs = dict(k=k, b=beta, cu_seqlens=cu_seqlens)
    if is_sm12x():
        kkt_kwargs["g"] = g
    A = kkt_solve(**kkt_kwargs)
    cp_seq_map = None
    raw_cu_seqlens = None
    if auto_cp:
        initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = (
            intra_card_cp_preprocess(
                k=k,
                v=v,
                a=A,
                g=g,
                b=beta,
                raw_h0=initial_state,
                raw_cu_seqlens=cu_seqlens,
            )
        )
    fwd_kwargs = dict(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        output_h=output_h,
        output_o=True,
        cu_seqlens=cu_seqlens,
        cp_seq_map=cp_seq_map,
        raw_cu_seqlens=raw_cu_seqlens,
        transpose_state_layout=transpose_state_layout,
    )
    if is_sm12x():
        fwd_kwargs["output_v_new"] = output_v_new
    elif output_v_new:
        raise ValueError("output_v_new is only supported on sm12x.")
    fwd_result = fused_gdr_fwd(**fwd_kwargs)
    if output_v_new:
        o, h, final_state, v_new = fwd_result
    else:
        o, h, final_state = fwd_result
        v_new = None
    if output_v_new:
        return g, A, o, h, final_state, v_new
    return g, A, o, h, final_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    h: torch.Tensor | None = None,
    v_new: torch.Tensor | None = None,
    transpose_state_layout: bool = False,
):
    internal_initial_state = initial_state
    internal_dht = dht
    if transpose_state_layout:
        if initial_state is not None:
            internal_initial_state = initial_state.transpose(-1, -2).contiguous()
        if dht is not None:
            internal_dht = dht.transpose(-1, -2).contiguous()

    if is_sm12x():
        dq, dk, dv, db, dg, dh0 = fused_gdr_bwd(
            q,
            k,
            v,
            A,
            g,
            beta,
            do,
            internal_dht,
            h,
            scale=scale,
            cu_seqlens=cu_seqlens,
            initial_state=internal_initial_state,
            v_new=v_new,
        )
        if transpose_state_layout and dh0 is not None:
            dh0 = dh0.transpose(-1, -2).contiguous()
        return dq, dk, dv, db, dg, dh0
    else:
        h, _, _ = fused_gdr_h(
            k=k,
            v=v,
            a=A,
            g=g,
            b=beta,
            initial_state=internal_initial_state,
            output_final_state=False,
            output_h=True,
            cu_seqlens=cu_seqlens,
        )
        dq, dk, dv, dg, db, dh0 = fused_gdr_bwd(
            q=q,
            k=k,
            v=v,
            a=A,
            g=g,
            b=beta,
            do=do,
            dht=internal_dht,
            h=h,
            scale=scale,
            cu_seqlens=cu_seqlens,
    )
    Hg, H = k.shape[-2], v.shape[-2]
    if Hg < H and not is_sm12x():
        dq = group_reduce_vector(dq, Hg)
        dk = group_reduce_vector(dk, Hg)
    assert dg.dtype == torch.float32, "dg should be fp32"
    if not is_sm12x():
        dg = chunk_local_cumsum(
            dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens
        )
    if transpose_state_layout and dh0 is not None:
        dh0 = dh0.transpose(-1, -2).contiguous()
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        transpose_state_layout: bool = False,
    ):
        q_orig = q
        k_orig = k

        save_bwd_intermediates = is_sm12x() and is_fla2_bwd_available()
        fwd_result = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            output_h=save_bwd_intermediates,
            cu_seqlens=cu_seqlens,
            output_v_new=save_bwd_intermediates,
            transpose_state_layout=transpose_state_layout,
        )
        if save_bwd_intermediates:
            g, A, o, h, final_state, v_new = fwd_result
        else:
            g, A, o, h, final_state = fwd_result
            v_new = None

        ctx.save_for_backward(
            q_orig, k_orig, v, g, beta, A, initial_state, cu_seqlens, h, v_new
        )
        ctx.scale = scale
        ctx.transpose_state_layout = transpose_state_layout
        return o.to(q.dtype), final_state

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q_orig, k_orig, v, g, beta, A, initial_state, cu_seqlens, h, v_new = (
            ctx.saved_tensors
        )

        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q_orig,
            k=k_orig,
            v=v,
            g=g,
            beta=beta,
            A=A,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            h=h,
            v_new=v_new,
            transpose_state_layout=ctx.transpose_state_layout,
        )

        return (
            dq.to(q_orig),
            dk.to(k_orig),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
    transpose_state_layout: bool = False,
):
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16 or float16."
    )
    assert not head_first, "head_first=True is not supported."
    assert v.shape[2] % k.shape[2] == 0, (
        "num_qk_heads must be divisible to num_v_heads."
    )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        transpose_state_layout,
    )

    return o, final_state
