# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
#
# Pure-TileLang sm12x (Blackwell sm120/sm121) GDN chunked-backward stage kernels.
#
# These complete the decomposed sm12x backward so it never calls FLA Triton.
# The other stages (recompute_w, dqkwg) already have TileLang kernels in
# fused_bwd.py (tilelang_recompute_w_fwd, tilelang_dqkwg_gqa_sm12x); this module
# adds the stages that previously fell back to FLA:
#   * dv_local        : intra-chunk dv         (ref_gdr.torch_chunk_dv_bwd)
#   * dh recurrence   : reverse scan -> dh, dv  (ref_gdr.torch_chunk_gdr_bwd)
#   * wy_repr_bwd     : WY backward            (ref_gdr.torch_chunk_wy_bwd)
# Every kernel uses token-layout inputs [B,T,H,D] and the chunked-state layout
# [B,num_chunks,HV,DK,DV], matching tilelang_dqkwg_gqa_sm12x. They are tiny in
# shared memory (each well under the sm12x 99 KB cap).

import torch
import tilelang
import tilelang.language as T

from flash_qla.utils import prepare_chunk_indices

_JIT = dict(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        # Disable TMA lowering: TMA tensor-map descriptors (e.g. a `dh_desc`
        # arg) are not representable in the kernel-jit manifest ABI and break
        # NVRTC host-launcher export. The other GDN kernels disable it too.
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)


@tilelang.jit(**_JIT)
def tilelang_dv_local_sm12x(
    H,
    HV,
    DK,
    DV,
    chunk_size,
    scale,
    qkva_dtype,
    g_dtype,
    accum_dtype,
    is_varlen=False,
):
    """Intra-chunk dv. Mirrors ref_gdr.torch_chunk_dv_bwd.

    For each chunk and value-head bhv (key-head bh = bhv // (HV//H)):
        P[c,d] = (scale * q[c] . k[d]) * exp(g[c]-g[d])   for d <= c, else 0
        dv[d]  = sum_c P[c,d] * do_grad[c]        i.e. dv = P^T @ do_grad
    """
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    group_size = HV // H
    C = chunk_size

    q_shape = (batch_size, num_tokens, H, DK)
    v_shape = (batch_size, num_tokens, HV, DV)
    g_shape = (batch_size, num_tokens, HV)

    @T.macro
    def kernel_body(bb, bh, left, seq_end, q, k, g, do_grad, dv):
        q_shared = T.alloc_shared((C, DK), dtype=qkva_dtype)
        k_shared = T.alloc_shared((C, DK), dtype=qkva_dtype)
        do_shared = T.alloc_shared((C, DV), dtype=qkva_dtype)
        p_shared = T.alloc_shared((C, C), dtype=qkva_dtype)
        g_shared = T.alloc_shared((C,), dtype=accum_dtype, scope="shared")
        p_frag = T.alloc_fragment((C, C), dtype=accum_dtype)
        dv_frag = T.alloc_fragment((C, DV), dtype=accum_dtype)

        for j_s, j_k in T.Parallel(C, DK):
            q_shared[j_s, j_k] = T.if_then_else(
                left + j_s < seq_end, q[bb, left + j_s, bh, j_k], T.cast(0, qkva_dtype)
            )
            k_shared[j_s, j_k] = T.if_then_else(
                left + j_s < seq_end, k[bb, left + j_s, bh, j_k], T.cast(0, qkva_dtype)
            )
        for i_g in T.serial(group_size):
            bhv = bh * group_size + i_g
            for j_s in T.Parallel(C):
                g_shared[j_s] = T.if_then_else(
                    left + j_s < seq_end, g[bb, left + j_s, bhv], T.cast(0, accum_dtype)
                )
            for j_s, j_v in T.Parallel(C, DV):
                do_shared[j_s, j_v] = T.if_then_else(
                    left + j_s < seq_end, do_grad[bb, left + j_s, bhv, j_v],
                    T.cast(0, qkva_dtype)
                )
            T.gemm_v1(q_shared, k_shared, p_frag, transpose_B=True, clear_accum=True)
            for c, d in T.Parallel(C, C):
                if d <= c:
                    p_shared[c, d] = (
                        p_frag[c, d] * scale * T.exp(g_shared[c] - g_shared[d])
                    ).astype(qkva_dtype)
                else:
                    p_shared[c, d] = T.cast(0, qkva_dtype)
            T.gemm_v1(p_shared, do_shared, dv_frag, transpose_A=True, clear_accum=True)
            for d, j_v in T.Parallel(C, DV):
                if left + d < seq_end:
                    dv[bb, left + d, bhv, j_v] = dv_frag[d, j_v].astype(qkva_dtype)

    if is_varlen:
        num_chunks = T.dynamic("num_chunks")

        @T.prim_func
        def kernel(
            q: T.Tensor((1, num_tokens, H, DK), dtype=qkva_dtype),
            k: T.Tensor((1, num_tokens, H, DK), dtype=qkva_dtype),
            g: T.Tensor((1, num_tokens, HV), dtype=g_dtype),
            do_grad: T.Tensor((1, num_tokens, HV, DV), dtype=qkva_dtype),
            cu_seqlens: T.Tensor([batch_size + 1], dtype="int32"),
            chunk_indices: T.Tensor([num_chunks, 2], dtype="int32"),
            dv: T.Tensor((1, num_tokens, HV, DV), dtype=qkva_dtype),
        ):
            with T.Kernel(num_chunks, H, threads=128) as (bc, bh):
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start = cu_seqlens[batch_idx]
                seq_end = cu_seqlens[batch_idx + 1]
                kernel_body(0, bh, seq_start + chunk_idx * C, seq_end, q, k, g, do_grad, dv)
    else:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_shape, dtype=qkva_dtype),
            k: T.Tensor(q_shape, dtype=qkva_dtype),
            g: T.Tensor(g_shape, dtype=g_dtype),
            do_grad: T.Tensor(v_shape, dtype=qkva_dtype),
            dv: T.Tensor(v_shape, dtype=qkva_dtype),
        ):
            with T.Kernel(
                tilelang.cdiv(num_tokens, chunk_size), batch_size * H, threads=128
            ) as (bc, bbh):
                bb = bbh // H
                bh = bbh % H
                kernel_body(bb, bh, bc * C, num_tokens, q, k, g, do_grad, dv)

    return kernel


def chunk_dv_local_sm12x(q, k, g, do_grad, scale, chunk_size=64, cu_seqlens=None):
    """Host wrapper for tilelang_dv_local_sm12x. Token-layout inputs.
    Varlen: q/k/g/do_grad packed as [1, total_tokens, ...]; cu_seqlens [n_seq+1] int32."""
    batch_size, num_tokens, num_key_heads, head_dim = k.shape
    num_value_heads = do_grad.shape[2]
    value_dim = do_grad.shape[-1]
    dv = torch.empty_like(do_grad)
    is_varlen = cu_seqlens is not None
    kernel = tilelang_dv_local_sm12x(
        H=num_key_heads,
        HV=num_value_heads,
        DK=head_dim,
        DV=value_dim,
        chunk_size=chunk_size,
        scale=scale,
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        accum_dtype="float32",
        is_varlen=is_varlen,
    )
    if is_varlen:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size).to(torch.int32)
        kernel(q, k, g, do_grad, cu_seqlens.to(torch.int32), chunk_indices, dv)
    else:
        kernel(q, k, g, do_grad, dv)
    return dv


@tilelang.jit(**_JIT)
def tilelang_dhu_sm12x(
    H,
    HV,
    DK,
    DV,
    chunk_size,
    scale,
    qkva_dtype,
    g_dtype,
    accum_dtype,
    block_V=32,
    is_varlen=False,
    use_dht=False,
):
    """Reverse dh-recurrence + dv update. Mirrors ref_gdr.torch_chunk_gdr_bwd.

    use_dht=True (non-varlen only) seeds the reverse recurrence from dht
    ([B,HV,DK,DV], the gradient w.r.t. the final state) instead of zero and
    writes the resulting dh0 ([B,HV,DK,DV]) — this keeps the dht!=None case on
    the pure-TileLang path (no FLA, no precision mixing).

    One block per (batch/sequence, value-head, V-tile); reverse scan over the
    chunks of that sequence, holding the per-tile state dstate[DK, block_V] in
    shared fp32. The recurrence resets per sequence (dstate cleared at the
    start), so packed sequences never bleed state across the boundary.

    Partial last chunk: all token tiles load 0 for out-of-range rows, and
    ``g_last`` is taken at the last VALID token — this matches the reference's
    ``fill_last_chunk_of_g`` (padded gate positions = last-valid gate), so the
    chunk decay ``dstate*exp(g_last)`` and ``exp(g_last-g)`` are correct.
    """
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    group_size = HV // H
    C = chunk_size

    q_shape = (batch_size, num_tokens, H, DK)
    v_shape = (batch_size, num_tokens, HV, DV)
    w_shape = (batch_size, num_tokens, HV, DK)
    g_shape = (batch_size, num_tokens, HV)
    # dh is written as a 4D tensor [B, num_chunks*HV, DK, DV]. A 5D *output*
    # tensor breaks NVRTC host-launcher generation (5D inputs are fine — see
    # dqkwg); the 4D view is the same row-major memory, so downstream dqkwg
    # still reads it as [B, num_chunks, HV, DK, DV]. The chunk dim is derived
    # from num_tokens (no standalone num_chunks dynamic).
    dh_shape = (batch_size, tilelang.cdiv(num_tokens, chunk_size) * HV, DK, DV)

    @T.macro
    def kernel_body(bb, bhv, v_off, seq_start, seq_end, chunk_base, nchunks,
                    q, k, w, g, do_grad, dv, dh, dht, dh0):
        bh = bhv // group_size

        q_shared = T.alloc_shared((C, DK), dtype=qkva_dtype)
        k_shared = T.alloc_shared((C, DK), dtype=qkva_dtype)
        w_shared = T.alloc_shared((C, DK), dtype=qkva_dtype)
        do_shared = T.alloc_shared((C, block_V), dtype=qkva_dtype)
        dv_shared = T.alloc_shared((C, block_V), dtype=qkva_dtype)
        kc_shared = T.alloc_shared((C, DK), dtype=qkva_dtype)
        state_bf = T.alloc_shared((DK, block_V), dtype=qkva_dtype)
        g_shared = T.alloc_shared((C,), dtype=accum_dtype, scope="shared")

        dstate = T.alloc_fragment((DK, block_V), dtype=accum_dtype)
        ddv = T.alloc_fragment((C, block_V), dtype=accum_dtype)
        dvf = T.alloc_fragment((C, block_V), dtype=accum_dtype)
        inter = T.alloc_fragment((DK, block_V), dtype=accum_dtype)
        wdv = T.alloc_fragment((DK, block_V), dtype=accum_dtype)
        g_last = T.alloc_var("float32")
        last_local = T.alloc_var("int32")

        if use_dht:
            # seed from dht ([B,HV,DK,DV]) so dht!=None stays on the pure path
            for j_k, j_v in T.Parallel(DK, block_V):
                dstate[j_k, j_v] = T.cast(dht[bb, bhv, j_k, v_off + j_v], "float32")
        else:
            T.clear(dstate)

        for i_rev in T.serial(nchunks):
            bc = nchunks - 1 - i_rev          # chunk-local index within sequence
            left = seq_start + bc * C
            grow = (chunk_base + bc) * HV + bhv  # packed dh row
            last_local = T.min(C - 1, seq_end - left - 1)

            # store dh[bc] = current dstate (pre-update); dh is 4D
            # [B, num_chunks*HV, DK, DV].
            T.copy(dstate, state_bf)
            T.copy(state_bf, dh[bb, grow, 0:DK, v_off : v_off + block_V])

            # load chunk tiles (masked: out-of-range rows -> 0)
            for c, j_k in T.Parallel(C, DK):
                q_shared[c, j_k] = T.if_then_else(
                    left + c < seq_end, q[bb, left + c, bh, j_k], T.cast(0, qkva_dtype))
                k_shared[c, j_k] = T.if_then_else(
                    left + c < seq_end, k[bb, left + c, bh, j_k], T.cast(0, qkva_dtype))
                w_shared[c, j_k] = T.if_then_else(
                    left + c < seq_end, w[bb, left + c, bhv, j_k], T.cast(0, qkva_dtype))
            for c, j_v in T.Parallel(C, block_V):
                do_shared[c, j_v] = T.if_then_else(
                    left + c < seq_end, do_grad[bb, left + c, bhv, v_off + j_v],
                    T.cast(0, qkva_dtype))
                dv_shared[c, j_v] = T.if_then_else(
                    left + c < seq_end, dv[bb, left + c, bhv, v_off + j_v],
                    T.cast(0, qkva_dtype))
            for j_s in T.Parallel(C):
                g_shared[j_s] = T.if_then_else(
                    left + j_s < seq_end, g[bb, left + j_s, bhv], T.cast(0, accum_dtype))
            g_last = g_shared[last_local]  # last VALID token's gate (partial-safe)

            # dv[bc] += (k * exp(g_last - g)) @ dstate
            for c, j_k in T.Parallel(C, DK):
                kc_shared[c, j_k] = (
                    k_shared[c, j_k] * T.exp(g_last - g_shared[c])
                ).astype(qkva_dtype)
            T.copy(dstate, state_bf)
            T.gemm_v1(kc_shared, state_bf, ddv, clear_accum=True)
            for c, j_v in T.Parallel(C, block_V):
                dvf[c, j_v] = T.cast(dv_shared[c, j_v], "float32") + ddv[c, j_v]
                dv_shared[c, j_v] = dvf[c, j_v].astype(qkva_dtype)
            for c, j_v in T.Parallel(C, block_V):
                if left + c < seq_end:
                    dv[bb, left + c, bhv, v_off + j_v] = dv_shared[c, j_v]

            # inter = (q * scale * exp(g))^T @ do_grad
            for c, j_k in T.Parallel(C, DK):
                kc_shared[c, j_k] = (
                    q_shared[c, j_k] * scale * T.exp(g_shared[c])
                ).astype(qkva_dtype)
            T.gemm_v1(kc_shared, do_shared, inter, transpose_A=True, clear_accum=True)

            # wdv = w^T @ dv_updated
            T.gemm_v1(w_shared, dv_shared, wdv, transpose_A=True, clear_accum=True)

            # dstate = dstate * exp(g_last) + inter - wdv
            for j_k, j_v in T.Parallel(DK, block_V):
                dstate[j_k, j_v] = (
                    dstate[j_k, j_v] * T.exp(g_last)
                    + inter[j_k, j_v]
                    - wdv[j_k, j_v]
                )

        if use_dht:
            # dh0 = final dstate after processing chunk 0 (the last reverse step)
            for j_k, j_v in T.Parallel(DK, block_V):
                dh0[bb, bhv, j_k, v_off + j_v] = dstate[j_k, j_v].astype(qkva_dtype)

    state0_shape = (batch_size, HV, DK, DV)

    if is_varlen:
        num_chunks = T.dynamic("num_chunks")

        @T.prim_func
        def kernel(
            q: T.Tensor((1, num_tokens, H, DK), dtype=qkva_dtype),
            k: T.Tensor((1, num_tokens, H, DK), dtype=qkva_dtype),
            w: T.Tensor((1, num_tokens, HV, DK), dtype=qkva_dtype),
            g: T.Tensor((1, num_tokens, HV), dtype=g_dtype),
            do_grad: T.Tensor((1, num_tokens, HV, DV), dtype=qkva_dtype),
            cu_seqlens: T.Tensor([batch_size + 1], dtype="int32"),
            chunk_offsets: T.Tensor([batch_size + 1], dtype="int32"),
            dv: T.Tensor((1, num_tokens, HV, DV), dtype=qkva_dtype),
            dh: T.Tensor((1, num_chunks * HV, DK, DV), dtype=qkva_dtype),
        ):
            with T.Kernel(
                tilelang.cdiv(DV, block_V), batch_size * HV, threads=128
            ) as (bv, bbh):
                batch_idx = bbh // HV
                bhv = bbh % HV
                seq_start = cu_seqlens[batch_idx]
                seq_end = cu_seqlens[batch_idx + 1]
                chunk_base = chunk_offsets[batch_idx]
                nchunks = T.alloc_var("int32")
                nchunks = T.ceildiv(seq_end - seq_start, C)
                # dh dummies for dht/dh0 (use_dht=False — never indexed)
                kernel_body(0, bhv, bv * block_V, seq_start, seq_end, chunk_base,
                            nchunks, q, k, w, g, do_grad, dv, dh, dh, dh)
    elif use_dht:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_shape, dtype=qkva_dtype),
            k: T.Tensor(q_shape, dtype=qkva_dtype),
            w: T.Tensor(w_shape, dtype=qkva_dtype),
            g: T.Tensor(g_shape, dtype=g_dtype),
            do_grad: T.Tensor(v_shape, dtype=qkva_dtype),
            dht: T.Tensor(state0_shape, dtype=qkva_dtype),
            dv: T.Tensor(v_shape, dtype=qkva_dtype),
            dh: T.Tensor(dh_shape, dtype=qkva_dtype),
            dh0: T.Tensor(state0_shape, dtype=qkva_dtype),
        ):
            with T.Kernel(
                tilelang.cdiv(DV, block_V), batch_size * HV, threads=128
            ) as (bv, bbh):
                bb = bbh // HV
                bhv = bbh % HV
                nchunks = T.alloc_var("int32")
                nchunks = T.ceildiv(num_tokens, C)
                kernel_body(bb, bhv, bv * block_V, 0, num_tokens, 0, nchunks,
                            q, k, w, g, do_grad, dv, dh, dht, dh0)
    else:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_shape, dtype=qkva_dtype),
            k: T.Tensor(q_shape, dtype=qkva_dtype),
            w: T.Tensor(w_shape, dtype=qkva_dtype),
            g: T.Tensor(g_shape, dtype=g_dtype),
            do_grad: T.Tensor(v_shape, dtype=qkva_dtype),
            dv: T.Tensor(v_shape, dtype=qkva_dtype),
            dh: T.Tensor(dh_shape, dtype=qkva_dtype),
        ):
            with T.Kernel(
                tilelang.cdiv(DV, block_V), batch_size * HV, threads=128
            ) as (bv, bbh):
                bb = bbh // HV
                bhv = bbh % HV
                # Loop bound as a local int32 (from num_tokens), not the dynamic
                # `num_chunks` symbolic: a dynamic serial-loop bound makes the
                # NVRTC host launcher emit an untyped scalar arg the exporter
                # cannot type. chunk_base=0 (dh indexed by bb directly).
                nchunks = T.alloc_var("int32")
                nchunks = T.ceildiv(num_tokens, C)
                # dh dummies for dht/dh0 (use_dht=False — never indexed)
                kernel_body(bb, bhv, bv * block_V, 0, num_tokens, 0, nchunks,
                            q, k, w, g, do_grad, dv, dh, dh, dh)

    return kernel


def chunk_dhu_sm12x(q, k, w, g, do_grad, dv, scale, chunk_size=64, block_V=32,
                    cu_seqlens=None, dht=None):
    """Host wrapper for tilelang_dhu_sm12x. Returns (dh, dv_updated, dh0).

    Production case: dht=None → dh0 is None (recurrence seeded from zero).
    When dht ([B,HV,DK,DV]) is given, the recurrence is seeded from it and dh0
    ([B,HV,DK,DV]) is returned — the pure-TileLang dht!=None path (no FLA).
    `dv` is updated out-of-place into a fresh tensor. For varlen, dh is packed
    [1, total_chunks, HV, DK, DV] (chunk-major per sequence, prepare_chunk_offsets
    ordering); dht is not supported with varlen.
    """
    batch_size, num_tokens, num_key_heads, head_dim = k.shape
    num_value_heads = do_grad.shape[2]
    value_dim = do_grad.shape[-1]
    is_varlen = cu_seqlens is not None
    use_dht = dht is not None
    if use_dht and is_varlen:
        raise NotImplementedError("dht is not supported together with cu_seqlens")
    if is_varlen:
        from flash_qla.utils import prepare_chunk_offsets
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size).to(torch.int32)
        num_chunks = int(chunk_offsets[-1].item())
    else:
        num_chunks = (num_tokens + chunk_size - 1) // chunk_size
    dh = q.new_empty(batch_size, num_chunks, num_value_heads, head_dim, value_dim)
    # kernel writes dh as a 4D view [B, num_chunks*HV, DK, DV] (same memory);
    # callers consume the 5D dh.
    dh4 = dh.view(batch_size, num_chunks * num_value_heads, head_dim, value_dim)
    dv_out = dv.clone()
    kernel = tilelang_dhu_sm12x(
        H=num_key_heads,
        HV=num_value_heads,
        DK=head_dim,
        DV=value_dim,
        chunk_size=chunk_size,
        scale=scale,
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        accum_dtype="float32",
        block_V=block_V,
        is_varlen=is_varlen,
        use_dht=use_dht,
    )
    dh0 = None
    if is_varlen:
        kernel(q, k, w, g, do_grad, cu_seqlens.to(torch.int32), chunk_offsets, dv_out, dh4)
    elif use_dht:
        dh0 = q.new_empty(batch_size, num_value_heads, head_dim, value_dim)
        kernel(q, k, w, g, do_grad, dht.to(q.dtype), dv_out, dh4, dh0)
    else:
        kernel(q, k, w, g, do_grad, dv_out, dh4)
    return dh, dv_out, dh0


@tilelang.jit(**_JIT)
def tilelang_wy_bwd_sm12x(
    H,
    HV,
    DK,
    DV,
    chunk_size,
    qkva_dtype,
    g_dtype,
    accum_dtype,
    is_varlen=False,
):
    """WY-representation backward. Mirrors ref_gdr.torch_chunk_wy_bwd.

    Per (chunk, key-head): loops group value-heads, accumulating dk into a
    key-head accumulator (GQA) and adding dk1; writes dv/db/dg per value-head.
    All work is on [C,C]/[C,DK]/[C,DV] tiles; one [C,C] shared scratch is reused
    across the chained dA matmuls to stay under the sm121 99 KB cap.

    All token loads/stores are masked by ``left + j < seq_end`` so a partial
    last chunk (varlen, or num_tokens not a multiple of C) is handled correctly:
    out-of-range rows load 0 (g/b=0 → eg=1 but always multiplied by a zeroed
    k/v/dw/du), and every store is guarded.
    """
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    C = chunk_size
    group_size = HV // H

    k_shape = (batch_size, num_tokens, H, DK)
    v_shape = (batch_size, num_tokens, HV, DV)
    w_shape = (batch_size, num_tokens, HV, DK)
    g_shape = (batch_size, num_tokens, HV)
    A_shape = (batch_size, num_tokens, HV, C)

    @T.macro
    def kernel_body(bb, bh, left, seq_end, k, v, beta, Amat, g, dw, du, dk1, dg1,
                    dk, dv, db, dg):
        k_s = T.alloc_shared((C, DK), dtype=qkva_dtype)
        A_s = T.alloc_shared((C, C), dtype=qkva_dtype)
        dw_s = T.alloc_shared((C, DK), dtype=qkva_dtype)
        du_s = T.alloc_shared((C, DV), dtype=qkva_dtype)
        kbg_s = T.alloc_shared((C, DK), dtype=qkva_dtype)  # k*beta*eg, reused for k*beta
        scrC = T.alloc_shared((C, C), dtype=qkva_dtype)    # reused: dA_masked, dA1
        g_s = T.alloc_shared((C,), dtype=accum_dtype, scope="shared")
        b_s = T.alloc_shared((C,), dtype=accum_dtype, scope="shared")
        eg_s = T.alloc_shared((C,), dtype=accum_dtype, scope="shared")

        acc_dk = T.alloc_fragment((C, DK), dtype=accum_dtype)
        dA = T.alloc_fragment((C, C), dtype=accum_dtype)
        dA1f = T.alloc_fragment((C, C), dtype=accum_dtype)
        dA2f = T.alloc_fragment((C, C), dtype=accum_dtype)
        A2f = T.alloc_fragment((C, C), dtype=accum_dtype)
        dkbg = T.alloc_fragment((C, DK), dtype=accum_dtype)
        dvb = T.alloc_fragment((C, DV), dtype=accum_dtype)
        dk_v = T.alloc_fragment((C, DK), dtype=accum_dtype)
        tmpDK = T.alloc_fragment((C, DK), dtype=accum_dtype)
        db_v = T.alloc_fragment((C,), dtype=accum_dtype)
        dg_v = T.alloc_fragment((C,), dtype=accum_dtype)
        rowv = T.alloc_fragment((C,), dtype=accum_dtype)
        prodDK = T.alloc_fragment((C, DK), dtype=accum_dtype)
        prodCC = T.alloc_fragment((C, C), dtype=accum_dtype)
        rcol = T.alloc_fragment((C,), dtype=accum_dtype)

        for d, kk in T.Parallel(C, DK):
            k_s[d, kk] = T.if_then_else(
                left + d < seq_end, k[bb, left + d, bh, kk], T.cast(0, qkva_dtype))
        T.clear(acc_dk)

        for i_g in T.serial(group_size):
            bhv = bh * group_size + i_g
            for j in T.Parallel(C):
                g_s[j] = T.if_then_else(
                    left + j < seq_end, g[bb, left + j, bhv], T.cast(0, accum_dtype))
                b_s[j] = T.if_then_else(
                    left + j < seq_end, beta[bb, left + j, bhv], T.cast(0, accum_dtype))
                eg_s[j] = T.exp(g_s[j])
            for c, t in T.Parallel(C, C):
                A_s[c, t] = T.if_then_else(
                    left + c < seq_end, Amat[bb, left + c, bhv, t], T.cast(0, qkva_dtype))
            for d, kk in T.Parallel(C, DK):
                dw_s[d, kk] = T.if_then_else(
                    left + d < seq_end, dw[bb, left + d, bhv, kk], T.cast(0, qkva_dtype))
            for d, jv in T.Parallel(C, DV):
                du_s[d, jv] = T.if_then_else(
                    left + d < seq_end, du[bb, left + d, bhv, jv], T.cast(0, qkva_dtype))

            # kbg[d,kk] = k[d,kk]*beta[d]*eg[d]
            for d, kk in T.Parallel(C, DK):
                kbg_s[d, kk] = (k_s[d, kk] * (b_s[d] * eg_s[d])).astype(qkva_dtype)

            # dA[c,d] = dw @ kbg^T ; dA += du @ vb^T  (vb=v*beta)
            T.gemm_v1(dw_s, kbg_s, dA, transpose_B=True, clear_accum=True)
            # vb stored into kbg_s (kbg_s is DK-sized; vb is DV-sized, both 128).
            for d, jv in T.Parallel(C, DV):
                kbg_s[d, jv] = (
                    T.cast(T.if_then_else(left + d < seq_end, v[bb, left + d, bhv, jv],
                                          T.cast(0, qkva_dtype)), "float32") * b_s[d]
                ).astype(qkva_dtype)
            T.gemm_v1(du_s, kbg_s, dA, transpose_B=True, clear_accum=False)

            # dk_beta_g[d,kk] = A^T @ dw ; dk_v = dk_beta_g*beta*eg
            T.gemm_v1(A_s, dw_s, dkbg, transpose_A=True, clear_accum=True)
            for d, kk in T.Parallel(C, DK):
                dk_v[d, kk] = dkbg[d, kk] * (b_s[d] * eg_s[d])
            # db[d] = sum_kk dk_beta_g*k*eg ; dg[d] = *beta too
            for d, kk in T.Parallel(C, DK):
                prodDK[d, kk] = dkbg[d, kk] * T.cast(k_s[d, kk], "float32") * eg_s[d]
            T.reduce_sum(prodDK, db_v, dim=1)
            for d in T.Parallel(C):
                dg_v[d] = db_v[d] * b_s[d]

            # dv_beta[d,v] = A^T @ du ; dv = dv_beta*beta ; db += sum_v dv_beta*v
            T.gemm_v1(A_s, du_s, dvb, transpose_A=True, clear_accum=True)
            for d, jv in T.Parallel(C, DV):
                if left + d < seq_end:
                    dv[bb, left + d, bhv, jv] = (dvb[d, jv] * b_s[d]).astype(qkva_dtype)
            for d, jv in T.Parallel(C, DV):
                dvb[d, jv] = dvb[d, jv] * T.cast(
                    T.if_then_else(left + d < seq_end, v[bb, left + d, bhv, jv],
                                   T.cast(0, qkva_dtype)), "float32")
            T.reduce_sum(dvb, rowv, dim=1)
            for d in T.Parallel(C):
                db_v[d] += rowv[d]

            # mask dA strict-lower (keep c>d), then dA1 = A^T@dA, dA2 = dA1@A^T, dA = -dA2*decay(c,e: e>c)
            for c, d in T.Parallel(C, C):
                scrC[c, d] = (dA[c, d] if c > d else T.cast(0, accum_dtype)).astype(qkva_dtype)
            T.gemm_v1(A_s, scrC, dA1f, transpose_A=True, clear_accum=True)  # [c,e]=sum_d A[d,c]*dAm[d,e]
            for c, e in T.Parallel(C, C):
                scrC[c, e] = dA1f[c, e].astype(qkva_dtype)
            T.gemm_v1(scrC, A_s, dA2f, transpose_B=True, clear_accum=True)  # [c,e]=sum_d dA1[c,d]*A[e,d]
            # decay applied to dA[c,e] is exp(g[c]-g[e]) for c>e (strict lower),
            # because the reference swapaxes(-2,-1) moves the Hv axis.
            for c, e in T.Parallel(C, C):
                if c > e:
                    dA[c, e] = -dA2f[c, e] * T.exp(g_s[c] - g_s[e])
                else:
                    dA[c, e] = T.cast(0, accum_dtype)

            # A2[c,d] = (k*beta) @ k^T   (kbg_s currently holds vb; rebuild k*beta)
            for c, kk in T.Parallel(C, DK):
                kbg_s[c, kk] = (k_s[c, kk] * b_s[c]).astype(qkva_dtype)
            T.gemm_v1(kbg_s, k_s, A2f, transpose_B=True, clear_accum=True)

            # dk_beta[c,kk] = dA @ k ; db[c] += sum_kk dk_beta*k
            for c, e in T.Parallel(C, C):
                scrC[c, e] = dA[c, e].astype(qkva_dtype)
            T.gemm_v1(scrC, k_s, tmpDK, clear_accum=True)  # dk_beta[c,kk]
            for c, kk in T.Parallel(C, DK):
                prodDK[c, kk] = tmpDK[c, kk] * T.cast(k_s[c, kk], "float32")
            T.reduce_sum(prodDK, rowv, dim=1)
            for c in T.Parallel(C):
                db_v[c] += rowv[c]
            # dk_v[c,kk] += dk_beta*beta
            for c, kk in T.Parallel(C, DK):
                dk_v[c, kk] += tmpDK[c, kk] * b_s[c]
            # dk_v[d,kk] += dA^T @ (k*beta)   (kbg_s holds k*beta)
            T.gemm_v1(scrC, kbg_s, tmpDK, transpose_A=True, clear_accum=True)
            for d, kk in T.Parallel(C, DK):
                dk_v[d, kk] += tmpDK[d, kk]

            # dg[c] += rowsum_c(dA*A2) - colsum_c(dA*A2)
            for c, e in T.Parallel(C, C):
                prodCC[c, e] = dA[c, e] * A2f[c, e]
            T.reduce_sum(prodCC, rowv, dim=1)   # rowsum over e -> [c]
            # colsum over c via shared transpose then dim=1 reduce (avoids dim=0 reduce)
            for c, e in T.Parallel(C, C):
                scrC[c, e] = prodCC[c, e].astype(qkva_dtype)
            for e, c in T.Parallel(C, C):
                prodCC[e, c] = T.cast(scrC[c, e], "float32")
            T.reduce_sum(prodCC, rcol, dim=1)   # rcol[e] = sum_c M[c,e]
            for c in T.Parallel(C):
                dg_v[c] += rowv[c] - rcol[c]
                dg_v[c] += T.if_then_else(
                    left + c < seq_end, dg1[bb, left + c, bhv], T.cast(0, accum_dtype))

            # write per-value-head db, dg ; accumulate dk into key-head acc
            for c in T.Parallel(C):
                if left + c < seq_end:
                    db[bb, left + c, bhv] = db_v[c]
                    dg[bb, left + c, bhv] = dg_v[c]
            for d, kk in T.Parallel(C, DK):
                acc_dk[d, kk] += dk_v[d, kk]

        # dk = acc_dk + dk1   (key-head)
        for d, kk in T.Parallel(C, DK):
            if left + d < seq_end:
                dk[bb, left + d, bh, kk] = (
                    acc_dk[d, kk] + T.cast(dk1[bb, left + d, bh, kk], "float32")
                ).astype(qkva_dtype)

    if is_varlen:
        num_chunks = T.dynamic("num_chunks")

        @T.prim_func
        def kernel(
            k: T.Tensor((1, num_tokens, H, DK), dtype=qkva_dtype),
            v: T.Tensor((1, num_tokens, HV, DV), dtype=qkva_dtype),
            beta: T.Tensor((1, num_tokens, HV), dtype=g_dtype),
            Amat: T.Tensor((1, num_tokens, HV, C), dtype=qkva_dtype),
            g: T.Tensor((1, num_tokens, HV), dtype=g_dtype),
            dw: T.Tensor((1, num_tokens, HV, DK), dtype=qkva_dtype),
            du: T.Tensor((1, num_tokens, HV, DV), dtype=qkva_dtype),
            dk1: T.Tensor((1, num_tokens, H, DK), dtype=qkva_dtype),
            dg1: T.Tensor((1, num_tokens, HV), dtype=g_dtype),
            cu_seqlens: T.Tensor([batch_size + 1], dtype="int32"),
            chunk_indices: T.Tensor([num_chunks, 2], dtype="int32"),
            dk: T.Tensor((1, num_tokens, H, DK), dtype=qkva_dtype),
            dv: T.Tensor((1, num_tokens, HV, DV), dtype=qkva_dtype),
            db: T.Tensor((1, num_tokens, HV), dtype=g_dtype),
            dg: T.Tensor((1, num_tokens, HV), dtype=g_dtype),
        ):
            with T.Kernel(num_chunks, H, threads=256) as (bc, bh):
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start = cu_seqlens[batch_idx]
                seq_end = cu_seqlens[batch_idx + 1]
                kernel_body(0, bh, seq_start + chunk_idx * C, seq_end,
                            k, v, beta, Amat, g, dw, du, dk1, dg1, dk, dv, db, dg)
    else:
        @T.prim_func
        def kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            v: T.Tensor(v_shape, dtype=qkva_dtype),
            beta: T.Tensor(g_shape, dtype=g_dtype),
            Amat: T.Tensor(A_shape, dtype=qkva_dtype),
            g: T.Tensor(g_shape, dtype=g_dtype),
            dw: T.Tensor(w_shape, dtype=qkva_dtype),
            du: T.Tensor(v_shape, dtype=qkva_dtype),
            dk1: T.Tensor(k_shape, dtype=qkva_dtype),
            dg1: T.Tensor(g_shape, dtype=g_dtype),
            dk: T.Tensor(k_shape, dtype=qkva_dtype),
            dv: T.Tensor(v_shape, dtype=qkva_dtype),
            db: T.Tensor(g_shape, dtype=g_dtype),
            dg: T.Tensor(g_shape, dtype=g_dtype),
        ):
            with T.Kernel(
                tilelang.cdiv(num_tokens, chunk_size), batch_size * H, threads=256
            ) as (bc, bbh):
                kernel_body(bbh // H, bbh % H, bc * C, num_tokens,
                            k, v, beta, Amat, g, dw, du, dk1, dg1, dk, dv, db, dg)

    return kernel


def chunk_wy_bwd_sm12x(k, v, beta, A, g, dw, du, dk1, dg1, chunk_size=64, cu_seqlens=None):
    """Host wrapper for tilelang_wy_bwd_sm12x. Token-layout inputs.

    dk1 is the (already key-head-reduced) dk from dqkwg; dg1 the dg from dqkwg.
    Returns dk (key-head), dv, db, dg (value-head).
    """
    batch_size, num_tokens, num_key_heads, head_dim = k.shape
    num_value_heads = v.shape[2]
    value_dim = v.shape[-1]
    dk = torch.empty_like(dk1)
    dv = torch.empty_like(v)
    db = torch.empty_like(beta)
    dg = torch.empty_like(g)
    is_varlen = cu_seqlens is not None
    kernel = tilelang_wy_bwd_sm12x(
        H=num_key_heads,
        HV=num_value_heads,
        DK=head_dim,
        DV=value_dim,
        chunk_size=chunk_size,
        qkva_dtype=k.dtype,
        g_dtype=g.dtype,
        accum_dtype="float32",
        is_varlen=is_varlen,
    )
    if is_varlen:
        ci = prepare_chunk_indices(cu_seqlens, chunk_size).to(torch.int32)
        kernel(k, v, beta, A, g, dw, du, dk1, dg1, cu_seqlens.to(torch.int32), ci,
               dk, dv, db, dg)
    else:
        kernel(k, v, beta, A, g, dw, du, dk1, dg1, dk, dv, db, dg)
    return dk, dv, db, dg


def chunk_gated_delta_rule_bwd_sm12x(
    q, k, v, g, beta, A, do_grad, h, v_new, scale, chunk_size=64, cu_seqlens=None,
    dht=None, output_dh0=False,
):
    """Fully pure-TileLang sm12x GDN chunked backward (no FLA, no Triton at the
    compute layer). `g` is the chunk-local-cumsum'd gate; `h`/`v_new` come from
    the forward recompute.

    `cu_seqlens` enables packed (varlen) sequences: B=1, h/v_new packed and
    dh internally packed per sequence (prepare_chunk_offsets ordering). Every
    stage resets state at sequence boundaries — no cross-sequence bleed.

    Two independent state-related flags (mirrors the fp32 reference):
      * `dht` ([B,HV,DK,DV], gradient w.r.t. the final state) SEEDS the reverse
        dh-recurrence (else it starts from zero). This keeps the path pure even
        when output_final_state is differentiated.
      * `output_dh0` requests dh0 ([B,HV,DK,DV]) = the gradient w.r.t. the
        initial state. Set it iff an initial_state was supplied; the reference
        returns dh0 only in that case, and autograd must return None for the
        initial_state slot otherwise.
    Neither is supported together with cu_seqlens (varlen).

    Stage pipeline (all TileLang):
      recompute_w -> dv_local -> dh-recurrence -> dqkwg -> wy_bwd -> dg reverse-cumsum
    Returns dq, dk, dv, db, dg, dh0 (dq/dk key-head-reduced for GQA; dh0 None
    unless output_dh0).
    """
    from flash_qla.ops.utils.cumsum import chunk_local_cumsum

    # The dhu kernel's stateful mode (seed + dh0 write) is one path; engage it
    # when we either seed from dht OR need dh0. Seed from a zeros dht when only
    # dh0 is requested (recurrence-from-zero, exactly the reference's h0 case).
    stateful = (dht is not None) or output_dh0
    dht_eff = dht
    if stateful and dht is None:
        dht_eff = torch.zeros(
            q.shape[0], v.shape[2], q.shape[-1], v.shape[-1],
            device=q.device, dtype=q.dtype)

    w = recompute_w_sm12x(k, beta, A, g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    dv = chunk_dv_local_sm12x(q, k, g, do_grad, scale, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    dh, dv, dh0 = chunk_dhu_sm12x(
        q, k, w, g, do_grad, dv, scale, chunk_size=chunk_size, cu_seqlens=cu_seqlens,
        dht=dht_eff)
    dq, dk1, dw, dg1 = chunk_dqkwg_sm12x(
        q, k, v_new, w, g, h, dv, do_grad, dh, scale, chunk_size=chunk_size, cu_seqlens=cu_seqlens
    )
    dk, dv, db, dg = chunk_wy_bwd_sm12x(
        k, v, beta, A, g, dw, dv, dk1, dg1, chunk_size=chunk_size, cu_seqlens=cu_seqlens
    )
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, (dh0 if output_dh0 else None)


# ---------------------------------------------------------------------------
# recompute_w and dqkwg kernels (relocated from fused_bwd.py, triton-free).
# ---------------------------------------------------------------------------


@tilelang.jit(**_JIT)
def tilelang_recompute_w_fwd_sm12x(
    H, HV, DK, chunk_size, accum_dtype, qkva_dtype, g_dtype, b_dtype, is_varlen=False
):
    """w = A @ (k * beta * exp(g)). Mirrors ref_gdr.torch_w_u_fwd (w only)."""
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    group_size = HV // H
    C = chunk_size
    k_shape = (batch_size, num_tokens, H, DK)
    g_shape = (batch_size, num_tokens, HV)
    a_shape = (batch_size, num_tokens, HV, C)
    w_shape = (batch_size, num_tokens, HV, DK)

    @T.macro
    def kernel_body(bb, bh, left, seq_end, k, beta, a, g, w):
        k_shared = T.alloc_shared((C, DK), dtype=qkva_dtype)
        a_shared = T.alloc_shared((C, C), dtype=qkva_dtype)
        kbg_shared = T.alloc_shared((C, DK), dtype=qkva_dtype)
        g_shared = T.alloc_shared((C,), dtype=accum_dtype, scope="shared")
        b_shared = T.alloc_shared((C,), dtype=accum_dtype, scope="shared")
        w_frag = T.alloc_fragment((C, DK), dtype=accum_dtype)

        for d, kk in T.Parallel(C, DK):
            k_shared[d, kk] = T.if_then_else(
                left + d < seq_end, k[bb, left + d, bh, kk], T.cast(0, qkva_dtype)
            )
        for i_g in T.serial(group_size):
            bhv = bh * group_size + i_g
            for j in T.Parallel(C):
                g_shared[j] = T.if_then_else(
                    left + j < seq_end, g[bb, left + j, bhv], T.cast(0, accum_dtype))
                b_shared[j] = T.if_then_else(
                    left + j < seq_end, beta[bb, left + j, bhv], T.cast(0, accum_dtype))
            for c, t in T.Parallel(C, C):
                a_shared[c, t] = T.if_then_else(
                    left + c < seq_end, a[bb, left + c, bhv, t], T.cast(0, qkva_dtype))
            for d, kk in T.Parallel(C, DK):
                kbg_shared[d, kk] = (
                    k_shared[d, kk] * (b_shared[d] * T.exp(g_shared[d]))
                ).astype(qkva_dtype)
            T.gemm_v1(a_shared, kbg_shared, w_frag, clear_accum=True)
            for c, kk in T.Parallel(C, DK):
                if left + c < seq_end:
                    w[bb, left + c, bhv, kk] = w_frag[c, kk].astype(qkva_dtype)

    if is_varlen:
        num_chunks = T.dynamic("num_chunks")

        @T.prim_func
        def kernel(
            k: T.Tensor((1, num_tokens, H, DK), dtype=qkva_dtype),
            beta: T.Tensor((1, num_tokens, HV), dtype=b_dtype),
            a: T.Tensor((1, num_tokens, HV, C), dtype=qkva_dtype),
            g: T.Tensor((1, num_tokens, HV), dtype=g_dtype),
            cu_seqlens: T.Tensor([batch_size + 1], dtype="int32"),
            chunk_indices: T.Tensor([num_chunks, 2], dtype="int32"),
            w: T.Tensor((1, num_tokens, HV, DK), dtype=qkva_dtype),
        ):
            with T.Kernel(num_chunks, H, threads=128) as (bc, bh):
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start = cu_seqlens[batch_idx]
                seq_end = cu_seqlens[batch_idx + 1]
                kernel_body(0, bh, seq_start + chunk_idx * C, seq_end, k, beta, a, g, w)
    else:
        @T.prim_func
        def kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            beta: T.Tensor(g_shape, dtype=b_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            g: T.Tensor(g_shape, dtype=g_dtype),
            w: T.Tensor(w_shape, dtype=qkva_dtype),
        ):
            with T.Kernel(
                tilelang.cdiv(num_tokens, chunk_size), batch_size * H, threads=128
            ) as (bc, bbh):
                kernel_body(bbh // H, bbh % H, bc * C, num_tokens, k, beta, a, g, w)

    return kernel


def recompute_w_sm12x(k, beta, A, g, chunk_size=64, cu_seqlens=None):
    batch_size, num_tokens, num_key_heads, head_dim = k.shape
    num_value_heads = beta.shape[-1]
    w = k.new_empty(batch_size, num_tokens, num_value_heads, head_dim)
    is_varlen = cu_seqlens is not None
    kernel = tilelang_recompute_w_fwd_sm12x(
        H=num_key_heads, HV=num_value_heads, DK=head_dim, chunk_size=chunk_size,
        accum_dtype="float32", qkva_dtype=k.dtype, g_dtype=g.dtype, b_dtype=beta.dtype,
        is_varlen=is_varlen,
    )
    if is_varlen:
        ci = prepare_chunk_indices(cu_seqlens, chunk_size).to(torch.int32)
        kernel(k, beta, A, g, cu_seqlens.to(torch.int32), ci, w)
    else:
        kernel(k, beta, A, g, w)
    return w


@tilelang.jit(**_JIT)
def tilelang_dqkwg_gqa_sm12x(
    H, HV, DK, DV, chunk_size, scale, qkva_dtype, g_dtype, accum_dtype,
    block_V=64, is_varlen=False,
):
    """dq, dk, dw, dg from h, dh, do_grad, dv, vn, w, g, q, k.
    Mirrors ref_gdr.torch_chunk_dqkwg_bwd (GQA, dq/dk reduced to key-head)."""
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    group_size = HV // H
    C = chunk_size

    q_shape = (batch_size, num_tokens, H, DK)
    v_shape = (batch_size, num_tokens, HV, DV)
    state_shape = (batch_size, tilelang.cdiv(num_tokens, chunk_size), HV, DK, DV)
    dg_shape = (batch_size, num_tokens, HV)

    @T.macro
    def kernel_body(bb, bh, bc, left, seq_end, q, k, v, w, g, h, dv, do_grad, dh, dq, dk, dw, dg):
            q_shared = T.alloc_shared((C, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((C, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((C, block_V), dtype=qkva_dtype)
            do_shared = T.alloc_shared((C, block_V), dtype=qkva_dtype)
            h_shared = T.alloc_shared((DK, block_V), dtype=qkva_dtype)
            dh_shared = T.alloc_shared((DK, block_V), dtype=qkva_dtype)
            ds_shared = T.alloc_shared((C, C), dtype=qkva_dtype)
            g_shared = T.alloc_shared((C,), dtype=accum_dtype, scope="shared")
            dgl_shared = T.alloc_shared((1,), dtype=accum_dtype, scope="shared")

            acc_dq = T.alloc_fragment((C, DK), dtype=accum_dtype)
            acc_dk = T.alloc_fragment((C, DK), dtype=accum_dtype)
            b_dq = T.alloc_fragment((C, DK), dtype=accum_dtype)
            b_dk = T.alloc_fragment((C, DK), dtype=accum_dtype)
            b_dw = T.alloc_fragment((C, DK), dtype=accum_dtype)
            b_ds = T.alloc_fragment((C, C), dtype=accum_dtype)
            hdh = T.alloc_fragment((DK, block_V), dtype=accum_dtype)
            hdh_row = T.alloc_fragment((DK,), dtype=accum_dtype)
            hdh_sum = T.alloc_fragment((1,), dtype=accum_dtype)
            prod = T.alloc_fragment((C, DK), dtype=accum_dtype)
            red = T.alloc_fragment((C,), dtype=accum_dtype)
            red_k = T.alloc_fragment((C,), dtype=accum_dtype)
            scalar = T.alloc_fragment((1,), dtype=accum_dtype)
            dg_last = T.alloc_fragment((1,), dtype=accum_dtype)
            ds_cast = T.alloc_fragment((C, C), dtype=qkva_dtype)

            g_last = T.alloc_var("float32")
            last_local = T.alloc_var("int32")
            # last valid token index within this (possibly partial) chunk
            last_local = T.min(C - 1, seq_end - left - 1)

            for j_s, j_k in T.Parallel(C, DK):
                q_shared[j_s, j_k] = T.if_then_else(
                    left + j_s < seq_end, q[bb, left + j_s, bh, j_k], T.cast(0, qkva_dtype))
                k_shared[j_s, j_k] = T.if_then_else(
                    left + j_s < seq_end, k[bb, left + j_s, bh, j_k], T.cast(0, qkva_dtype))
            T.clear(acc_dq)
            T.clear(acc_dk)

            for i_g in T.serial(group_size):
                bhv = bh * group_size + i_g
                T.clear(b_dq)
                T.clear(b_dk)
                T.clear(b_dw)
                T.clear(b_ds)
                T.clear(dg_last)

                for i_v in T.serial(tilelang.cdiv(DV, block_V)):
                    v_off = i_v * block_V
                    for j_s, j_v in T.Parallel(C, block_V):
                        v_shared[j_s, j_v] = T.if_then_else(
                            left + j_s < seq_end, v[bb, left + j_s, bhv, v_off + j_v],
                            T.cast(0, qkva_dtype))
                        do_shared[j_s, j_v] = T.if_then_else(
                            left + j_s < seq_end, do_grad[bb, left + j_s, bhv, v_off + j_v],
                            T.cast(0, qkva_dtype))
                    T.copy(h[bb, bc, bhv, 0:DK, v_off : v_off + block_V], h_shared)
                    T.copy(dh[bb, bc, bhv, 0:DK, v_off : v_off + block_V], dh_shared)

                    T.gemm_v1(do_shared, v_shared, b_ds, transpose_B=True, clear_accum=False)
                    T.gemm_v1(do_shared, h_shared, b_dq, transpose_B=True, clear_accum=False)
                    T.gemm_v1(v_shared, dh_shared, b_dk, transpose_B=True, clear_accum=False)
                    for j_s, j_v in T.Parallel(C, block_V):
                        v_shared[j_s, j_v] = T.if_then_else(
                            left + j_s < seq_end, dv[bb, left + j_s, bhv, v_off + j_v],
                            T.cast(0, qkva_dtype))
                    T.gemm_v1(v_shared, h_shared, b_dw, transpose_B=True, clear_accum=False)

                    for j_k, j_v in T.Parallel(DK, block_V):
                        hdh[j_k, j_v] = (
                            T.cast(h_shared[j_k, j_v], "float32")
                            * T.cast(dh_shared[j_k, j_v], "float32")
                        )
                    T.reduce_sum(hdh, hdh_row, dim=1)
                    T.reduce_sum(hdh_row, hdh_sum, dim=0)
                    dg_last[0] += hdh_sum[0]

                for j_s in T.Parallel(C):
                    g_shared[j_s] = T.if_then_else(
                        left + j_s < seq_end, g[bb, left + j_s, bhv], T.cast(0, accum_dtype))
                g_last = g_shared[last_local]  # last VALID token's gate (partial-chunk safe)
                dg_last[0] *= T.exp(g_last)

                for j_s, j_k in T.Parallel(C, DK):
                    b_dq[j_s, j_k] *= T.exp(g_shared[j_s]) * scale
                    b_dk[j_s, j_k] *= T.exp(g_last - g_shared[j_s])

                for j_s, j_k in T.Parallel(C, DK):
                    prod[j_s, j_k] = b_dk[j_s, j_k] * T.cast(k_shared[j_s, j_k], "float32")
                T.reduce_sum(prod, red, dim=1)
                T.reduce_sum(red, scalar, dim=0)
                dg_last[0] += scalar[0]

                for j_s, j_t in T.Parallel(C, C):
                    if j_s >= j_t:
                        b_ds[j_s, j_t] *= T.exp(g_shared[j_s] - g_shared[j_t]) * scale
                    else:
                        b_ds[j_s, j_t] = 0.0
                    ds_cast[j_s, j_t] = b_ds[j_s, j_t]

                T.copy(ds_cast, ds_shared)
                T.gemm_v1(ds_shared, k_shared, b_dq, clear_accum=False)
                T.gemm_v1(ds_shared, q_shared, b_dk, transpose_A=True, clear_accum=False)

                for j_s, j_k in T.Parallel(C, DK):
                    prod[j_s, j_k] = b_dq[j_s, j_k] * T.cast(q_shared[j_s, j_k], "float32")
                T.reduce_sum(prod, red, dim=1)
                for j_s, j_k in T.Parallel(C, DK):
                    prod[j_s, j_k] = b_dk[j_s, j_k] * T.cast(k_shared[j_s, j_k], "float32")
                T.reduce_sum(prod, red_k, dim=1)

                # Fold dg_last into the single parallel store via a shared scalar
                # (no global read-modify-write race; see the Hv=16 fix). dg_last
                # is added to the last VALID token of the chunk (partial-safe).
                dgl_shared[0] = dg_last[0]
                for j_s in T.Parallel(C):
                    if left + j_s < seq_end:
                        dg[bb, left + j_s, bhv] = red[j_s] - red_k[j_s] + T.if_then_else(
                            j_s == last_local, dgl_shared[0], T.cast(0, accum_dtype)
                        )

                for j_s, j_k in T.Parallel(C, DK):
                    acc_dq[j_s, j_k] += b_dq[j_s, j_k]
                    acc_dk[j_s, j_k] += b_dk[j_s, j_k]
                    if left + j_s < seq_end:
                        dw[bb, left + j_s, bhv, j_k] = -b_dw[j_s, j_k]

            for j_s, j_k in T.Parallel(C, DK):
                if left + j_s < seq_end:
                    dq[bb, left + j_s, bh, j_k] = acc_dq[j_s, j_k]
                    dk[bb, left + j_s, bh, j_k] = acc_dk[j_s, j_k]

    if is_varlen:
        num_chunks = T.dynamic("num_chunks")

        @T.prim_func
        def kernel(
            q: T.Tensor((1, num_tokens, H, DK), dtype=qkva_dtype),
            k: T.Tensor((1, num_tokens, H, DK), dtype=qkva_dtype),
            v: T.Tensor((1, num_tokens, HV, DV), dtype=qkva_dtype),
            w: T.Tensor((1, num_tokens, HV, DK), dtype=qkva_dtype),
            g: T.Tensor((1, num_tokens, HV), dtype=g_dtype),
            h: T.Tensor((1, num_chunks, HV, DK, DV), dtype=qkva_dtype),
            dv: T.Tensor((1, num_tokens, HV, DV), dtype=qkva_dtype),
            do_grad: T.Tensor((1, num_tokens, HV, DV), dtype=qkva_dtype),
            dh: T.Tensor((1, num_chunks, HV, DK, DV), dtype=qkva_dtype),
            cu_seqlens: T.Tensor([batch_size + 1], dtype="int32"),
            chunk_indices: T.Tensor([num_chunks, 2], dtype="int32"),
            dq: T.Tensor((1, num_tokens, H, DK), dtype=qkva_dtype),
            dk: T.Tensor((1, num_tokens, H, DK), dtype=qkva_dtype),
            dw: T.Tensor((1, num_tokens, HV, DK), dtype=qkva_dtype),
            dg: T.Tensor((1, num_tokens, HV), dtype=g_dtype),
        ):
            with T.Kernel(num_chunks, H, threads=256) as (bc, bh):
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start = cu_seqlens[batch_idx]
                seq_end = cu_seqlens[batch_idx + 1]
                kernel_body(0, bh, bc, seq_start + chunk_idx * C, seq_end,
                            q, k, v, w, g, h, dv, do_grad, dh, dq, dk, dw, dg)
    else:
        @T.prim_func
        def kernel(
            q: T.Tensor(q_shape, dtype=qkva_dtype),
            k: T.Tensor(q_shape, dtype=qkva_dtype),
            v: T.Tensor(v_shape, dtype=qkva_dtype),
            w: T.Tensor(q_shape[:-2] + (HV, DK), dtype=qkva_dtype),
            g: T.Tensor(dg_shape, dtype=g_dtype),
            h: T.Tensor(state_shape, dtype=qkva_dtype),
            dv: T.Tensor(v_shape, dtype=qkva_dtype),
            do_grad: T.Tensor(v_shape, dtype=qkva_dtype),
            dh: T.Tensor(state_shape, dtype=qkva_dtype),
            dq: T.Tensor(q_shape, dtype=qkva_dtype),
            dk: T.Tensor(q_shape, dtype=qkva_dtype),
            dw: T.Tensor(q_shape[:-2] + (HV, DK), dtype=qkva_dtype),
            dg: T.Tensor(dg_shape, dtype=g_dtype),
        ):
            with T.Kernel(
                tilelang.cdiv(num_tokens, chunk_size), batch_size * H, threads=256
            ) as (bc, bbh):
                kernel_body(bbh // H, bbh % H, bc, bc * C, num_tokens,
                            q, k, v, w, g, h, dv, do_grad, dh, dq, dk, dw, dg)

    return kernel


def chunk_dqkwg_sm12x(q, k, v, w, g, h, dv, do_grad, dh, scale, chunk_size=64, cu_seqlens=None):
    batch_size, num_tokens, num_key_heads, head_dim = k.shape
    num_value_heads = v.shape[2]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w)
    dg = torch.empty_like(g)
    is_varlen = cu_seqlens is not None
    kernel = tilelang_dqkwg_gqa_sm12x(
        H=num_key_heads, HV=num_value_heads, DK=head_dim, DV=v.shape[-1],
        chunk_size=chunk_size, scale=scale, qkva_dtype=q.dtype, g_dtype=g.dtype,
        accum_dtype="float32", block_V=64, is_varlen=is_varlen,
    )
    if is_varlen:
        ci = prepare_chunk_indices(cu_seqlens, chunk_size).to(torch.int32)
        kernel(q, k, v, w, g, h, dv, do_grad, dh, cu_seqlens.to(torch.int32), ci,
               dq, dk, dw, dg)
    else:
        kernel(q, k, v, w, g, h, dv, do_grad, dh, dq, dk, dw, dg)
    return dq, dk, dw, dg
