import os

import torch
import tilelang
import tilelang.language as T
import triton
import triton.language as tl

from flash_qla.ops.utils import chunk_local_cumsum
from flash_qla.utils import (
    fill_last_chunk_of_g,
    pack,
    pad_and_reshape,
    prepare_chunk_indices,
    prepare_chunk_offsets,
    unpack,
)


def _maybe_unpack_tokens(x: torch.Tensor, cu_seqlens: torch.Tensor | None):
    if cu_seqlens is None:
        return x
    return unpack(x, cu_seqlens)


def _maybe_pack_tokens(x: torch.Tensor, cu_seqlens: torch.Tensor | None):
    if cu_seqlens is None:
        return x
    return pack(x, cu_seqlens)


def _maybe_unpack_chunks(
    x: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int,
):
    if x is None:
        return None
    if cu_seqlens is None:
        return x
    return unpack(x, prepare_chunk_offsets(cu_seqlens, chunk_size))


def _repeat_kv_heads(x: torch.Tensor, num_value_heads: int):
    num_key_heads = x.shape[2]
    if num_key_heads == num_value_heads:
        return x
    return x.repeat_interleave(num_value_heads // num_key_heads, dim=2)


def _chunk_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int,
):
    q = _maybe_unpack_tokens(q, cu_seqlens).float()
    k = _maybe_unpack_tokens(k, cu_seqlens).float()
    v = _maybe_unpack_tokens(v, cu_seqlens).float()
    g = _maybe_unpack_tokens(g, cu_seqlens).float()
    b = _maybe_unpack_tokens(b, cu_seqlens).float()
    do = _maybe_unpack_tokens(do, cu_seqlens).float()
    h = _maybe_unpack_chunks(h, cu_seqlens, chunk_size)
    if h is not None:
        h = h.float()
    if dht is not None:
        dht = dht.float()

    num_value_heads = v.shape[2]
    q = _repeat_kv_heads(q, num_value_heads)
    k = _repeat_kv_heads(k, num_value_heads)

    num_tokens = k.shape[1]
    q = pad_and_reshape(q, dim=1, chunk_size=chunk_size)
    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)
    v = pad_and_reshape(v, dim=1, chunk_size=chunk_size)
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)
    b = pad_and_reshape(b, dim=1, chunk_size=chunk_size)
    do = pad_and_reshape(do, dim=1, chunk_size=chunk_size)
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    return q, k, v, g, b, do, h, dht, num_tokens


def _wy_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
):
    g_exp = g.exp().unsqueeze(-1)
    b_exp = b.unsqueeze(-1)
    w = torch.einsum("bnchd,bndhk->bnchk", a, k * b_exp * g_exp)
    u = torch.einsum("bnchd,bndhv->bnchv", a, v * b_exp)
    return w, u


def _kkt_solve_fwd(
    k: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int,
):
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device)
    )
    decay_mask = torch.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(mask[None, None, :, :, None], 0.0)
    attn = torch.einsum("bnchk,bndhk->bnchd", k * b.unsqueeze(-1), k)
    a = attn * decay_mask.swapaxes(-2, -1)

    lower = a.swapaxes(2, 3).contiguous()
    eye = torch.eye(chunk_size, dtype=lower.dtype, device=lower.device)
    eye = eye.expand(*lower.shape[:-2], chunk_size, chunk_size)
    lower = lower + eye
    return torch.linalg.solve_triangular(lower, eye, upper=False).swapaxes(2, 3)


def _ensure_fla2_triton_backend():
    # FLA2's TileLang bwd backend currently misaligns on sm12x in this setup.
    os.environ["FLA_TILELANG"] = "0"


_FLA2_BWD_AVAILABLE = None
_FLA2_BWD_OPS = None


def _get_fla2_bwd_ops():
    global _FLA2_BWD_OPS, _FLA2_BWD_AVAILABLE
    if _FLA2_BWD_OPS is not None:
        return _FLA2_BWD_OPS
    if _FLA2_BWD_AVAILABLE is False:
        return None
    _ensure_fla2_triton_backend()
    try:
        from fla.ops.common.chunk_delta_h import (
            chunk_gated_delta_rule_bwd_dhu,
            chunk_gated_delta_rule_fwd_h,
            chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
        )
        from fla.ops.common.chunk_o import (
            chunk_bwd_dqkwg,
            chunk_bwd_dv_local,
            chunk_bwd_kernel_dqkwg,
        )
        from fla.ops.gated_delta_rule.wy_fast import (
            prepare_wy_repr_bwd,
            prepare_wy_repr_bwd_kernel,
            recompute_w_u_fwd,
        )
        from fla.ops.utils import chunk_local_cumsum as fla_chunk_local_cumsum
    except ImportError:
        _FLA2_BWD_AVAILABLE = False
        _FLA2_BWD_OPS = None
    else:
        _FLA2_BWD_AVAILABLE = True
        _FLA2_BWD_OPS = (
            recompute_w_u_fwd,
            chunk_gated_delta_rule_fwd_h,
            chunk_bwd_dv_local,
            chunk_gated_delta_rule_bwd_dhu,
            chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
            chunk_bwd_dqkwg,
            chunk_bwd_kernel_dqkwg,
            prepare_wy_repr_bwd,
            prepare_wy_repr_bwd_kernel,
            fla_chunk_local_cumsum,
        )
    return _FLA2_BWD_OPS


def _fla2_bwd_available():
    return _get_fla2_bwd_ops() is not None


def is_fla2_bwd_available():
    return _fla2_bwd_available()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_prepare_fla2_bwd_a(
    H,
    chunk_size,
    a_dtype,
    g_dtype,
    seqlen_dtype,
    is_varlen,
):
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")

    if is_varlen:
        data_batch_size = 1
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
    else:
        data_batch_size = T.dynamic("data_batch_size")
        a_shape = (data_batch_size, num_tokens, H, chunk_size)
        g_shape = (data_batch_size, num_tokens, H)

    @T.prim_func
    def tilelang_prepare_fla2_bwd_a_kernel(
        a: T.Tensor(a_shape, dtype=a_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        cu_seqlens: T.Tensor((real_batch_size + 1,), dtype=seqlen_dtype),
        chunk_indices: T.Tensor((num_chunks, 2), dtype=seqlen_dtype),
    ):
        with T.Kernel(num_chunks, data_batch_size * H, threads=256) as (bc, bbh):
            bb = bbh // H
            bh = bbh % H

            batch_idx = T.alloc_var("int32")
            chunk_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")

            if is_varlen:
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]
                bb = 0
            else:
                batch_idx = bb
                chunk_idx = bc
                seq_start_idx = 0
                seq_end_idx = num_tokens

            left = seq_start_idx + chunk_idx * chunk_size
            for j_m, j_n in T.Parallel(chunk_size, chunk_size):
                row = left + j_m
                col = left + j_n
                if row < seq_end_idx:
                    if col < seq_end_idx:
                        a[bb, row, bh, j_n] = a[bb, row, bh, j_n] * T.exp(
                            g[bb, row, bh] - g[bb, col, bh]
                        )
                    else:
                        a[bb, row, bh, j_n] = 0.0

    return tilelang_prepare_fla2_bwd_a_kernel


@triton.jit
def _prepare_fla2_bwd_a_kernel(
    a,
    g,
    cu_seqlens,
    chunk_indices,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_c * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        i_h = i_bh
    else:
        i_n = i_bh // H
        i_h = i_bh - i_n * H
        i_t = i_c
        bos = i_n * T
        eos = bos + T

    o_m = tl.arange(0, BT)
    o_n = tl.arange(0, BT)
    row = bos + i_t * BT + o_m
    col = bos + i_t * BT + o_n
    row_mask = row < eos
    col_mask = col < eos

    g_row = tl.load(g + row * H + i_h, mask=row_mask, other=0.0)
    g_col = tl.load(g + col * H + i_h, mask=col_mask, other=0.0)
    a_ptrs = a + (row[:, None] * H + i_h) * BT + o_n[None, :]
    a_vals = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
    a_vals *= tl.exp(g_row[:, None] - g_col[None, :])
    a_vals = tl.where(col_mask[None, :], a_vals, 0.0)
    tl.store(a_ptrs, a_vals, mask=row_mask[:, None])


def prepare_fla2_bwd_a(
    a: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
):
    assert chunk_size == 64
    batch_size, num_tokens, num_heads, _ = a.shape
    if cu_seqlens is None:
        num_chunks = triton.cdiv(num_tokens, chunk_size)
        chunk_indices = torch.empty((num_chunks, 2), dtype=torch.int32, device=a.device)
        cu_seqlens_arg = torch.empty(
            (batch_size + 1), dtype=torch.int32, device=a.device
        )
        seqlen_dtype = torch.int32
        is_varlen = False
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        cu_seqlens_arg = cu_seqlens
        num_chunks = len(chunk_indices)
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    kernel = tilelang_prepare_fla2_bwd_a(
        H=num_heads,
        chunk_size=chunk_size,
        a_dtype=a.dtype,
        g_dtype=g.dtype,
        seqlen_dtype=seqlen_dtype,
        is_varlen=is_varlen,
    )
    kernel(
        a,
        g,
        cu_seqlens_arg,
        chunk_indices,
    )
    return a


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_recompute_w_fwd(
    H,
    HV,
    DK,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    seqlen_dtype,
    is_varlen,
    block_K=64,
):
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    if is_varlen:
        data_batch_size = 1
        k_shape = (1, num_tokens, H, DK)
        a_shape = (1, num_tokens, HV, chunk_size)
        g_shape = (1, num_tokens, HV)
        b_shape = (1, num_tokens, HV)
        w_shape = (1, num_tokens, HV, DK)
    else:
        data_batch_size = T.dynamic("data_batch_size")
        k_shape = (data_batch_size, num_tokens, H, DK)
        a_shape = (data_batch_size, num_tokens, HV, chunk_size)
        g_shape = (data_batch_size, num_tokens, HV)
        b_shape = (data_batch_size, num_tokens, HV)
        w_shape = (data_batch_size, num_tokens, HV, DK)

    @T.prim_func
    def tilelang_recompute_w_fwd_kernel(
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        beta: T.Tensor(b_shape, dtype=b_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
        chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
        w: T.Tensor(w_shape, dtype=qkva_dtype),
    ):
        with T.Kernel(num_chunks, data_batch_size * HV, tilelang.cdiv(DK, block_K), threads=256) as (
            bc,
            bbh,
            bk,
        ):
            bb = bbh // HV
            bh = bbh % HV
            bhg = bh // (HV // H)

            batch_idx = T.alloc_var("int32")
            chunk_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")

            if is_varlen:
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]
                bb = 0
            else:
                batch_idx = bb
                chunk_idx = bc
                seq_start_idx = 0
                seq_end_idx = num_tokens

            left = seq_start_idx + chunk_idx * block_S

            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            kbg_shared = T.alloc_shared((block_S, block_K), dtype=qkva_dtype)
            w_fragment = T.alloc_fragment((block_S, block_K), dtype=accum_dtype)

            for j_s, j_t in T.Parallel(block_S, block_S):
                if left + j_s < seq_end_idx:
                    a_shared[j_s, j_t] = a[bb, left + j_s, bh, j_t]
                else:
                    a_shared[j_s, j_t] = 0

            for j_s, j_k in T.Parallel(block_S, block_K):
                if (left + j_s < seq_end_idx) and (bk * block_K + j_k < DK):
                    kbg_shared[j_s, j_k] = (
                        k[bb, left + j_s, bhg, bk * block_K + j_k]
                        * beta[bb, left + j_s, bh]
                        * T.exp(g[bb, left + j_s, bh])
                    )
                else:
                    kbg_shared[j_s, j_k] = 0

            T.gemm_v1(a_shared, kbg_shared, w_fragment, clear_accum=True)

            for j_s, j_k in T.Parallel(block_S, block_K):
                if (left + j_s < seq_end_idx) and (bk * block_K + j_k < DK):
                    w[bb, left + j_s, bh, bk * block_K + j_k] = w_fragment[
                        j_s, j_k
                    ]

    return tilelang_recompute_w_fwd_kernel


@triton.jit(do_not_specialize=["T"])
def _recompute_w_fwd_kernel(
    k,
    beta,
    a,
    g,
    w,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HV
    i_h = i_bh - i_b * HV
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    p_b = tl.make_block_ptr(
        beta + bos * HV + i_h, (T,), (HV,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(
        g + bos * HV + i_h, (T,), (HV,), (i_t * BT,), (BT,), (0,)
    )
    p_a = tl.make_block_ptr(
        a + (bos * HV + i_h) * BT,
        (T, BT),
        (HV * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    b_b = tl.load(p_b, boundary_check=(0,))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))
    b_a = tl.load(p_a, boundary_check=(0, 1))

    i_kh = i_h // (HV // H)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_kh) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * HV + i_h) * K,
            (T, K),
            (HV * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kbg = b_k * (b_b * b_g)[:, None]
        b_w = tl.dot(b_a, b_kbg.to(b_k.dtype), allow_tf32=False)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def _recompute_w_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    chunk_size: int,
):
    batch_size, num_tokens, num_key_heads, head_dim = k.shape
    num_value_heads = beta.shape[-1]
    num_chunks = (
        triton.cdiv(num_tokens, chunk_size)
        if cu_seqlens is None
        else len(chunk_indices)
    )
    w = k.new_empty(batch_size, num_tokens, num_value_heads, head_dim)
    if cu_seqlens is None:
        cu_seqlens_arg = torch.empty(
            (batch_size + 1), dtype=torch.int32, device=k.device
        )
        chunk_indices_arg = torch.empty(
            (num_chunks, 2), dtype=torch.int32, device=k.device
        )
        seqlen_dtype = torch.int32
        is_varlen = False
    else:
        cu_seqlens_arg = cu_seqlens
        chunk_indices_arg = chunk_indices
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    kernel = tilelang_recompute_w_fwd(
        H=num_key_heads,
        HV=num_value_heads,
        DK=head_dim,
        chunk_size=chunk_size,
        accum_dtype="float32",
        qkva_dtype=k.dtype,
        g_dtype=g.dtype,
        b_dtype=beta.dtype,
        seqlen_dtype=seqlen_dtype,
        is_varlen=is_varlen,
        block_K=128,
    )
    kernel(
        k,
        beta,
        a,
        g,
        cu_seqlens_arg,
        chunk_indices_arg,
        w,
    )
    return w


def prepare_fla2_bwd_w(
    k: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int,
):
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    return _recompute_w_fwd(
        k=k,
        beta=beta,
        a=a,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )


def _recompute_w_fwd_triton(
    k: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    chunk_size: int,
):
    batch_size, num_tokens, num_key_heads, head_dim = k.shape
    num_value_heads = beta.shape[-1]
    num_chunks = (
        triton.cdiv(num_tokens, chunk_size)
        if cu_seqlens is None
        else len(chunk_indices)
    )
    w = k.new_empty(batch_size, num_tokens, num_value_heads, head_dim)
    _recompute_w_fwd_kernel[(num_chunks, batch_size * num_value_heads)](
        k,
        beta,
        a,
        g,
        w,
        cu_seqlens,
        chunk_indices,
        T=num_tokens,
        H=num_key_heads,
        HV=num_value_heads,
        K=head_dim,
        BT=chunk_size,
        BK=64,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=2,
        num_stages=3,
    )
    return w


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_add_bwd_partials(
    dk_dtype,
    dg_dtype,
    block: int = 256,
    items_per_thread: int = 4,
):
    n_dk = T.dynamic("n_dk")
    n_dg = T.dynamic("n_dg")

    @T.prim_func
    def tilelang_add_bwd_partials_kernel(
        dk: T.Tensor((n_dk,), dtype=dk_dtype),
        dk2: T.Tensor((n_dk,), dtype=dk_dtype),
        dg: T.Tensor((n_dg,), dtype=dg_dtype),
        dg2: T.Tensor((n_dg,), dtype=dg_dtype),
    ):
        with T.Kernel(tilelang.cdiv(n_dk, block * items_per_thread), threads=block) as (
            bid,
        ):
            for tx in T.Parallel(block):
                for item in T.serial(items_per_thread):
                    offset = bid * block * items_per_thread + item * block + tx
                    if offset < n_dk:
                        dk[offset] = dk[offset] + dk2[offset]
                    if offset < n_dg:
                        dg[offset] = dg[offset] + dg2[offset]

    return tilelang_add_bwd_partials_kernel


@triton.jit
def _add_bwd_partials_kernel(
    dk,
    dk2,
    dg,
    dg2,
    n_dk: tl.constexpr,
    n_dg: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask_dk = offsets < n_dk
    mask_dg = offsets < n_dg
    dk_vals = tl.load(dk + offsets, mask=mask_dk, other=0.0)
    dk2_vals = tl.load(dk2 + offsets, mask=mask_dk, other=0.0)
    tl.store(dk + offsets, dk_vals + dk2_vals, mask=mask_dk)
    dg_vals = tl.load(dg + offsets, mask=mask_dg, other=0.0)
    dg2_vals = tl.load(dg2 + offsets, mask=mask_dg, other=0.0)
    tl.store(dg + offsets, dg_vals + dg2_vals, mask=mask_dg)


def _add_bwd_partials(
    dk: torch.Tensor,
    dk2: torch.Tensor,
    dg: torch.Tensor,
    dg2: torch.Tensor,
):
    if dg.numel() <= dk.numel():
        kernel = tilelang_add_bwd_partials(
            dk_dtype=dk.dtype,
            dg_dtype=dg.dtype,
            block=256,
            items_per_thread=4,
        )
        kernel(
            dk.reshape(-1),
            dk2.reshape(-1),
            dg.reshape(-1),
            dg2.reshape(-1),
        )
        return

    block = 1024
    n_dk = dk.numel()
    n_dg = dg.numel()
    grid = (triton.cdiv(max(n_dk, n_dg), block),)
    _add_bwd_partials_kernel[grid](
        dk,
        dk2,
        dg,
        dg2,
        n_dk=n_dk,
        n_dg=n_dg,
        BLOCK=block,
    )


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_reduce_add_wy_gqa(
    H,
    HV,
    DK,
    dk_dtype,
    dg_dtype,
    accum_dtype,
    block_size: int = 16,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    group_size = HV // H

    dk_shape = (batch_size, num_tokens, H, DK)
    dk_hv_shape = (batch_size, num_tokens, HV, DK)
    dg_shape = (batch_size, num_tokens, HV)

    @T.prim_func
    def tilelang_reduce_add_wy_gqa_kernel(
        dk: T.Tensor(dk_shape, dtype=dk_dtype),
        dk_hv: T.Tensor(dk_hv_shape, dtype=dk_dtype),
        dg: T.Tensor(dg_shape, dtype=dg_dtype),
        dg2: T.Tensor(dg_shape, dtype=dg_dtype),
    ):
        with T.Kernel(
            tilelang.cdiv(num_tokens, block_size), H, batch_size, threads=128
        ) as (bt, bh, bb):
            tmp = T.alloc_fragment((block_size, DK), dtype=accum_dtype)
            acc = T.alloc_fragment((block_size, DK), dtype=accum_dtype)

            T.clear(acc)
            for i in T.serial(group_size):
                T.copy(
                    dk_hv[
                        bb,
                        bt * block_size : (bt + 1) * block_size,
                        bh * group_size + i,
                        0:DK,
                    ],
                    tmp,
                )
                for j, k in T.Parallel(block_size, DK):
                    acc[j, k] += tmp[j, k]

            T.copy(
                dk[bb, bt * block_size : (bt + 1) * block_size, bh, 0:DK],
                tmp,
            )
            for j, k in T.Parallel(block_size, DK):
                acc[j, k] += tmp[j, k]
            T.copy(
                acc,
                dk[bb, bt * block_size : (bt + 1) * block_size, bh, 0:DK],
            )

            for i, j in T.Parallel(group_size, block_size):
                token = bt * block_size + j
                if token < num_tokens:
                    vh = bh * group_size + i
                    dg[bb, token, vh] = dg[bb, token, vh] + dg2[bb, token, vh]

    return tilelang_reduce_add_wy_gqa_kernel


def _reduce_add_wy_gqa(
    dk: torch.Tensor,
    dk_hv: torch.Tensor,
    dg: torch.Tensor,
    dg2: torch.Tensor,
    num_key_heads: int,
):
    kernel = tilelang_reduce_add_wy_gqa(
        H=num_key_heads,
        HV=dk_hv.shape[2],
        DK=dk_hv.shape[-1],
        dk_dtype=dk.dtype,
        dg_dtype=dg.dtype,
        accum_dtype="float32",
        block_size=32,
    )
    kernel(dk, dk_hv, dg, dg2)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_reduce_bwd_gqa_combined(
    H,
    HV,
    DK,
    NK,
    qk_dtype,
    dg_dtype,
    accum_dtype,
    block_size: int = 32,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    group_size = HV // H

    qk_shape = (batch_size, num_tokens, H, DK)
    qk_hv_shape = (batch_size, num_tokens, HV, DK)
    dg_parts_shape = (NK, batch_size, num_tokens, HV)
    dg_shape = (batch_size, num_tokens, HV)

    @T.prim_func
    def tilelang_reduce_bwd_gqa_combined_kernel(
        dq: T.Tensor(qk_shape, dtype=qk_dtype),
        dk: T.Tensor(qk_shape, dtype=qk_dtype),
        dq_hv: T.Tensor(qk_hv_shape, dtype=qk_dtype),
        dk_hv: T.Tensor(qk_hv_shape, dtype=qk_dtype),
        dk_wy_hv: T.Tensor(qk_hv_shape, dtype=qk_dtype),
        dg: T.Tensor(dg_shape, dtype=dg_dtype),
        dg_parts: T.Tensor(dg_parts_shape, dtype=dg_dtype),
        dg_wy: T.Tensor(dg_shape, dtype=dg_dtype),
    ):
        with T.Kernel(
            tilelang.cdiv(num_tokens, block_size), H, batch_size, threads=128
        ) as (bt, bh, bb):
            tmp_q = T.alloc_fragment((block_size, DK), dtype=accum_dtype)
            tmp_k = T.alloc_fragment((block_size, DK), dtype=accum_dtype)
            tmp_kwy = T.alloc_fragment((block_size, DK), dtype=accum_dtype)
            acc_q = T.alloc_fragment((block_size, DK), dtype=accum_dtype)
            acc_k = T.alloc_fragment((block_size, DK), dtype=accum_dtype)
            dg_acc = T.alloc_var("float32")

            T.clear(acc_q)
            T.clear(acc_k)
            for i in T.serial(group_size):
                vh = bh * group_size + i
                T.copy(
                    dq_hv[bb, bt * block_size : (bt + 1) * block_size, vh, 0:DK],
                    tmp_q,
                )
                T.copy(
                    dk_hv[bb, bt * block_size : (bt + 1) * block_size, vh, 0:DK],
                    tmp_k,
                )
                T.copy(
                    dk_wy_hv[
                        bb, bt * block_size : (bt + 1) * block_size, vh, 0:DK
                    ],
                    tmp_kwy,
                )
                for j, k in T.Parallel(block_size, DK):
                    acc_q[j, k] += tmp_q[j, k]
                    acc_k[j, k] += tmp_k[j, k] + tmp_kwy[j, k]

            for j, k in T.Parallel(block_size, DK):
                token = bt * block_size + j
                if token < num_tokens:
                    dq[bb, token, bh, k] = acc_q[j, k]
                    dk[bb, token, bh, k] = acc_k[j, k]

            for i, j in T.Parallel(group_size, block_size):
                token = bt * block_size + j
                if token < num_tokens:
                    vh = bh * group_size + i
                    dg_acc = dg_wy[bb, token, vh]
                    for nk in T.serial(NK):
                        dg_acc += dg_parts[nk, bb, token, vh]
                    dg[bb, token, vh] = dg_acc

    return tilelang_reduce_bwd_gqa_combined_kernel


def _reduce_bwd_gqa_combined(
    dq_hv: torch.Tensor,
    dk_hv: torch.Tensor,
    dk_wy_hv: torch.Tensor,
    dg_parts: torch.Tensor,
    dg_wy: torch.Tensor,
    num_key_heads: int,
):
    dq = dq_hv.new_empty(
        dq_hv.shape[0], dq_hv.shape[1], num_key_heads, dq_hv.shape[-1]
    )
    dk = dk_hv.new_empty(
        dk_hv.shape[0], dk_hv.shape[1], num_key_heads, dk_hv.shape[-1]
    )
    dg = torch.empty_like(dg_wy)
    kernel = tilelang_reduce_bwd_gqa_combined(
        H=num_key_heads,
        HV=dq_hv.shape[2],
        DK=dq_hv.shape[-1],
        NK=dg_parts.shape[0],
        qk_dtype=dq_hv.dtype,
        dg_dtype=dg_parts.dtype,
        accum_dtype="float32",
        block_size=32,
    )
    kernel(dq, dk, dq_hv, dk_hv, dk_wy_hv, dg, dg_parts, dg_wy)
    return dq, dk, dg


@triton.jit
def _prepare_wy_repr_bwd_accum_kernel(
    k,
    v,
    beta,
    g,
    a,
    dw,
    du,
    dk,
    dv,
    db,
    dg,
    cu_seqlens,
    chunk_indices,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh - i_b * H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    p_b = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_db = tl.make_block_ptr(
        db + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_a = tl.make_block_ptr(
        a + (bos * H + i_h) * BT,
        (BT, T),
        (1, H * BT),
        (0, i_t * BT),
        (BT, BT),
        (0, 1),
    )
    p_g = tl.make_block_ptr(
        g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )

    b_b = tl.load(p_b, boundary_check=(0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g_exp = tl.exp(b_g)
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)
    b_a = tl.load(p_a, boundary_check=(0, 1))
    b_da = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_dk = tl.make_block_ptr(
            dk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_dw = tl.make_block_ptr(
            dw + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kbg = b_k * (b_b * b_g_exp)[:, None]
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_da += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype))
        b_dkbg = tl.dot(b_a, b_dw)
        b_dk = b_dkbg * (b_g_exp * b_b)[:, None]
        b_db += tl.sum(b_dkbg * b_k * b_g_exp[:, None], 1)
        b_dg += tl.sum(b_dkbg * b_kbg, 1)
        b_dk += tl.load(p_dk, boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_dv = tl.make_block_ptr(
            dv + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_du = tl.make_block_ptr(
            du + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_da += tl.dot(b_du, tl.trans(b_vb))
        b_dvb = tl.dot(b_a, b_du)
        b_dv = b_dvb * b_b[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_a = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_da = tl.where(m_a, b_da, 0)
    b_da = tl.dot(b_da.to(b_a.dtype), b_a)
    b_da = tl.dot(b_a, b_da.to(b_a.dtype))
    b_da *= tl.exp(b_g[:, None] - b_g[None, :])
    b_a = tl.zeros([BT, BT], dtype=tl.float32)
    b_da = tl.where(m_a, -b_da, 0).to(k.dtype.element_ty)

    tl.debug_barrier()
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_dk = tl.make_block_ptr(
            dk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kt = tl.trans(b_k)
        b_kb = b_k * b_b[:, None]
        b_a += tl.dot(b_k, b_kt)
        b_dkb = tl.dot(b_da, b_k)
        b_db += tl.sum(b_dkb * b_k, 1)
        b_dk = b_dkb * b_b[:, None] + tl.trans(
            tl.dot(tl.trans(b_kb).to(b_da.dtype), b_da)
        )
        b_dk += tl.load(p_dk, boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))
    b_a *= b_b[:, None]
    b_ada = b_da * b_a
    p_dg = tl.make_block_ptr(
        dg + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_dg += tl.load(p_dg, boundary_check=(0,)).to(tl.float32)
    b_dg += tl.sum(b_ada, axis=1) - tl.sum(b_ada, axis=0)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


def _prepare_wy_repr_bwd_accum(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dk: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    chunk_size: int,
):
    batch_size, num_tokens, num_heads, head_dim = k.shape
    value_dim = v.shape[-1]
    num_chunks = (
        triton.cdiv(num_tokens, chunk_size)
        if cu_seqlens is None
        else len(chunk_indices)
    )
    dv = torch.empty_like(v)
    db = torch.empty_like(beta)
    num_warps = 8 if num_heads >= 64 else 4
    num_stages = 3 if num_heads >= 64 else 4
    _prepare_wy_repr_bwd_accum_kernel[(num_chunks, batch_size * num_heads)](
        k,
        v,
        beta,
        g,
        a,
        dw,
        du,
        dk,
        dv,
        db,
        dg,
        cu_seqlens,
        chunk_indices,
        T=num_tokens,
        H=num_heads,
        K=head_dim,
        V=value_dim,
        BT=chunk_size,
        BK=64,
        BV=64,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dk, dv, db, dg


def _prepare_wy_repr_bwd_gqa_tilelang_reduce_add(
    prepare_wy_repr_bwd_kernel,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dk: torch.Tensor,
    dg_base: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    chunk_size: int,
):
    batch_size, num_tokens, num_key_heads, head_dim = k.shape
    num_value_heads = v.shape[2]
    value_dim = v.shape[-1]
    num_chunks = (
        triton.cdiv(num_tokens, chunk_size)
        if cu_seqlens is None
        else len(chunk_indices)
    )
    dk_hv = k.new_empty(batch_size, num_tokens, num_value_heads, head_dim)
    dv = torch.empty_like(v)
    dg2 = torch.empty_like(g)
    db = torch.empty_like(beta)
    prepare_wy_repr_bwd_kernel[(num_chunks, batch_size * num_value_heads)](
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=a,
        dw=dw,
        du=du,
        dk=dk_hv,
        dv=dv,
        db=db,
        dg=dg2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=num_tokens,
        H=num_key_heads,
        HV=num_value_heads,
        K=head_dim,
        V=value_dim,
        BT=chunk_size,
        BK=64,
        BV=64,
        USE_EXP2=False,
    )
    _reduce_add_wy_gqa(dk, dk_hv, dg_base, dg2, num_key_heads)
    return dv, db, dg_base


def _prepare_wy_repr_bwd_gqa_raw(
    prepare_wy_repr_bwd_kernel,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    chunk_size: int,
):
    batch_size, num_tokens, num_key_heads, head_dim = k.shape
    num_value_heads = v.shape[2]
    value_dim = v.shape[-1]
    num_chunks = triton.cdiv(num_tokens, chunk_size)
    dk_hv = k.new_empty(batch_size, num_tokens, num_value_heads, head_dim)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    db = torch.empty_like(beta)
    prepare_wy_repr_bwd_kernel[(num_chunks, batch_size * num_value_heads)](
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=a,
        dw=dw,
        du=du,
        dk=dk_hv,
        dv=dv,
        db=db,
        dg=dg,
        cu_seqlens=None,
        chunk_indices=None,
        T=num_tokens,
        H=num_key_heads,
        HV=num_value_heads,
        K=head_dim,
        V=value_dim,
        BT=chunk_size,
        BK=64,
        BV=64,
        USE_EXP2=False,
    )
    return dk_hv, dv, db, dg


def _chunk_bwd_dhu_sm12x_tuned(
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    dht: torch.Tensor | None,
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    chunk_size: int,
):
    batch_size, num_tokens, num_key_heads, head_dim = q.shape
    num_value_heads = do.shape[2]
    value_dim = do.shape[-1]
    num_chunks = triton.cdiv(num_tokens, chunk_size)

    dh = q.new_empty(
        batch_size, num_chunks, num_value_heads, head_dim, value_dim
    )
    dh0 = (
        torch.empty_like(initial_state, dtype=torch.float32)
        if initial_state is not None
        else None
    )
    dv2 = torch.empty_like(dv)
    dh0_arg = (
        dh0
        if dh0 is not None
        else torch.empty((1,), dtype=torch.float32, device=q.device)
    )
    dht_arg = (
        dht
        if dht is not None
        else torch.empty((1,), dtype=torch.float32, device=q.device)
    )
    raw_kernel = chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64.fn.fn
    block_v = 128
    raw_kernel[(triton.cdiv(value_dim, block_v), batch_size * num_value_heads)](
        q,
        k,
        w,
        g,
        None,
        dht_arg,
        dh0_arg,
        do,
        dh,
        dv,
        dv2,
        None,
        None,
        scale,
        num_tokens,
        num_key_heads,
        num_value_heads,
        head_dim,
        value_dim,
        chunk_size,
        block_v,
        USE_G=True,
        USE_GK=False,
        USE_INITIAL_STATE=initial_state is not None,
        USE_FINAL_STATE_GRADIENT=dht is not None,
        USE_EXP2=False,
        TRANSPOSE_STATE=False,
        IS_VARLEN=False,
        num_warps=8,
        num_stages=1,
    )
    return dh, dh0, dv2


def _chunk_bwd_dqkwg_sm12x_tuned(
    chunk_bwd_kernel_dqkwg,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    dv: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
    chunk_size: int,
):
    batch_size, num_tokens, num_key_heads, head_dim = k.shape
    num_value_heads = v.shape[2]
    value_dim = v.shape[-1]
    block_k = 128
    block_v = 64
    num_k_blocks = triton.cdiv(head_dim, block_k)
    num_chunks = triton.cdiv(num_tokens, chunk_size)

    dq_hv = q.new_empty(batch_size, num_tokens, num_value_heads, head_dim)
    dk_hv = k.new_empty(batch_size, num_tokens, num_value_heads, head_dim)
    dw = torch.empty_like(w)
    dg_parts = torch.empty(
        num_k_blocks, *g.shape, dtype=torch.float32, device=g.device
    )
    raw_kernel = chunk_bwd_kernel_dqkwg.fn.fn
    raw_kernel[(num_k_blocks, num_chunks, batch_size * num_value_heads)](
        q,
        k,
        v,
        g,
        None,
        h,
        do,
        dh,
        dq_hv,
        dk_hv,
        dw,
        dv,
        dg_parts,
        None,
        None,
        scale,
        batch_size,
        num_tokens,
        num_key_heads,
        num_value_heads,
        head_dim,
        value_dim,
        chunk_size,
        block_k,
        block_v,
        USE_G=True,
        USE_G_GAMMA=False,
        USE_DW=True,
        USE_EXP2=False,
        TRANSPOSE_STATE=False,
        IS_VARLEN=False,
        num_warps=8,
        num_stages=1,
    )
    group_size = num_value_heads // num_key_heads
    dq = dq_hv.view(batch_size, num_tokens, num_key_heads, group_size, head_dim).sum(
        dim=3
    )
    dk = dk_hv.view(batch_size, num_tokens, num_key_heads, group_size, head_dim).sum(
        dim=3
    )
    dg = dg_parts[0] if num_k_blocks == 1 else dg_parts.sum(dim=0)
    return dq, dk, dw, dg


def _chunk_bwd_dqkwg_sm12x_tuned_raw(
    chunk_bwd_kernel_dqkwg,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    dv: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
    chunk_size: int,
):
    batch_size, num_tokens, num_key_heads, head_dim = k.shape
    num_value_heads = v.shape[2]
    value_dim = v.shape[-1]
    block_k = 128
    block_v = 64
    num_k_blocks = triton.cdiv(head_dim, block_k)
    num_chunks = triton.cdiv(num_tokens, chunk_size)

    dq_hv = q.new_empty(batch_size, num_tokens, num_value_heads, head_dim)
    dk_hv = k.new_empty(batch_size, num_tokens, num_value_heads, head_dim)
    dw = torch.empty_like(w)
    dg_parts = torch.empty(
        num_k_blocks, *g.shape, dtype=torch.float32, device=g.device
    )
    raw_kernel = chunk_bwd_kernel_dqkwg.fn.fn
    raw_kernel[(num_k_blocks, num_chunks, batch_size * num_value_heads)](
        q,
        k,
        v,
        g,
        None,
        h,
        do,
        dh,
        dq_hv,
        dk_hv,
        dw,
        dv,
        dg_parts,
        None,
        None,
        scale,
        batch_size,
        num_tokens,
        num_key_heads,
        num_value_heads,
        head_dim,
        value_dim,
        chunk_size,
        block_k,
        block_v,
        USE_G=True,
        USE_G_GAMMA=False,
        USE_DW=True,
        USE_EXP2=False,
        TRANSPOSE_STATE=False,
        IS_VARLEN=False,
        num_warps=8,
        num_stages=1,
    )
    return dq_hv, dk_hv, dw, dg_parts


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_dqkwg_gqa_sm12x(
    H,
    HV,
    DK,
    DV,
    chunk_size,
    scale,
    qkva_dtype,
    g_dtype,
    accum_dtype,
    block_V=64,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    group_size = HV // H

    q_shape = (batch_size, num_tokens, H, DK)
    v_shape = (batch_size, num_tokens, HV, DV)
    state_shape = (batch_size, num_chunks, HV, DK, DV)
    dg_shape = (batch_size, num_tokens, HV)

    @T.prim_func
    def tilelang_dqkwg_gqa_sm12x_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(q_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        w: T.Tensor(q_shape[:-2] + (HV, DK), dtype=qkva_dtype),
        g: T.Tensor(dg_shape, dtype=g_dtype),
        h: T.Tensor(state_shape, dtype=qkva_dtype),
        dv: T.Tensor(v_shape, dtype=qkva_dtype),
        do: T.Tensor(v_shape, dtype=qkva_dtype),
        dh: T.Tensor(state_shape, dtype=qkva_dtype),
        dq: T.Tensor(q_shape, dtype=qkva_dtype),
        dk: T.Tensor(q_shape, dtype=qkva_dtype),
        dw: T.Tensor(q_shape[:-2] + (HV, DK), dtype=qkva_dtype),
        dg: T.Tensor(dg_shape, dtype=g_dtype),
    ):
        with T.Kernel(tilelang.cdiv(num_tokens, chunk_size), batch_size * H, threads=256) as (
            bc,
            bbh,
        ):
            bb = bbh // H
            bh = bbh % H
            left = bc * chunk_size

            q_shared = T.alloc_shared((chunk_size, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((chunk_size, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((chunk_size, block_V), dtype=qkva_dtype)
            do_shared = T.alloc_shared((chunk_size, block_V), dtype=qkva_dtype)
            h_shared = T.alloc_shared((DK, block_V), dtype=qkva_dtype)
            dh_shared = T.alloc_shared((DK, block_V), dtype=qkva_dtype)
            ds_shared = T.alloc_shared((chunk_size, chunk_size), dtype=qkva_dtype)
            g_shared = T.alloc_shared((chunk_size,), dtype=accum_dtype, scope="shared")

            acc_dq = T.alloc_fragment((chunk_size, DK), dtype=accum_dtype)
            acc_dk = T.alloc_fragment((chunk_size, DK), dtype=accum_dtype)
            b_dq = T.alloc_fragment((chunk_size, DK), dtype=accum_dtype)
            b_dk = T.alloc_fragment((chunk_size, DK), dtype=accum_dtype)
            b_dw = T.alloc_fragment((chunk_size, DK), dtype=accum_dtype)
            b_ds = T.alloc_fragment((chunk_size, chunk_size), dtype=accum_dtype)
            hdh = T.alloc_fragment((DK, block_V), dtype=accum_dtype)
            hdh_row = T.alloc_fragment((DK,), dtype=accum_dtype)
            hdh_sum = T.alloc_fragment((1,), dtype=accum_dtype)
            prod = T.alloc_fragment((chunk_size, DK), dtype=accum_dtype)
            red = T.alloc_fragment((chunk_size,), dtype=accum_dtype)
            red_k = T.alloc_fragment((chunk_size,), dtype=accum_dtype)
            scalar = T.alloc_fragment((1,), dtype=accum_dtype)
            dg_last = T.alloc_fragment((1,), dtype=accum_dtype)
            ds_cast = T.alloc_fragment((chunk_size, chunk_size), dtype=qkva_dtype)

            g_last = T.alloc_var("float32")

            T.copy(q[bb, left : left + chunk_size, bh, 0:DK], q_shared)
            T.copy(k[bb, left : left + chunk_size, bh, 0:DK], k_shared)
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
                    T.copy(
                        v[bb, left : left + chunk_size, bhv, v_off : v_off + block_V],
                        v_shared,
                    )
                    T.copy(
                        do[bb, left : left + chunk_size, bhv, v_off : v_off + block_V],
                        do_shared,
                    )
                    T.copy(h[bb, bc, bhv, 0:DK, v_off : v_off + block_V], h_shared)
                    T.copy(
                        dh[bb, bc, bhv, 0:DK, v_off : v_off + block_V],
                        dh_shared,
                    )

                    T.gemm_v1(do_shared, v_shared, b_ds, transpose_B=True, clear_accum=False)
                    T.gemm_v1(do_shared, h_shared, b_dq, transpose_B=True, clear_accum=False)
                    T.gemm_v1(v_shared, dh_shared, b_dk, transpose_B=True, clear_accum=False)
                    T.copy(
                        dv[bb, left : left + chunk_size, bhv, v_off : v_off + block_V],
                        v_shared,
                    )
                    T.gemm_v1(v_shared, h_shared, b_dw, transpose_B=True, clear_accum=False)

                    for j_k, j_v in T.Parallel(DK, block_V):
                        hdh[j_k, j_v] = (
                            T.cast(h_shared[j_k, j_v], "float32")
                            * T.cast(dh_shared[j_k, j_v], "float32")
                        )
                    T.reduce_sum(hdh, hdh_row, dim=1)
                    T.reduce_sum(hdh_row, hdh_sum, dim=0)
                    dg_last[0] += hdh_sum[0]

                for j_s in T.Parallel(chunk_size):
                    g_shared[j_s] = g[bb, left + j_s, bhv]
                g_last = g_shared[chunk_size - 1]
                dg_last[0] *= T.exp(g_last)

                for j_s, j_k in T.Parallel(chunk_size, DK):
                    b_dq[j_s, j_k] *= T.exp(g_shared[j_s]) * scale
                    b_dk[j_s, j_k] *= T.exp(g_last - g_shared[j_s])

                for j_s, j_k in T.Parallel(chunk_size, DK):
                    prod[j_s, j_k] = b_dk[j_s, j_k] * T.cast(
                        k_shared[j_s, j_k], "float32"
                    )
                T.reduce_sum(prod, red, dim=1)
                T.reduce_sum(red, scalar, dim=0)
                dg_last[0] += scalar[0]

                for j_s, j_t in T.Parallel(chunk_size, chunk_size):
                    if j_s >= j_t:
                        b_ds[j_s, j_t] *= (
                            T.exp(g_shared[j_s] - g_shared[j_t]) * scale
                        )
                    else:
                        b_ds[j_s, j_t] = 0.0
                    ds_cast[j_s, j_t] = b_ds[j_s, j_t]

                T.copy(ds_cast, ds_shared)
                T.gemm_v1(ds_shared, k_shared, b_dq, clear_accum=False)
                T.gemm_v1(ds_shared, q_shared, b_dk, transpose_A=True, clear_accum=False)

                for j_s, j_k in T.Parallel(chunk_size, DK):
                    prod[j_s, j_k] = b_dq[j_s, j_k] * T.cast(
                        q_shared[j_s, j_k], "float32"
                    )
                T.reduce_sum(prod, red, dim=1)
                for j_s, j_k in T.Parallel(chunk_size, DK):
                    prod[j_s, j_k] = b_dk[j_s, j_k] * T.cast(
                        k_shared[j_s, j_k], "float32"
                    )
                T.reduce_sum(prod, red_k, dim=1)

                for j_s in T.Parallel(chunk_size):
                    dg[bb, left + j_s, bhv] = red[j_s] - red_k[j_s]
                dg[bb, left + chunk_size - 1, bhv] += dg_last[0]

                for j_s, j_k in T.Parallel(chunk_size, DK):
                    acc_dq[j_s, j_k] += b_dq[j_s, j_k]
                    acc_dk[j_s, j_k] += b_dk[j_s, j_k]
                    dw[bb, left + j_s, bhv, j_k] = -b_dw[j_s, j_k]

            for j_s, j_k in T.Parallel(chunk_size, DK):
                dq[bb, left + j_s, bh, j_k] = acc_dq[j_s, j_k]

            for j_s, j_k in T.Parallel(chunk_size, DK):
                dk[bb, left + j_s, bh, j_k] = acc_dk[j_s, j_k]

    return tilelang_dqkwg_gqa_sm12x_kernel


def _chunk_bwd_dqkwg_gqa_direct_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    dv: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
    chunk_size: int,
):
    batch_size, num_tokens, num_key_heads, head_dim = k.shape
    num_value_heads = v.shape[2]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w)
    dg = torch.empty_like(g)
    kernel = tilelang_dqkwg_gqa_sm12x(
        H=num_key_heads,
        HV=num_value_heads,
        DK=head_dim,
        DV=v.shape[-1],
        chunk_size=chunk_size,
        scale=scale,
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        accum_dtype="float32",
        block_V=64,
    )
    kernel(q, k, v, w, g, h, dv, do, dh, dq, dk, dw, dg)
    return dq, dk, dw, dg


def _fla2_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int,
    h: torch.Tensor | None = None,
    v_new: torch.Tensor | None = None,
    w: torch.Tensor | None = None,
):
    ops = _get_fla2_bwd_ops()
    if ops is None:
        raise RuntimeError("FLA2 gated delta rule backward is not available")
    (
        recompute_w_u_fwd,
        chunk_gated_delta_rule_fwd_h,
        chunk_bwd_dv_local,
        chunk_gated_delta_rule_bwd_dhu,
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
        chunk_bwd_dqkwg,
        chunk_bwd_kernel_dqkwg,
        prepare_wy_repr_bwd,
        prepare_wy_repr_bwd_kernel,
        fla_chunk_local_cumsum,
    ) = ops
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    if h is not None and v_new is not None:
        if w is None:
            w = _recompute_w_fwd(
                k=k,
                beta=b,
                a=a,
                g=g,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=chunk_size,
            )
    else:
        w, u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=b,
            A=a,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=False,
        )
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=False,
        )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=False,
    )
    if (
        cu_seqlens is None
        and q.shape[-1] == do.shape[-1] == 128
        and chunk_size == 64
    ):
        dh, dh0, dv = _chunk_bwd_dhu_sm12x_tuned(
            chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64=(
                chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64
            ),
            q=q,
            k=k,
            w=w,
            g=g,
            initial_state=initial_state,
            dht=dht,
            do=do,
            dv=dv,
            scale=scale,
            chunk_size=chunk_size,
        )
    else:
        dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
            q=q,
            k=k,
            w=w,
            g=g,
            h0=initial_state,
            dht=dht,
            do=do,
            dv=dv,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=False,
        )
    use_gqa_sm12x_path = (
        cu_seqlens is None
        and v.shape[2] > k.shape[2]
        and v.shape[2] % k.shape[2] == 0
        and k.shape[-1] == v.shape[-1] == 128
        and chunk_size == 64
    )
    use_gqa_direct_tilelang = (
        use_gqa_sm12x_path and q.shape[1] % chunk_size == 0
    )
    if use_gqa_direct_tilelang:
        dq, dk, dw, dg = _chunk_bwd_dqkwg_gqa_direct_tilelang(
            q=q,
            k=k,
            v=v_new,
            w=w,
            g=g,
            h=h,
            dv=dv,
            do=do,
            dh=dh,
            scale=scale,
            chunk_size=chunk_size,
        )
    elif use_gqa_sm12x_path:
        dq_hv, dk_hv, dw, dg_parts = _chunk_bwd_dqkwg_sm12x_tuned_raw(
            chunk_bwd_kernel_dqkwg=chunk_bwd_kernel_dqkwg,
            q=q,
            k=k,
            v=v_new,
            w=w,
            g=g,
            h=h,
            dv=dv,
            do=do,
            dh=dh,
            scale=scale,
            chunk_size=chunk_size,
        )
    else:
        dq, dk, dw, dg = chunk_bwd_dqkwg(
            q=q,
            k=k,
            v=v_new,
            w=w,
            g=g,
            h=h,
            dv=dv,
            do=do,
            dh=dh,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=False,
        )
    if k.shape[2] == v.shape[2] and k.shape[-1] == v.shape[-1] == 128:
        dk, dv, db, dg = _prepare_wy_repr_bwd_accum(
            k=k,
            v=v,
            beta=b,
            g=g,
            a=a,
            dw=dw,
            du=dv,
            dk=dk,
            dg=dg,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )
    elif (
        v.shape[2] % k.shape[2] == 0
        and k.shape[-1] == v.shape[-1] == 128
        and cu_seqlens is None
    ):
        if use_gqa_direct_tilelang:
            dv, db, dg = _prepare_wy_repr_bwd_gqa_tilelang_reduce_add(
                prepare_wy_repr_bwd_kernel=prepare_wy_repr_bwd_kernel,
                k=k,
                v=v,
                beta=b,
                g=g,
                a=a,
                dw=dw,
                du=dv,
                dk=dk,
                dg_base=dg,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=chunk_size,
            )
        elif use_gqa_sm12x_path:
            dk_wy_hv, dv, db, dg_wy = _prepare_wy_repr_bwd_gqa_raw(
                prepare_wy_repr_bwd_kernel=prepare_wy_repr_bwd_kernel,
                k=k,
                v=v,
                beta=b,
                g=g,
                a=a,
                dw=dw,
                du=dv,
                chunk_size=chunk_size,
            )
            dq, dk, dg = _reduce_bwd_gqa_combined(
                dq_hv=dq_hv,
                dk_hv=dk_hv,
                dk_wy_hv=dk_wy_hv,
                dg_parts=dg_parts,
                dg_wy=dg_wy,
                num_key_heads=k.shape[2],
            )
        else:
            dv, db, dg = _prepare_wy_repr_bwd_gqa_tilelang_reduce_add(
                prepare_wy_repr_bwd_kernel=prepare_wy_repr_bwd_kernel,
                k=k,
                v=v,
                beta=b,
                g=g,
                a=a,
                dw=dw,
                du=dv,
                dk=dk,
                dg_base=dg,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=chunk_size,
            )
    else:
        dk2, dv, db, dg2 = prepare_wy_repr_bwd(
            k=k,
            v=v,
            beta=b,
            g=g,
            A=a,
            dw=dw,
            du=dv,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=False,
        )
        _add_bwd_partials(dk, dk2, dg, dg2)
    dg = fla_chunk_local_cumsum(
        dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens
    )
    return dq, dk, dv, db, dg, dh0


def _chunk_dv_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    scale: float,
):
    chunk_size = q.shape[2]
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    decay_mask = torch.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(mask[None, None, :, :, None], 0.0)
    attn = torch.einsum("bnchk,bndhk->bncdh", q * scale, k) * decay_mask
    return torch.einsum("bncdh,bnchv->bndhv", attn, do)


def _chunk_gdr_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
):
    batch_size, num_chunks, _, num_heads, head_dim_k = q.shape
    head_dim_v = do.shape[-1]
    if dht is None:
        dstate = torch.zeros(
            (batch_size, num_heads, head_dim_k, head_dim_v),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        dstate = dht

    dstate_inter = torch.einsum(
        "bnchk,bnchv->bnhkv", q * scale * g.exp().unsqueeze(-1), do
    )

    dh = []
    for chunk_idx in reversed(range(num_chunks)):
        dh.insert(0, dstate)
        chunk_decay = (
            g[:, chunk_idx, -1:, :, None] - g[:, chunk_idx, :, :, None]
        ).exp()
        dv[:, chunk_idx] += torch.einsum(
            "bchk,bhkv->bchv", k[:, chunk_idx] * chunk_decay, dstate
        )
        dstate = dstate * g[:, chunk_idx, -1, :, None, None].exp()
        dstate = (
            dstate
            + dstate_inter[:, chunk_idx]
            - torch.einsum("bchk,bchv->bhkv", w[:, chunk_idx], dv[:, chunk_idx])
        )

    return torch.stack(dh, dim=1).contiguous(), dstate if dht is not None else None, dv


def _chunk_dqkwg_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    vn: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    dv: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
    num_tokens: int,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int,
):
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    decay_mask = torch.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(mask[None, None, :, :, None], 0.0)

    dg_last = (h * dh).sum(dim=-1).sum(dim=-1)
    ds = torch.einsum("bnchv,bndhv->bncdh", do, vn)
    dq = torch.einsum("bnchv,bnhkv->bnchk", do, h)
    dk = torch.einsum("bnchv,bnhkv->bnchk", vn, dh)
    dw = -torch.einsum("bnchv,bnhkv->bnchk", dv, h)

    g_last = g[:, :, -1]
    dg_last *= g_last.exp()
    dq = dq * g.exp().unsqueeze(-1) * scale
    dg = (q * dq).sum(dim=-1)
    dk = dk * (g_last.unsqueeze(-2) - g).unsqueeze(-1).exp()
    k_dot_dk = (k * dk).sum(dim=-1)
    dg -= k_dot_dk
    dg_last += k_dot_dk.sum(dim=-2)
    ds *= decay_mask * scale
    ds2 = ds * torch.einsum("bnchk,bndhk->bncdh", q, k)
    dg += ds2.sum(dim=-2)
    dg -= ds2.sum(dim=-3)
    dq += torch.einsum("bncdh,bndhk->bnchk", ds, k)
    dk += torch.einsum("bncdh,bnchk->bndhk", ds, q)
    dg[:, :, -1] += dg_last

    dg = fill_last_chunk_of_g(
        dg, num_tokens, cu_seqlens, chunk_size=chunk_size, reverse=True
    )
    return dq, dk, dw, dg


def _chunk_wy_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dk1: torch.Tensor,
    dg1: torch.Tensor,
    chunk_size: int,
):
    beta = b
    dA = torch.einsum(
        "bnchk,bndhk->bnchd", dw, k * (beta * g.exp()).unsqueeze(-1)
    )
    dk_beta_g = torch.einsum("bnchd,bnchk->bndhk", a, dw)
    dk = dk_beta_g * (beta * g.exp()).unsqueeze(-1)
    db = (dk_beta_g * k * g.exp().unsqueeze(-1)).sum(dim=-1)
    dg = (dk_beta_g * k * (g.exp() * beta).unsqueeze(-1)).sum(dim=-1)

    dA += torch.einsum("bnchv,bndhv->bnchd", du, v * beta.unsqueeze(-1))
    dv_beta = torch.einsum("bnchd,bnchv->bndhv", a, du)
    dv = dv_beta * beta.unsqueeze(-1)
    db += (dv_beta * v).sum(dim=-1)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device)
    )
    decay_mask = torch.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(mask[None, None, :, :, None], 0.0).swapaxes(
        -2, -1
    )
    dA = dA.masked_fill(mask[None, None, :, None, :], 0.0)
    dA = torch.einsum("bndhc,bndhe->bnche", a, dA)
    dA = torch.einsum("bnchd,bnehd->bnche", dA, a)
    dA = -dA * decay_mask

    kk = torch.einsum("bnchk,bndhk->bnchd", k * beta.unsqueeze(-1), k)
    dk_beta = torch.einsum("bnchd,bndhk->bnchk", dA, k)
    db += (dk_beta * k).sum(dim=-1)
    dk += torch.einsum("bnchd,bnchk->bndhk", dA, k * beta.unsqueeze(-1))
    dk += dk_beta * beta.unsqueeze(-1)
    dk += dk1

    dA_kk = dA * kk
    dg += dA_kk.sum(dim=-1) - dA_kk.sum(dim=-3).swapaxes(-1, -2)
    dg += dg1

    return dk, dv, db, dg


def _flatten_chunks(x: torch.Tensor, num_tokens: int):
    batch_size = x.shape[0]
    return x.reshape(batch_size, -1, *x.shape[3:])[:, :num_tokens].contiguous()


def fused_gdr_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    h: torch.Tensor | None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    v_new: torch.Tensor | None = None,
    w: torch.Tensor | None = None,
):
    _, _, num_key_heads, K = k.shape
    _, _, num_value_heads, V = v.shape
    scale = scale or K ** (-0.5)
    assert K == V == 128
    assert chunk_size == 64

    if _fla2_bwd_available():
        return _fla2_bwd(
            q=q,
            k=k,
            v=v,
            a=a,
            g=g,
            b=b,
            do=do,
            dht=dht,
            initial_state=initial_state,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            h=h,
            v_new=v_new,
            w=w,
        )

    q_dtype = q.dtype
    k_dtype = k.dtype
    v_dtype = v.dtype
    use_dht = dht is not None
    k_tokens = k
    v_tokens = v
    g_tokens = g
    b_tokens = b

    q, k, v, g, b, do, h, dht, num_tokens = _chunk_inputs(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        do=do,
        h=h,
        dht=dht,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    # Blackwell KKT is stored in q/k/v dtype by the forward path. Backward is
    # more sensitive to this inverse, so recompute it in fp32 from k, g, beta.
    a = _kkt_solve_fwd(k=k, g=g, b=b, chunk_size=chunk_size)

    if h is None:
        from .prepare_h import fused_gdr_h

        a_tokens = _maybe_pack_tokens(_flatten_chunks(a, num_tokens), cu_seqlens)
        h, _, _ = fused_gdr_h(
            k=k_tokens,
            v=v_tokens,
            a=a_tokens,
            g=g_tokens,
            b=b_tokens,
            initial_state=initial_state,
            output_final_state=False,
            output_h=True,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        h = _maybe_unpack_chunks(h, cu_seqlens, chunk_size)
        if h is not None:
            h = h.float()

    w, u = _wy_fwd(k=k, v=v, a=a, g=g, b=b)
    vn = u - torch.einsum("bnchk,bnhkv->bnchv", w, h)
    dv = _chunk_dv_bwd(q=q, k=k, g=g, do=do, scale=scale)
    dh, dh0, dv = _chunk_gdr_bwd(
        q=q,
        k=k,
        w=w,
        g=g,
        do=do,
        dv=dv,
        dht=dht,
        scale=scale,
    )
    dq, dk1, dw, dg1 = _chunk_dqkwg_bwd(
        q=q,
        k=k,
        vn=vn,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        num_tokens=num_tokens,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dk, dv, db, dg = _chunk_wy_bwd(
        k=k,
        v=v,
        b=b,
        a=a,
        g=g,
        dw=dw,
        du=dv,
        dk1=dk1,
        dg1=dg1,
        chunk_size=chunk_size,
    )

    dq = _flatten_chunks(dq, num_tokens)
    dk = _flatten_chunks(dk, num_tokens)
    dv = _flatten_chunks(dv, num_tokens)
    db = _flatten_chunks(db, num_tokens)
    dg = _flatten_chunks(dg, num_tokens)

    dq = _maybe_pack_tokens(dq, cu_seqlens)
    dk = _maybe_pack_tokens(dk, cu_seqlens)
    dv = _maybe_pack_tokens(dv, cu_seqlens)
    db = _maybe_pack_tokens(db, cu_seqlens)
    dg = _maybe_pack_tokens(dg, cu_seqlens)
    dg = chunk_local_cumsum(
        dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens
    )

    if not use_dht:
        dh0 = None

    if num_key_heads < num_value_heads:
        batch_size, num_tokens, _, head_dim = dq.shape
        group_size = num_value_heads // num_key_heads
        dq = dq.reshape(batch_size, num_tokens, num_key_heads, group_size, head_dim)
        dk = dk.reshape(batch_size, num_tokens, num_key_heads, group_size, head_dim)
        dq = dq.sum(dim=3)
        dk = dk.sum(dim=3)

    return (
        dq.to(q_dtype),
        dk.to(k_dtype),
        dv.to(v_dtype),
        db,
        dg,
        dh0,
    )
