import os

import torch
import triton
import triton.language as tl

from flash_qla.ops.utils import chunk_local_cumsum
from flash_qla.ops.gated_delta_rule.chunk.hopper.fused_bwd import (
    tilelang_fused_chunk_gdr_bwd,
)
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
        )
        from fla.ops.common.chunk_o import (
            chunk_bwd_dqkwg,
            chunk_bwd_dv_local,
        )
        from fla.ops.gated_delta_rule.wy_fast import (
            prepare_wy_repr_bwd,
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
            chunk_bwd_dqkwg,
            prepare_wy_repr_bwd,
            fla_chunk_local_cumsum,
        )
    return _FLA2_BWD_OPS


def _fla2_bwd_available():
    return _get_fla2_bwd_ops() is not None


def is_fla2_bwd_available():
    return _fla2_bwd_available()


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
        grid = (triton.cdiv(num_tokens, chunk_size), batch_size * num_heads)
        chunk_indices = torch.empty((0, 2), dtype=torch.int32, device=a.device)
        cu_seqlens_arg = torch.empty((0,), dtype=torch.int32, device=a.device)
        is_varlen = False
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        cu_seqlens_arg = cu_seqlens
        grid = (len(chunk_indices), num_heads)
        is_varlen = True

    _prepare_fla2_bwd_a_kernel[grid](
        a,
        g,
        cu_seqlens_arg,
        chunk_indices,
        T=num_tokens,
        H=num_heads,
        BT=chunk_size,
        IS_VARLEN=is_varlen,
    )
    return a


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
):
    ops = _get_fla2_bwd_ops()
    if ops is None:
        raise RuntimeError("FLA2 gated delta rule backward is not available")
    (
        recompute_w_u_fwd,
        chunk_gated_delta_rule_fwd_h,
        chunk_bwd_dv_local,
        chunk_gated_delta_rule_bwd_dhu,
        chunk_bwd_dqkwg,
        prepare_wy_repr_bwd,
        fla_chunk_local_cumsum,
    ) = ops
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    if h is not None and v_new is not None:
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
