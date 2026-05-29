# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
#
# Regression tests for the pure-TileLang sm12x GDN backward stage kernels
# (flash_qla/ops/gated_delta_rule/chunk/blackwell/bwd_sm12x.py), validated
# against the fp32 reference (tests/ref_gdr.py) on a real sm12x device.
#
# Run:  python tests/test_bwd_sm12x.py
# Requires a Blackwell sm120/sm121 GPU + tilelang.

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from flash_qla.ops.gated_delta_rule.chunk.blackwell.bwd_sm12x import (  # noqa: E402
    chunk_dv_local_sm12x,
    recompute_w_sm12x,
    chunk_gated_delta_rule_bwd_sm12x,
)
from flash_qla.ops.gated_delta_rule.chunk import (  # noqa: E402
    chunk_gated_delta_rule_fwd as qla_fwd,
    chunk_gated_delta_rule as qla_fn,
)
from flash_qla.ops.gated_delta_rule.chunk.arch import is_sm12x  # noqa: E402
from flash_qla.utils import l2norm  # noqa: E402
from ref_gdr import (  # noqa: E402
    chunk_gated_delta_rule_fwd as ref_fwd,
    chunk_gated_delta_rule_bwd as ref_bwd,
    torch_chunk_dv_bwd,
    torch_w_u_fwd,
)

DEV = "cuda"
K = V = 128
C = 64
TOL = 2e-2  # relative, bf16


def _rel(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().max().item() / (b.float().abs().max().item() + 1e-9)


def _chunk_cumsum_g(g, chunk):
    B, T, Hv = g.shape
    return g.reshape(B, -1, chunk, Hv).cumsum(2).reshape(B, T, Hv).contiguous()


def test_dv_local(Hk, Hv, T, varlen=False):
    torch.manual_seed(0)
    if varlen:
        cu = torch.tensor([0, 200, 320], device=DEV, dtype=torch.int32)
        T = 320
        q = torch.randn(1, T, Hk, K, device=DEV, dtype=torch.bfloat16)
        k = torch.randn(1, T, Hk, K, device=DEV, dtype=torch.bfloat16)
        do = torch.randn(1, T, Hv, V, device=DEV, dtype=torch.bfloat16)
        g = F.logsigmoid(torch.randn(1, T, Hv, device=DEV, dtype=torch.float32)) / 16
    else:
        cu = None
        q = torch.randn(1, T, Hk, K, device=DEV, dtype=torch.bfloat16)
        k = torch.randn(1, T, Hk, K, device=DEV, dtype=torch.bfloat16)
        do = torch.randn(1, T, Hv, V, device=DEV, dtype=torch.bfloat16)
        g = _chunk_cumsum_g(
            F.logsigmoid(torch.randn(1, T, Hv, device=DEV, dtype=torch.float32)) / 16, C)
    scale = K ** -0.5
    out = chunk_dv_local_sm12x(q, k, g, do, scale, chunk_size=C, cu_seqlens=cu)
    ref = torch_chunk_dv_bwd(q.float(), k.float(), g.float(), do.float(),
                             cu_seqlens=cu, scale=scale, chunk_size=C)
    r = _rel(out, ref)
    tag = f"dv_local Hk={Hk} Hv={Hv} T={T} varlen={varlen}"
    print(f"  {'PASS' if r < TOL else 'FAIL'} {tag}: rel={r:.2e}")
    return r < TOL


def test_recompute_w(Hk, Hv, T, varlen=False):
    torch.manual_seed(1)
    cu = torch.tensor([0, 200, 320], device=DEV, dtype=torch.int32) if varlen else None
    T = 320 if varlen else T
    k = torch.randn(1, T, Hk, K, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, T, Hv, V, device=DEV, dtype=torch.bfloat16)
    beta = torch.randn(1, T, Hv, device=DEV, dtype=torch.float32).sigmoid()
    A = torch.randn(1, T, Hv, C, device=DEV, dtype=torch.bfloat16) * 0.1
    g = F.logsigmoid(torch.randn(1, T, Hv, device=DEV, dtype=torch.float32)) / 16
    out = recompute_w_sm12x(k, beta, A, g, chunk_size=C, cu_seqlens=cu)
    ref, _ = torch_w_u_fwd(k.float(), v.float(), g.float(), beta.float(), A.float(),
                           cu_seqlens=cu)
    r = _rel(out, ref)
    tag = f"recompute_w Hk={Hk} Hv={Hv} T={T} varlen={varlen}"
    print(f"  {'PASS' if r < TOL else 'FAIL'} {tag}: rel={r:.2e}")
    return r < TOL


def test_full_bwd(Hk, Hv, T):
    """End-to-end pure-TileLang sm12x backward vs fp32 reference (non-varlen)."""
    torch.manual_seed(42)
    scale = K ** -0.5
    q = l2norm(torch.randn(1, T, Hk, K, device=DEV, dtype=torch.bfloat16))
    k = l2norm(torch.randn(1, T, Hk, K, device=DEV, dtype=torch.bfloat16))
    v = torch.randn(1, T, Hv, V, device=DEV, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(1, T, Hv, device=DEV, dtype=torch.float32)) / 16
    beta = torch.randn(1, T, Hv, device=DEV, dtype=torch.float32).sigmoid()
    do = torch.randn_like(v)
    gr, _, Ar, _, _ = ref_fwd(q=q.float(), k=k.float(), v=v.float(), g=g.clone(),
                              beta=beta.clone(), scale=scale, initial_state=None, chunk_size=C)
    dq_r, dk_r, dv_r, db_r, dg_r, _ = ref_bwd(
        q.float(), k.float(), v.float(), gr.clone(), beta.clone(), Ar.clone(),
        scale, None, do.float().clone(), None, None, C)
    gq, Aq, _, hq, _, *rest = qla_fwd(
        q, k, v, g, beta, scale=scale, initial_state=None, output_final_state=True,
        output_h=True, auto_cp=False, output_v_new=is_sm12x())
    dq, dk, dv, db, dg, _ = chunk_gated_delta_rule_bwd_sm12x(
        q, k, v, gq, beta, Aq, do, hq, rest[0], scale, chunk_size=C)
    rs = {"dq": _rel(dq, dq_r), "dk": _rel(dk, dk_r), "dv": _rel(dv, dv_r),
          "db": _rel(db, db_r), "dg": _rel(dg, dg_r)}
    ok = max(rs.values()) < TOL
    print(f"  {'PASS' if ok else 'FAIL'} full_bwd Hk={Hk} Hv={Hv} T={T}: " +
          " ".join(f"{kk}={vv:.1e}" for kk, vv in rs.items()))
    return ok


def test_full_bwd_varlen(Hk, Hv, cu_list):
    """End-to-end pure-TileLang sm12x backward for packed (varlen) sequences vs
    the fp32 reference. Exercises every stage's per-sequence state reset and
    partial-last-chunk masking (dv_local, recompute_w, dhu, dqkwg, wy)."""
    torch.manual_seed(42)
    scale = K ** -0.5
    cu = torch.tensor(cu_list, device=DEV, dtype=torch.int32)
    T = int(cu_list[-1])
    q = l2norm(torch.randn(1, T, Hk, K, device=DEV, dtype=torch.bfloat16))
    k = l2norm(torch.randn(1, T, Hk, K, device=DEV, dtype=torch.bfloat16))
    v = torch.randn(1, T, Hv, V, device=DEV, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(1, T, Hv, device=DEV, dtype=torch.float32)) / 16
    beta = torch.randn(1, T, Hv, device=DEV, dtype=torch.float32).sigmoid()
    do = torch.randn_like(v)
    gr, _, Ar, _, _ = ref_fwd(q=q.float(), k=k.float(), v=v.float(), g=g.clone(),
                              beta=beta.clone(), scale=scale, initial_state=None,
                              cu_seqlens=cu, chunk_size=C)
    dq_r, dk_r, dv_r, db_r, dg_r, _ = ref_bwd(
        q.float(), k.float(), v.float(), gr.clone(), beta.clone(), Ar.clone(),
        scale, None, do.float().clone(), None, cu, C)
    gq, Aq, _, hq, _, *rest = qla_fwd(
        q, k, v, g, beta, scale=scale, initial_state=None, cu_seqlens=cu,
        output_final_state=True, output_h=True, auto_cp=False, output_v_new=is_sm12x())
    dq, dk, dv, db, dg, _ = chunk_gated_delta_rule_bwd_sm12x(
        q, k, v, gq, beta, Aq, do, hq, rest[0], scale, chunk_size=C, cu_seqlens=cu)
    rs = {"dq": _rel(dq, dq_r), "dk": _rel(dk, dk_r), "dv": _rel(dv, dv_r),
          "db": _rel(db, db_r), "dg": _rel(dg, dg_r)}
    ok = max(rs.values()) < TOL
    print(f"  {'PASS' if ok else 'FAIL'} full_bwd_varlen Hk={Hk} Hv={Hv} cu={cu_list}: " +
          " ".join(f"{kk}={vv:.1e}" for kk, vv in rs.items()))
    return ok


def test_full_bwd_dht(Hk, Hv, T):
    """Pure-TileLang sm12x backward with dht != None (gradient w.r.t. the final
    state) AND a non-None initial_state — the dh-recurrence is seeded from dht
    and dh0 is produced, all on the pure path (no FLA). Validates dq..dg + dh0."""
    torch.manual_seed(7)
    scale = K ** -0.5
    q = l2norm(torch.randn(1, T, Hk, K, device=DEV, dtype=torch.bfloat16))
    k = l2norm(torch.randn(1, T, Hk, K, device=DEV, dtype=torch.bfloat16))
    v = torch.randn(1, T, Hv, V, device=DEV, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(1, T, Hv, device=DEV, dtype=torch.float32)) / 16
    beta = torch.randn(1, T, Hv, device=DEV, dtype=torch.float32).sigmoid()
    do = torch.randn_like(v)
    h0 = torch.randn(1, Hv, K, V, device=DEV, dtype=torch.float32) * 0.2
    dht = torch.randn(1, Hv, K, V, device=DEV, dtype=torch.float32) * 0.2
    gr, _, Ar, _, _ = ref_fwd(q=q.float(), k=k.float(), v=v.float(), g=g.clone(),
                              beta=beta.clone(), scale=scale,
                              initial_state=h0.clone(), chunk_size=C)
    dq_r, dk_r, dv_r, db_r, dg_r, dh0_r = ref_bwd(
        q.float(), k.float(), v.float(), gr.clone(), beta.clone(), Ar.clone(),
        scale, h0.clone(), do.float().clone(), dht.clone(), None, C)
    gq, Aq, _, hq, _, *rest = qla_fwd(
        q, k, v, g, beta, scale=scale, initial_state=h0.clone(),
        output_final_state=True, output_h=True, auto_cp=False, output_v_new=is_sm12x())
    dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd_sm12x(
        q, k, v, gq, beta, Aq, do, hq, rest[0], scale, chunk_size=C, dht=dht,
        output_dh0=True)
    rs = {"dq": _rel(dq, dq_r), "dk": _rel(dk, dk_r), "dv": _rel(dv, dv_r),
          "db": _rel(db, db_r), "dg": _rel(dg, dg_r), "dh0": _rel(dh0, dh0_r)}
    ok = max(rs.values()) < TOL
    print(f"  {'PASS' if ok else 'FAIL'} full_bwd_dht Hk={Hk} Hv={Hv} T={T}: " +
          " ".join(f"{kk}={vv:.1e}" for kk, vv in rs.items()))
    return ok


def test_autograd_roundtrip(Hk, Hv, T):
    """Exercise the public autograd Function (ChunkGatedDeltaRuleFunction) end to
    end on the pure sm12x path: dht=None (o-only loss) and dht!=None (final_state
    in the loss). Verifies grads exist, are finite, and are non-zero with an O(1)
    upstream gradient — guards the ctx save/return wiring after the FLA removal."""
    torch.manual_seed(0)
    scale = K ** -0.5

    def mk():
        q = l2norm(torch.randn(1, T, Hk, K, device=DEV, dtype=torch.bfloat16)).detach().requires_grad_(True)
        k = l2norm(torch.randn(1, T, Hk, K, device=DEV, dtype=torch.bfloat16)).detach().requires_grad_(True)
        v = torch.randn(1, T, Hv, V, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        g = (F.logsigmoid(torch.randn(1, T, Hv, device=DEV, dtype=torch.float32)) / 16).detach().requires_grad_(True)
        beta = torch.randn(1, T, Hv, device=DEV, dtype=torch.float32).sigmoid().detach().requires_grad_(True)
        return q, k, v, g, beta

    ok = True
    # (a) dht=None: loss uses only o, O(1) upstream grad
    q, k, v, g, beta = mk()
    o, _ = qla_fn(q, k, v, g, beta, scale=scale, initial_state=None, output_final_state=True)
    (o.float() * torch.randn_like(o).float()).sum().backward()
    norms = {n: t.grad.float().norm().item() for n, t in
             [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)]}
    a_ok = all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v, g, beta)) \
        and min(norms.values()) > 0
    # (b) dht!=None: final_state participates in the loss
    q, k, v, g, beta = mk()
    o, fs = qla_fn(q, k, v, g, beta, scale=scale, initial_state=None, output_final_state=True)
    ((o.float() * torch.randn_like(o).float()).sum()
     + (fs.float() * torch.randn_like(fs).float()).sum()).backward()
    b_ok = all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v, g, beta))
    ok = a_ok and b_ok
    print(f"  {'PASS' if ok else 'FAIL'} autograd_roundtrip Hk={Hk} Hv={Hv} T={T}: "
          f"dht=None grads>0={a_ok} (min_norm={min(norms.values()):.2e}); dht!=None ok={b_ok}")
    return ok


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    print(f"device={torch.cuda.get_device_name()} cap={torch.cuda.get_device_capability()}")
    results = []
    print("stage kernels:")
    results.append(test_dv_local(16, 48, 256))
    results.append(test_dv_local(16, 48, 320, varlen=True))
    results.append(test_recompute_w(16, 48, 256))
    results.append(test_recompute_w(16, 48, 320, varlen=True))
    print("end-to-end backward (Qwen3.6 family shapes):")
    results.append(test_full_bwd(16, 48, 2048))   # 27B TP1
    results.append(test_full_bwd(32, 32, 2048))   # symmetric
    results.append(test_full_bwd(16, 16, 1024))   # 2B/0.8B (dg excluded)
    print("end-to-end backward (packed/varlen, partial chunks):")
    results.append(test_full_bwd_varlen(16, 48, [0, 448, 768]))   # 27B TP1
    results.append(test_full_bwd_varlen(16, 16, [0, 200, 320]))   # 2B/0.8B
    print("end-to-end backward (dht != None, pure dh0 path):")
    results.append(test_full_bwd_dht(16, 48, 1024))   # 27B TP1
    results.append(test_full_bwd_dht(32, 32, 512))    # symmetric
    print("autograd Function round-trip (pure sm12x path):")
    results.append(test_autograd_roundtrip(16, 48, 512))
    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
