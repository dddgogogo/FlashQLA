#!/usr/bin/env python3
"""bf16 vs f32 intermediate-scratch parity for the warp verify kernel.

Covers `tilelang_flashqla_gdn_regular_bv32_warp`'s `intermediate_dtype`
parameter (the DFlash verify-scratch bf16 variant): the recurrence accumulates
fp32 in registers; only the per-step `intermediate_states_buffer` store (and
the tree parent-hop reload) round-trips through bf16. `h0_source` stays f32.

Invariants:
  1. linear verify (CACHE_INTERMEDIATE=1, TREE=0): output is BITWISE identical
     across scratch dtypes — the buffer is write-only on that path;
  2. the bf16 buffer equals the f32 buffer rounded to bf16, elementwise;
  3. the f32 kernel matches a float32 CPU reference (cos > 0.9999);
  4. tree verify with chain parents (parent == step-1) reproduces linear
     verify bitwise on both dtypes (no reload);
  5. tree verify with a real parent hop matches a CPU reference that reloads
     the per-dtype-rounded buffered state, and the pre-hop steps stay bitwise
     equal across dtypes.

Each kernel variant runs in its own subprocess: TileLang 0.1.9's NVRTC
adapter segfaults in cuLaunchKernelEx when several @tilelang.jit kernels are
launched from one process (the production compile path only exports cubins
per-process, so this is a test-harness-only constraint).

Run (needs an sm12x GPU + nvcc on PATH):
    python tests/test_recurrent_verify_scratch_dtype.py
"""
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FLASHQLA = Path(__file__).resolve().parents[1]

# ---- shared geometry ----
H, HV, DK, DV = 1, 2, 128, 128
T = 4          # tokens_per_seq (== verify_q_len == cache_steps)
N = 2          # sequences
CACHE = T
BLOCK_DV = 4
SCALE = 1.0 / math.sqrt(DK)
SP_BETA, SP_THRESH = 1.0, 20.0
SEED = 0xDF1A

VARIANTS = {
    # name: (has_tree, inter_bf16, parents_kind)
    "lin_f32": (False, False, "chain"),
    "lin_bf16": (False, True, "chain"),
    "tree_f32_chain": (True, False, "chain"),
    "tree_bf16_chain": (True, True, "chain"),
    "tree_f32_hop": (True, False, "hop"),
    "tree_bf16_hop": (True, True, "hop"),
}


def make_inputs(torch, dev):
    g = torch.Generator(device="cpu").manual_seed(SEED)
    total = N * T

    def r(*shape, s=0.4):
        return torch.randn(*shape, generator=g) * s

    q = r(1, total, H, DK).bfloat16().to(dev)
    k = r(1, total, H, DK).bfloat16().to(dev)
    v = r(1, total, HV, DV).bfloat16().to(dev)
    a = r(1, total, HV).bfloat16().to(dev)
    b = r(1, total, HV).bfloat16().to(dev)
    A_log = (-0.1 - torch.rand(HV, generator=g) * 0.5).float().to(dev)
    dt_bias = r(HV).bfloat16().to(dev)
    h0 = r(N, HV, DV, DK, s=0.2).float().to(dev)
    return q, k, v, a, b, A_log, dt_bias, h0


def make_parents(torch, kind, dev):
    chain = torch.stack(
        [torch.arange(-1, CACHE - 1, dtype=torch.int32)] * N
    ).contiguous()
    if kind == "hop":
        chain[:, 2] = 0  # step2 re-parents to step0 -> buffer reload
    return chain.to(dev)


def do_run(variant, out_dir):
    has_tree, inter_bf16, parents_kind = VARIANTS[variant]
    os.environ.setdefault("TILELANG_EXECUTION_BACKEND", "nvrtc")
    os.environ.setdefault("TILELANG_CACHE_DIR", str(out_dir / "tilelang_cache"))
    os.environ["FLASHQLA_COMPILE_BACKEND"] = "nvcc"
    sys.path.insert(0, str(FLASHQLA))
    import importlib.util

    def load_mod(key, rel):
        spec = importlib.util.spec_from_file_location(f"vsd_{key}", FLASHQLA / rel)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"vsd_{key}"] = mod
        spec.loader.exec_module(mod)
        return mod

    compile_mod = load_mod("compile", "flash_qla/compile_sm12x_gdn.py")
    import torch

    torch.cuda.set_device(0)
    compile_mod.patch_tilelang_compat()
    compile_mod.patch_tilelang_nvrtc_scalar_params()
    compile_mod.patch_tilelang_use_nvcc()
    rec = load_mod("recurrent", "flash_qla/ops/gated_delta_rule/recurrent_fused.py")

    dev = "cuda"
    q, k, v, a, b, A_log, dt_bias, h0 = make_inputs(torch, dev)
    idx = torch.arange(N, dtype=torch.int32, device=dev)
    parents = make_parents(torch, parents_kind, dev)
    inter_dtype = torch.bfloat16 if inter_bf16 else torch.float32

    kern = rec.tilelang_flashqla_gdn_regular_bv32_warp(
        H=H, HV=HV, DK=DK, DV=DV, tokens_per_seq=T, scale=SCALE,
        softplus_beta=SP_BETA, softplus_threshold=SP_THRESH,
        q_dtype=torch.bfloat16, k_dtype=torch.bfloat16, v_dtype=torch.bfloat16,
        a_dtype=torch.bfloat16, b_dtype=torch.bfloat16,
        a_log_dtype=torch.float32, dt_bias_dtype=torch.bfloat16,
        state_dtype=torch.float32, indices_dtype=torch.int32,
        o_dtype=torch.bfloat16, accum_dtype="float32",
        use_qk_l2norm_in_kernel=True,
        disable_state_update=True,
        cache_intermediate_states=True,
        has_tree_attention=has_tree,
        block_DV=BLOCK_DV,
        intermediate_dtype=inter_dtype,
    )
    total = N * T
    o = torch.zeros(1, total, HV, DV, device=dev, dtype=torch.bfloat16)
    inter = torch.zeros(N, CACHE, HV, DV, DK, device=dev, dtype=inter_dtype)
    h0_before = h0.clone()
    kern(A_log, a, dt_bias, q, k, v, b, h0, idx, inter, idx, parents, o)
    torch.cuda.synchronize()
    assert torch.equal(h0, h0_before), "DISABLE_STATE_UPDATE must leave h0 untouched"
    torch.save({"o": o.cpu(), "inter": inter.cpu()}, out_dir / f"{variant}.pt")
    print(f"saved {variant}: o_norm={o.float().norm().item():.6f}")


def softplus_ref(torch, x, beta, thresh):
    bx = beta * x
    return torch.where(bx <= thresh, torch.log1p(torch.exp(bx)) / beta, x)


def cpu_ref(torch, parents_kind, inter_store_dtype):
    """float32 reference. The register chain state stays UNROUNDED; only the
    buffered snapshots (and a parent-hop reload) round through
    inter_store_dtype — exactly the kernel's semantics."""
    q, k, v, a, b, A_log, dt_bias, h0 = make_inputs(torch, "cpu")
    par = make_parents(torch, parents_kind, "cpu")
    qf = q.float()[0].view(N, T, H, DK)
    kf = k.float()[0].view(N, T, H, DK)
    vf = v.float()[0].view(N, T, HV, DV)
    af = a.float()[0].view(N, T, HV)
    bf = b.float()[0].view(N, T, HV)
    dtb = dt_bias.float()
    o_ref = torch.zeros(N, T, HV, DV)
    inter_ref = torch.zeros(N, CACHE, HV, DV, DK)
    for n in range(N):
        h = h0[n].clone()  # [HV, DV, DK]
        for t in range(T):
            if t != 0 and par[n, t] != t - 1:
                h = inter_ref[n, par[n, t]].to(inter_store_dtype).float().clone()
            g = torch.exp(
                -torch.exp(A_log) * softplus_ref(torch, af[n, t] + dtb, SP_BETA, SP_THRESH)
            )
            beta = torch.sigmoid(bf[n, t])
            for hv in range(HV):
                hq = hv // (HV // H)
                qt, kt = qf[n, t, hq], kf[n, t, hq]
                qn = qt * torch.rsqrt((qt * qt).sum() + 1e-6) * SCALE
                kn = kt * torch.rsqrt((kt * kt).sum() + 1e-6)
                h[hv] *= g[hv]
                kv = h[hv] @ kn
                delta = (vf[n, t, hv] - kv) * beta[hv]
                h[hv] += torch.outer(delta, kn)
                o_ref[n, t, hv] = h[hv] @ qn
            inter_ref[n, t] = h.to(inter_store_dtype).float()
    return o_ref, inter_ref


def do_compare(out_dir):
    import torch

    d = {name: torch.load(out_dir / f"{name}.pt") for name in VARIANTS}

    def cos(x, y):
        x, y = x.double().flatten(), y.double().flatten()
        return float((x @ y) / (x.norm() * y.norm() + 1e-30))

    fails = []

    eq1 = torch.equal(d["lin_f32"]["o"], d["lin_bf16"]["o"])
    print(f"[1] linear o bitwise f32==bf16: {eq1}")
    if not eq1:
        fails.append("linear o differs across scratch dtypes")

    mism = (d["lin_f32"]["inter"].bfloat16() != d["lin_bf16"]["inter"]).sum().item()
    print(f"[2] linear inter bf16 == round(f32): mismatches={mism}")
    if mism:
        fails.append(f"linear inter rounding mismatch at {mism} elems")

    o_ref, i_ref = cpu_ref(torch, "chain", torch.float32)
    c_o = cos(d["lin_f32"]["o"].float()[0].view(N, T, HV, DV), o_ref)
    c_i = cos(d["lin_f32"]["inter"].float(), i_ref)
    print(f"[3] linear f32 vs CPU ref: o_cos={c_o:.9f} inter_cos={c_i:.9f}")
    if c_o < 0.9999 or c_i < 0.9999:
        fails.append(f"linear f32 vs CPU ref too low (o={c_o}, inter={c_i})")

    eq4a = torch.equal(d["tree_f32_chain"]["o"], d["lin_f32"]["o"])
    eq4b = torch.equal(d["tree_bf16_chain"]["o"], d["lin_bf16"]["o"])
    print(f"[4] tree-chain == linear: f32={eq4a} bf16={eq4b}")
    if not (eq4a and eq4b):
        fails.append("tree chain != linear")

    o_ref_hb, _ = cpu_ref(torch, "hop", torch.bfloat16)
    o_ref_hf, _ = cpu_ref(torch, "hop", torch.float32)
    c_hb = cos(d["tree_bf16_hop"]["o"].float()[0].view(N, T, HV, DV), o_ref_hb)
    c_hf = cos(d["tree_f32_hop"]["o"].float()[0].view(N, T, HV, DV), o_ref_hf)
    print(f"[5] tree-hop vs CPU ref: bf16_cos={c_hb:.9f} f32_cos={c_hf:.9f}")
    if c_hb < 0.9999 or c_hf < 0.9999:
        fails.append(f"tree hop vs CPU ref too low (bf16={c_hb}, f32={c_hf})")

    a01 = d["tree_f32_hop"]["o"].float()[0].view(N, T, HV, DV)[:, :2]
    b01 = d["tree_bf16_hop"]["o"].float()[0].view(N, T, HV, DV)[:, :2]
    eq5b = torch.equal(a01, b01)
    print(f"[5b] tree-hop pre-hop steps bitwise f32==bf16: {eq5b}")
    if not eq5b:
        fails.append("pre-hop steps differ across scratch dtypes")

    post = (
        d["tree_f32_hop"]["o"].float()[0].view(N, T, HV, DV)[:, 2:]
        != d["tree_bf16_hop"]["o"].float()[0].view(N, T, HV, DV)[:, 2:]
    ).any()
    print(f"[5c] post-hop rounding visible: {bool(post)}")
    if not post:
        fails.append("post-hop identical — bf16 reload path not exercised")

    if fails:
        print("\nFAILURES:")
        for f in fails:
            print(" -", f)
        sys.exit(1)
    print("\nALL CHECKS PASSED")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        do_run(sys.argv[2], Path(sys.argv[3]))
        return
    # driver mode: one subprocess per variant, then compare in-process
    with tempfile.TemporaryDirectory(prefix="flashqla_vsd_") as tmp:
        out_dir = Path(tmp)
        for variant in VARIANTS:
            subprocess.run(
                [sys.executable, __file__, "run", variant, str(out_dir)],
                check=True,
            )
        do_compare(out_dir)


if __name__ == "__main__":
    main()
