#!/usr/bin/env python3
"""Compile sm12x FlashQLA TileLang GDN kernels into a kernel-jit manifest.

This is the FlashQLA-owned entrypoint for ling-rl's sm120/sm121 backend. It
exports the same manifest ABI used by kernel-jit, but it only compiles QLA
TileLang kernels. It does not import FLA or the FlashQLA Python runtime dispatch
module that can select FLA2 for backward.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path


CHUNK_SIZE = 64

EXPORT_CUMSUM_FWD = "flashqla_chunk_cumsum_fwd"
EXPORT_CUMSUM_BWD = "flashqla_chunk_cumsum_bwd"
EXPORT_KKT = "flashqla_chunk_kkt_solve"
EXPORT_FWD = "flashqla_chunk_fwd"
# Standard-layout ([B,HV,DK,DV]) forward (TRANSPOSE_STATE=0). The default
# EXPORT_FWD is TRANSPOSE_STATE=1 (state stored transposed [DV,DK]) — a
# self-consistent inference-only convention. The training path needs the
# *standard* layout so the forward, the carried recurrent state, and the
# decomposed backward (which is TRANSPOSE_STATE=0, matching FLA) all agree;
# otherwise a non-zero carried h0 is read with mismatched layouts and the
# gradients are wrong. Same params as EXPORT_FWD, only the state layout differs.
EXPORT_FWD_TS0 = "flashqla_chunk_fwd_ts0"
EXPORT_RECAPTURE = "flashqla_chunk_recapture_state"
EXPORT_BWD = "flashqla_chunk_bwd"
# Decomposed sm12x backward (each kernel fits the sm121 99 KB shared-mem cap,
# unlike the 228 KB fused monolith EXPORT_BWD which only runs on sm90/sm100).
EXPORT_BWD_RECOMPUTE_W = "flashqla_chunk_bwd_recompute_w"
EXPORT_BWD_DV_LOCAL = "flashqla_chunk_bwd_dv_local"
EXPORT_BWD_DHU = "flashqla_chunk_bwd_dhu"
EXPORT_BWD_DQKWG = "flashqla_chunk_bwd_dqkwg"
EXPORT_BWD_WY = "flashqla_chunk_bwd_wy"
EXPORT_RECURRENT_DECODE = "flashqla_recurrent_decode"
EXPORT_RECURRENT_VERIFY = "flashqla_recurrent_verify"
EXPORT_RECURRENT_VERIFY_TREE = "flashqla_recurrent_verify_tree"

_ROOT = Path(__file__).resolve().parents[1]
_LOADED_MODULES = {}


def setup_tilelang_env(output_dir: Path) -> None:
    os.environ.setdefault("TILELANG_EXECUTION_BACKEND", "nvrtc")
    os.environ.setdefault("TILELANG_CACHE_DIR", str(output_dir / "tilelang_cache"))
    os.environ.setdefault("TILELANG_PRINT_ON_COMPILATION", "1")


def patch_tilelang_compat() -> None:
    import tilelang.language as T

    if not hasattr(T, "gemm_v1"):
        T.gemm_v1 = T.gemm


def patch_tilelang_nvrtc_scalar_params() -> None:
    """Keep TileLang 0.1.9 NVRTC dynamic-shape handling off scalar params."""

    import tilelang  # noqa: F401
    from tvm import tir
    from tilelang.jit.adapter.nvrtc.adapter import NVRTCKernelAdapter

    if getattr(NVRTCKernelAdapter, "_flashqla_scalar_param_patch", False):
        return

    def _process_dynamic_symbolic(self) -> dict[tir.Var, tuple[int, int]]:
        func = self.prim_func
        dynamic_symbolic_map = {}
        self._dynamic_symbolic_name_map = {}
        for i, param in enumerate(func.params):
            if param not in func.buffer_map:
                continue
            buffer = func.buffer_map[param]
            for j, shape in enumerate(buffer.shape):
                if isinstance(shape, tir.Var) and shape not in dynamic_symbolic_map:
                    dynamic_symbolic_map[shape] = (i, j)
                    self._dynamic_symbolic_name_map[shape.name] = (i, j)
        return dynamic_symbolic_map

    NVRTCKernelAdapter._process_dynamic_symbolic = _process_dynamic_symbolic
    NVRTCKernelAdapter._flashqla_scalar_param_patch = True


# Cubin arch (e.g. "sm_121a" / "sm_120f") the nvcc compile should target. Set
# from --cubin-arch by main() so it matches kernel-jit's family-suffix policy
# (compiler.rs::arch_with_family_suffix). None => derive from the current torch
# device, applying the same 120->120f / 12x->12xa rule.
_FLASHQLA_CUBIN_ARCH: str | None = None


def _resolve_cubin_arch() -> str:
    """Resolve the nvcc cubin arch, preferring an explicit --cubin-arch.

    Mirrors kernel-jit's `arch_with_family_suffix`: a TRUE sm_120 part wants the
    arch-FAMILY suffix `120f` (forward-compatible Blackwell-consumer), while
    sm_12x (x>0, e.g. GB10 / DGX Spark sm_121) wants the arch-SPECIFIC `121a`.
    The previous unconditional `sm_{cc}a` mislabeled a real sm_120 cubin as
    `120a`, which kernel-jit (expecting `120f`) then could not load.
    """
    if _FLASHQLA_CUBIN_ARCH:
        return _FLASHQLA_CUBIN_ARCH
    import torch

    major, minor = torch.cuda.get_device_capability()
    n = major * 10 + minor
    if major == 12 and n == 120:
        return "sm_120f"
    return f"sm_{n}a"


def patch_tilelang_use_nvcc() -> None:
    """Route TileLang's generated-source .cu -> cubin compile through nvcc
    instead of NVRTC, while keeping the NVRTC *adapter* (so the launcher
    metadata extraction in export_tilelang_kernel, the cubin loader, and the
    dynamic-shape patch above are all unchanged).

    NVRTC cannot compile warp-shuffle reductions (`tl::warp_reduce_sum`), which
    silently forced the slow shared-memory recurrent kernel (general
    `tilelang_flashqla_gdn_update`, 128-thread + per-step block barrier) to be
    deployed instead of the warp-specialized kernel. On CUDA 13.3, TileLang
    0.1.9's NVRTC path also collides with CCCL's cuda::std tuple declarations in
    the chunk kernels. nvcc compiles both cases. The produced cubin loads via
    cuModuleLoadData identically.

    Set FLASHQLA_COMPILE_BACKEND=nvrtc to skip the patch for debugging; sm12x
    production should keep the nvcc path.
    """
    if os.environ.get("FLASHQLA_COMPILE_BACKEND", "nvcc").lower() != "nvcc":
        return

    from tilelang.contrib import nvcc as _nvcc
    from tilelang.jit.adapter.nvrtc import libgen as _libgen

    if getattr(_libgen, "_flashqla_nvcc_patch", False):
        return

    # Preflight: nvcc + a host C++ compiler must be reachable from this process's
    # PATH (the JIT child inherits the ling-llm service env). Fail with a clear,
    # actionable message instead of a deep nvcc/Python traceback when a stripped
    # systemd PATH drops /usr/bin or build-essential is missing.
    if shutil.which("nvcc") is None:
        raise RuntimeError(
            "FlashQLA recurrent warp compile needs nvcc on PATH (set PATH to include "
            "/usr/local/cuda/bin, or CUDA_HOME). Set FLASHQLA_COMPILE_BACKEND=nvrtc only "
            "to debug (the warp kernel will then fail to compile)."
        )
    # Pin the host compiler explicitly. tilelang's nvcc.compile_cuda passes no
    # -ccbin, so nvcc otherwise locates g++/gcc purely via the (service) PATH —
    # a stripped systemd `Environment=PATH=...` that drops /usr/bin makes the
    # host step fail with "gcc: No such file or directory". Honor
    # FLASHQLA_HOST_CXX, else the first c++/g++ on PATH.
    host_cxx = os.environ.get("FLASHQLA_HOST_CXX") or shutil.which("c++") or shutil.which("g++")
    if host_cxx is None:
        raise RuntimeError(
            "FlashQLA recurrent warp compile needs a host C++ compiler (g++/c++) on PATH "
            "or FLASHQLA_HOST_CXX set — install build-essential and ensure /usr/bin is on the "
            "service PATH."
        )

    def _nvcc_compile_cuda(code, target_format="ptx", arch=None, options=None, verbose=False):
        if arch is None:
            arch = _resolve_cubin_arch()
        # Use TileLang's own nvcc option builder (correct -I for tl templates /
        # cutlass / cuda include + -std=c++17). The NVRTC-oriented `options`
        # libgen passes point -I directly at cccl/cuda/std, which breaks CUDA 13
        # host-std resolution under a device --cubin compile ("global scope has
        # no memcpy"); default_compile_options uses the parent CUDA include so
        # cccl sets up its own host-std shim. relaxed-constexpr + fast-math match
        # the kernel-jit wrapper compile.
        extra = ["--expt-relaxed-constexpr", "--use_fast_math"]
        if host_cxx:
            extra += ["-ccbin", host_cxx]
        opts = _nvcc.default_compile_options(extra)
        return _nvcc.compile_cuda(
            code, target_format=target_format, arch=arch, options=opts, verbose=verbose
        )

    _libgen.compile_cuda = _nvcc_compile_cuda
    _libgen._flashqla_nvcc_patch = True
    print(
        f"FlashQLA: routing TileLang cubin compile through nvcc "
        f"(warp kernels; arch={_resolve_cubin_arch()}, ccbin={host_cxx})"
    )


def require_sm12x() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("FlashQLA sm12x compile requires a CUDA device")
    major, minor = torch.cuda.get_device_capability()
    if major != 12:
        raise RuntimeError(
            f"flash_qla.compile_sm12x_gdn requires sm12x; current device is sm{major}{minor}"
        )
    return f"sm{major}{minor}"


def load_source_module(key: str, relative_path: str):
    cached = _LOADED_MODULES.get(key)
    if cached is not None:
        return cached
    path = _ROOT / relative_path
    if not path.is_file():
        raise RuntimeError(f"missing FlashQLA source module: {path}")
    module_name = f"flash_qla._compile_sm12x_gdn_{key}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FlashQLA source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _LOADED_MODULES[key] = module
    return module


def compile_cumsum(num_value_heads: int, reverse: bool):
    import torch

    mod = load_source_module("cumsum", "flash_qla/ops/utils/cumsum.py")
    return mod.tilelang_chunk_local_cumsum(
        num_value_heads,
        CHUNK_SIZE,
        accum_dtype="float32",
        g_dtype=torch.float32,
        seqlen_dtype="int32",
        is_varlen=False,
        reverse=reverse,
    )


def compile_kkt(num_key_heads: int, num_value_heads: int, head_k_dim: int):
    import torch

    mod = load_source_module(
        "blackwell_kkt",
        "flash_qla/ops/gated_delta_rule/chunk/blackwell/kkt_solve.py",
    )
    return mod.tilelang_kkt_solve(
        num_value_heads,
        num_key_heads,
        head_k_dim,
        CHUNK_SIZE,
        accum_dtype="float32",
        qkva_dtype=torch.bfloat16,
        b_dtype=torch.bfloat16,
        seqlen_dtype="int32",
        is_varlen=False,
        use_set_max_nreg=False,
    )


def fwd_block_dv(num_value_heads: int) -> tuple[int, int]:
    if num_value_heads <= 8:
        return 8, 2
    if num_value_heads <= 32:
        return 16, 1
    return 64, 1


def compile_fwd(
    num_key_heads: int,
    num_value_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    output_h: bool,
    output_v_new: bool,
    output_o: bool = True,
    transpose_state_layout: bool = True,
):
    import torch

    mod = load_source_module(
        "blackwell_fwd",
        "flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py",
    )
    block_dv, num_stages = fwd_block_dv(num_value_heads)
    return mod.tilelang_fused_chunk_gdr_fwd(
        num_value_heads,
        num_key_heads,
        head_k_dim,
        head_v_dim,
        CHUNK_SIZE,
        1.0 / head_k_dim**0.5,
        qkva_dtype=torch.bfloat16,
        g_dtype=torch.float32,
        b_dtype=torch.bfloat16,
        h0_dtype=torch.float32,
        ht_dtype=torch.float32,
        h_dtype=torch.bfloat16,
        o_dtype=torch.bfloat16,
        seqlen_dtype="int32",
        accum_dtype="float32",
        use_initial_state=True,
        store_final_state=True,
        store_h=output_h,
        store_o=output_o,
        store_v_new=output_v_new,
        is_varlen=False,
        is_cp=False,
        block_DV=block_dv,
        num_stages=num_stages,
        use_set_max_nreg=False,
        transpose_state_layout=transpose_state_layout,
    )


def compile_bwd(
    num_key_heads: int,
    num_value_heads: int,
    head_k_dim: int,
    head_v_dim: int,
):
    import torch

    mod = load_source_module(
        "qla_fused_bwd",
        "flash_qla/ops/gated_delta_rule/chunk/hopper/fused_bwd.py",
    )
    return mod.tilelang_fused_chunk_gdr_bwd(
        num_value_heads,
        num_key_heads,
        head_k_dim,
        head_v_dim,
        CHUNK_SIZE,
        1.0 / head_k_dim**0.5,
        qkva_dtype=torch.bfloat16,
        g_dtype=torch.float32,
        b_dtype=torch.bfloat16,
        h_dtype=torch.bfloat16,
        o_dtype=torch.bfloat16,
        seqlen_dtype="int32",
        accum_dtype="float32",
        is_varlen=False,
        use_dht=False,
    )


_BWD_SM12X = "flash_qla/ops/gated_delta_rule/chunk/blackwell/bwd_sm12x.py"


def _bwd_mod():
    return load_source_module("bwd_sm12x", _BWD_SM12X)


def compile_bwd_recompute_w(num_key_heads, num_value_heads, head_k_dim):
    import torch

    return _bwd_mod().tilelang_recompute_w_fwd_sm12x(
        num_key_heads, num_value_heads, head_k_dim, CHUNK_SIZE,
        accum_dtype="float32", qkva_dtype=torch.bfloat16, g_dtype=torch.float32,
        b_dtype=torch.bfloat16,
    )


def compile_bwd_dv_local(num_key_heads, num_value_heads, head_k_dim, head_v_dim):
    import torch

    return _bwd_mod().tilelang_dv_local_sm12x(
        num_key_heads, num_value_heads, head_k_dim, head_v_dim, CHUNK_SIZE,
        1.0 / head_k_dim**0.5, qkva_dtype=torch.bfloat16, g_dtype=torch.float32,
        accum_dtype="float32",
    )


def compile_bwd_dhu(num_key_heads, num_value_heads, head_k_dim, head_v_dim, block_dv=32):
    import torch

    # use_dht=True so the kernel also emits dh0 (the gradient w.r.t. the initial
    # state); seeded from a (zero) dht buffer for the production no-final-state
    # case, this is exactly the reference's d_initial_state. Costs ~0 (one extra
    # dh0 write in the kernel that already runs) and makes the decomposed bwd
    # fully match FLA incl. dh0.
    return _bwd_mod().tilelang_dhu_sm12x(
        num_key_heads, num_value_heads, head_k_dim, head_v_dim, CHUNK_SIZE,
        1.0 / head_k_dim**0.5, qkva_dtype=torch.bfloat16, g_dtype=torch.float32,
        accum_dtype="float32", block_V=block_dv, use_dht=True,
    )


def compile_bwd_dqkwg(num_key_heads, num_value_heads, head_k_dim, head_v_dim, block_dv=64):
    import torch

    return _bwd_mod().tilelang_dqkwg_gqa_sm12x(
        num_key_heads, num_value_heads, head_k_dim, head_v_dim, CHUNK_SIZE,
        1.0 / head_k_dim**0.5, qkva_dtype=torch.bfloat16, g_dtype=torch.float32,
        accum_dtype="float32", block_V=block_dv,
    )


def compile_bwd_wy(num_key_heads, num_value_heads, head_k_dim, head_v_dim):
    import torch

    return _bwd_mod().tilelang_wy_bwd_sm12x(
        num_key_heads, num_value_heads, head_k_dim, head_v_dim, CHUNK_SIZE,
        qkva_dtype=torch.bfloat16, g_dtype=torch.float32, accum_dtype="float32",
    )


def compile_recurrent_update(
    num_key_heads: int,
    num_value_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    tokens_per_seq: int,
    disable_state_update: bool,
    cache_intermediate_states: bool,
    has_tree_attention: bool,
    block_dv: int,
    intermediate_bf16: bool = False,
):
    import torch

    # The warp kernel hard-unrolls exactly 4 lanes (tx, +32, +64, +96 = 128 K
    # values per warp); it does NOT derive the lane count from head_k_dim. Calling
    # the factory directly bypasses the dispatcher's K==128 guard
    # (recurrent_fused.py), so assert here — otherwise K<128 reads OOB and K>128
    # silently truncates the recurrent state.
    if head_k_dim != 128 or head_v_dim != 128:
        raise ValueError(
            "FlashQLA warp recurrent kernel requires head_k_dim==head_v_dim==128 "
            f"(got K={head_k_dim}, V={head_v_dim})"
        )

    mod = load_source_module(
        "recurrent_fused",
        "flash_qla/ops/gated_delta_rule/recurrent_fused.py",
    )
    # Warp-specialized recurrent kernel (32-thread, compile-time-unrolled
    # `tokens_per_seq`, warp-shuffle reductions, no shared-mem tree reduction and
    # no per-step block barrier). ~40x faster than the general
    # `tilelang_flashqla_gdn_update` at the fixed DFlash shapes (decode T=1,
    # verify T=verify_q_len). tokens_per_seq is baked in, so decode and verify
    # get distinct cubins. block_dv=4 keeps the launch grid identical to the
    # general kernel (num_value_tiles=ceil(DV/4)); only the in-cubin thread count
    # (128 -> 32) and reduction strategy change. Requires the nvcc compile path
    # (patch_tilelang_use_nvcc) — NVRTC rejects the warp-shuffle reductions.
    return mod.tilelang_flashqla_gdn_regular_bv32_warp(
        num_key_heads,
        num_value_heads,
        head_k_dim,
        head_v_dim,
        tokens_per_seq,
        1.0 / head_k_dim**0.5,
        1.0,
        20.0,
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
        v_dtype=torch.bfloat16,
        a_dtype=torch.bfloat16,
        b_dtype=torch.bfloat16,
        a_log_dtype=torch.float32,
        dt_bias_dtype=torch.bfloat16,
        state_dtype=torch.float32,
        indices_dtype=torch.int32,
        o_dtype=torch.bfloat16,
        accum_dtype="float32",
        use_qk_l2norm_in_kernel=True,
        disable_state_update=disable_state_update,
        cache_intermediate_states=cache_intermediate_states,
        has_tree_attention=has_tree_attention,
        block_DV=block_dv,
        # bf16 halves the DFlash verify intermediate-state scratch VRAM. Only
        # the per-step scratch store (and tree parent-hop reload) narrows to
        # bf16 — the recurrence accumulates fp32 in registers and `h0_source`
        # (the persistent recurrent pool) stays f32. Decode never caches
        # intermediates, so this is verify/verify_tree-only.
        intermediate_dtype=torch.bfloat16 if intermediate_bf16 else None,
    )


def split_top_level_csv(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def parse_python_launch(host_func: str) -> dict[str, dict]:
    launches: dict[str, dict] = {}
    pattern = re.compile(
        r"config\.gridDimX = (?P<grid_x>.*?)\n"
        r"\s*config\.gridDimY = (?P<grid_y>.*?)\n"
        r"\s*config\.gridDimZ = (?P<grid_z>.*?)\n"
        r"\s*config\.blockDimX = (?P<block_x>.*?)\n"
        r"\s*config\.blockDimY = (?P<block_y>.*?)\n"
        r"\s*config\.blockDimZ = (?P<block_z>.*?)\n"
        r"\s*config\.sharedMemBytes = (?P<shared>.*?)\n"
        r".*?"
        r"\s*arg_values = (?P<arg_values>.*?)\n"
        r"\s*arg_types = (?P<arg_types>.*?)\n"
        r"\s*res = cuLaunchKernelEx\(config, kernels\[\"(?P<kernel>[^\"]+)\"\]",
        re.DOTALL,
    )
    for match in pattern.finditer(host_func):
        kernel = match.group("kernel")
        launches[kernel] = {
            "grid": [
                match.group("grid_x").strip(),
                match.group("grid_y").strip(),
                match.group("grid_z").strip(),
            ],
            "block": [
                match.group("block_x").strip(),
                match.group("block_y").strip(),
                match.group("block_z").strip(),
            ],
            "shared": match.group("shared").strip(),
            "arg_values": split_top_level_csv(match.group("arg_values")),
            "arg_types": split_top_level_csv(match.group("arg_types")),
        }
    return launches


def arg_name(expr: str) -> str:
    expr = expr.strip()
    if expr.endswith(".data_ptr()"):
        return expr[: -len(".data_ptr()")]
    cast = re.fullmatch(r"ctypes\.c_[A-Za-z0-9_]+\((.*)\)", expr)
    if cast:
        return arg_name(cast.group(1).strip())
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr):
        raise RuntimeError(f"unsupported TileLang launch argument expression: {expr!r}")
    return expr


def abi_type(ctypes_expr: str) -> str:
    mapping = {
        "ctypes.c_void_p": "u64*",
        "ctypes.c_float": "f32",
        "ctypes.c_double": "f64",
        "ctypes.c_int": "i32",
        "ctypes.c_uint": "u32",
        "ctypes.c_int32": "i32",
        "ctypes.c_uint32": "u32",
        "ctypes.c_int64": "i64",
        "ctypes.c_uint64": "u64",
        "ctypes.c_bool": "bool",
    }
    ctypes_expr = ctypes_expr.strip()
    if ctypes_expr not in mapping:
        raise RuntimeError(f"unsupported TileLang launch argument type: {ctypes_expr!r}")
    return mapping[ctypes_expr]


def as_int(expr: str, label: str) -> int:
    text = expr.strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise RuntimeError(f"expected integer {label}, got {expr!r}")
    return int(text)


def export_tilelang_kernel(
    output_dir: Path,
    export_name: str,
    kernel,
    *,
    config_kwargs: dict[str, int],
):
    adapter = kernel.adapter
    lib_generator = getattr(adapter, "lib_generator", None)
    if lib_generator is None or not getattr(lib_generator, "libpath", None):
        raise RuntimeError(
            f"{export_name}: TileLang did not use the NVRTC backend; "
            "set TILELANG_EXECUTION_BACKEND=nvrtc"
        )

    function_names = list(getattr(adapter, "function_names", []))
    host_func = getattr(lib_generator, "host_func", None)
    if not host_func:
        raise RuntimeError(f"{export_name}: TileLang did not expose generated host launcher")
    launches = parse_python_launch(host_func)
    if len(function_names) != 1:
        raise RuntimeError(f"{export_name}: expected one TileLang kernel, got {function_names}")
    kernel_name = function_names[0]
    if kernel_name not in launches:
        raise RuntimeError(
            f"{export_name}: generated launcher has no metadata for {kernel_name}; "
            f"available={sorted(launches)}"
        )
    launch = launches[kernel_name]

    block = [as_int(v, f"{export_name}.block") for v in launch["block"]]
    if block[1] != 1 or block[2] != 1 or block[0] % 32 != 0:
        raise RuntimeError(f"{export_name}: unsupported block dimensions {block}")
    shared = as_int(launch["shared"], f"{export_name}.shared")
    arg_values = launch["arg_values"]
    arg_types = launch["arg_types"]
    if len(arg_values) != len(arg_types):
        raise RuntimeError(
            f"{export_name}: launch argument count mismatch values={arg_values} types={arg_types}"
        )

    dst_cubin = output_dir / f"{export_name}.cubin"
    shutil.copyfile(lib_generator.libpath, dst_cubin)
    if getattr(lib_generator, "srcpath", None):
        shutil.copyfile(lib_generator.srcpath, output_dir / f"{export_name}.cu")
    (output_dir / f"{export_name}.launch.py").write_text(host_func)

    param_names = [arg_name(v) for v in arg_values]
    abi_param_types = [abi_type(t) for t in arg_types]
    entry = {
        "export_name": export_name,
        "kernel_name": kernel_name,
        "cubin": dst_cubin.name,
        "metadata": "",
        "num_warps": block[0] // 32,
        "num_ctas": 1,
        "shared": shared,
        "num_stages": 0,
        "num_params": len(param_names),
        "abi_num_params": len(param_names),
        "hidden_params": 0,
        "param_names": param_names,
        "abi_param_types": abi_param_types,
        "semantic_param_types": abi_param_types,
        "config_kwargs": config_kwargs,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
        "profile_scratch_size": 0,
        "profile_scratch_align": 1,
        "launch_cooperative_grid": False,
        "launch_pdl": False,
    }
    print(
        f"  exported {export_name} (kernel={kernel_name}): "
        f"threads={block[0]} shared={shared} args={param_names}"
    )
    return entry


def write_manifest(output_dir: Path, manifest_name: str, entries) -> Path:
    manifest = {
        "abi_version": 1,
        "module_name": manifest_name,
        "kernels": entries,
    }
    path = output_dir / f"{manifest_name}.manifest.json"
    with path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"manifest: {path}")
    return path


def compile_chunk_entries(args: argparse.Namespace, base_cfg: dict[str, int]):
    print("Compiling sm12x FlashQLA chunk forward kernels...")
    block_dv, _ = fwd_block_dv(args.num_value_heads)
    return [
        export_tilelang_kernel(
            args.output_dir,
            EXPORT_CUMSUM_FWD,
            compile_cumsum(args.num_value_heads, reverse=False),
            config_kwargs={**base_cfg, "H": args.num_value_heads, "REVERSE": 0},
        ),
        export_tilelang_kernel(
            args.output_dir,
            EXPORT_KKT,
            compile_kkt(args.num_key_heads, args.num_value_heads, args.head_k_dim),
            config_kwargs=base_cfg,
        ),
        export_tilelang_kernel(
            args.output_dir,
            EXPORT_FWD,
            compile_fwd(
                args.num_key_heads,
                args.num_value_heads,
                args.head_k_dim,
                args.head_v_dim,
                output_h=False,
                output_v_new=False,
            ),
            config_kwargs={**base_cfg, "BV": block_dv, "TRANSPOSE_STATE": 1},
        ),
        # Standard-layout forward for the training path (TRANSPOSE_STATE=0).
        # Identical to EXPORT_FWD except it reads/writes the recurrent state in
        # the standard [B,HV,DK,DV] layout, matching the decomposed backward and
        # FLA so non-zero carried initial states are handled consistently.
        export_tilelang_kernel(
            args.output_dir,
            EXPORT_FWD_TS0,
            compile_fwd(
                args.num_key_heads,
                args.num_value_heads,
                args.head_k_dim,
                args.head_v_dim,
                output_h=False,
                output_v_new=False,
                transpose_state_layout=False,
            ),
            config_kwargs={**base_cfg, "BV": block_dv, "TRANSPOSE_STATE": 0},
        ),
        export_tilelang_kernel(
            args.output_dir,
            EXPORT_RECAPTURE,
            compile_fwd(
                args.num_key_heads,
                args.num_value_heads,
                args.head_k_dim,
                args.head_v_dim,
                output_h=False,
                output_v_new=False,
                output_o=False,
            ),
            config_kwargs={
                **base_cfg,
                "BV": block_dv,
                "TRANSPOSE_STATE": 1,
                "STORE_O": 0,
            },
        ),
    ]


DHU_BLOCK_DV = 32
DQKWG_BLOCK_DV = 64


def compile_chunk_bwd_decomposed_entries(args: argparse.Namespace, base_cfg: dict[str, int]):
    """Decomposed sm12x backward manifest (all kernels <=99KB shared, launchable
    on sm120/sm121 — unlike the 228KB monolith EXPORT_BWD). This is the production
    sm12x backward; the monolith is kept only for the legacy sm90/sm100 ABI.

    All 9 kernels export cleanly under NVRTC (TMA lowering disabled + the dh
    state written as a 4D [B,num_chunks*HV,DK,DV] view, deriving the chunk dim
    from num_tokens — no standalone num_chunks scalar). Max shared ~81KB (dqkwg).
    The Ling-RL Rust FlashQlaGdnChunkBwd launches these 9 stages in sequence.
    """
    nk, nv, dk, dv = (
        args.num_key_heads, args.num_value_heads, args.head_k_dim, args.head_v_dim
    )
    block_dv, _ = fwd_block_dv(nv)
    return [
        export_tilelang_kernel(args.output_dir, EXPORT_CUMSUM_FWD,
            compile_cumsum(nv, reverse=False),
            config_kwargs={**base_cfg, "H": nv, "REVERSE": 0}),
        export_tilelang_kernel(args.output_dir, EXPORT_KKT,
            compile_kkt(nk, nv, dk), config_kwargs=base_cfg),
        # transpose_state_layout=False: the decomposed bwd reads the initial
        # state h0 as [B,HV,DK,DV] (standard), matching what the Rust launcher
        # passes and what the Python qla_fwd default produces (validated in
        # tests/test_bwd_sm12x.py). transpose=True would read h0 as [DV,DK] and
        # corrupt h for a non-zero/non-symmetric initial state (constant init is
        # transpose-invariant, which is why only random-init cases diverged).
        export_tilelang_kernel(args.output_dir, EXPORT_FWD,
            compile_fwd(nk, nv, dk, dv, output_h=True, output_v_new=True,
                        transpose_state_layout=False),
            config_kwargs={**base_cfg, "BV": block_dv, "STORE_H": 1,
                           "STORE_V_NEW": 1, "TRANSPOSE_STATE": 0}),
        export_tilelang_kernel(args.output_dir, EXPORT_BWD_RECOMPUTE_W,
            compile_bwd_recompute_w(nk, nv, dk), config_kwargs={**base_cfg}),
        export_tilelang_kernel(args.output_dir, EXPORT_BWD_DV_LOCAL,
            compile_bwd_dv_local(nk, nv, dk, dv), config_kwargs={**base_cfg}),
        export_tilelang_kernel(args.output_dir, EXPORT_BWD_DHU,
            compile_bwd_dhu(nk, nv, dk, dv, block_dv=DHU_BLOCK_DV),
            config_kwargs={**base_cfg, "BV": DHU_BLOCK_DV}),
        export_tilelang_kernel(args.output_dir, EXPORT_BWD_DQKWG,
            compile_bwd_dqkwg(nk, nv, dk, dv, block_dv=DQKWG_BLOCK_DV),
            config_kwargs={**base_cfg, "BV": DQKWG_BLOCK_DV}),
        export_tilelang_kernel(args.output_dir, EXPORT_BWD_WY,
            compile_bwd_wy(nk, nv, dk, dv), config_kwargs={**base_cfg}),
        export_tilelang_kernel(args.output_dir, EXPORT_CUMSUM_BWD,
            compile_cumsum(nv, reverse=True),
            config_kwargs={**base_cfg, "H": nv, "REVERSE": 1}),
    ]


def compile_chunk_bwd_entries(args: argparse.Namespace, base_cfg: dict[str, int]):
    print("Compiling sm12x FlashQLA chunk backward kernels...")
    if getattr(args, "bwd_decomposed", False):
        return compile_chunk_bwd_decomposed_entries(args, base_cfg)
    block_dv, _ = fwd_block_dv(args.num_value_heads)
    return [
        export_tilelang_kernel(
            args.output_dir,
            EXPORT_CUMSUM_FWD,
            compile_cumsum(args.num_value_heads, reverse=False),
            config_kwargs={**base_cfg, "H": args.num_value_heads, "REVERSE": 0},
        ),
        export_tilelang_kernel(
            args.output_dir,
            EXPORT_KKT,
            compile_kkt(args.num_key_heads, args.num_value_heads, args.head_k_dim),
            config_kwargs=base_cfg,
        ),
        export_tilelang_kernel(
            args.output_dir,
            EXPORT_FWD,
            compile_fwd(
                args.num_key_heads,
                args.num_value_heads,
                args.head_k_dim,
                args.head_v_dim,
                output_h=True,
                output_v_new=False,
            ),
            config_kwargs={
                **base_cfg,
                "BV": block_dv,
                "STORE_H": 1,
                "STORE_V_NEW": 0,
                "TRANSPOSE_STATE": 1,
            },
        ),
        export_tilelang_kernel(
            args.output_dir,
            EXPORT_CUMSUM_BWD,
            compile_cumsum(args.num_value_heads, reverse=True),
            config_kwargs={**base_cfg, "H": args.num_value_heads, "REVERSE": 1},
        ),
        export_tilelang_kernel(
            args.output_dir,
            EXPORT_BWD,
            compile_bwd(
                args.num_key_heads,
                args.num_value_heads,
                args.head_k_dim,
                args.head_v_dim,
            ),
            config_kwargs=base_cfg,
        ),
    ]


# (export, tokens_per_seq, disable_state_update, cache_intermediate, has_tree,
#  block_dv, intermediate_bf16)
# decode is single-token; verify/verify_tree process the fixed DFlash verify_q_len
# rows. tokens_per_seq is baked into the warp cubin, so decode (always 1) and
# verify (verify_q_len) MUST live in separate modules with separate identities —
# a verify_tree compiled at tokens_per_seq=1 dead-code-eliminates retrieve_parent_token
# and would fail the Rust ABI param check that aborts the whole module load.
# intermediate_bf16 (--verify-scratch-dtype bf16) applies to the verify exports
# only: decode never declares the intermediate buffer, so it is pinned False.
def _recurrent_decode_export(_args: argparse.Namespace):
    return (EXPORT_RECURRENT_DECODE, 1, False, False, False, 4, False)


def _recurrent_verify_exports(args: argparse.Namespace):
    if args.verify_q_len < 2:
        # tokens_per_seq=1 prunes the tree branch (retrieve_parent_token); the
        # verify module is only meaningful for a real DFlash round (verify_q_len
        # = tree_budget + 1 >= 2). Decode is the single-token path.
        raise ValueError(
            f"--verify-q-len must be >= 2 for the recurrent verify module "
            f"(got {args.verify_q_len}); decode uses the recurrent_decode module"
        )
    intermediate_bf16 = args.verify_scratch_dtype == "bf16"
    return [
        (EXPORT_RECURRENT_VERIFY, args.verify_q_len, True, True, False, 4, intermediate_bf16),
        (EXPORT_RECURRENT_VERIFY_TREE, args.verify_q_len, True, True, True, 4, intermediate_bf16),
    ]


def _emit_recurrent_exports(args: argparse.Namespace, recurrent_exports):
    # The warp recurrent kernel needs the nvcc compile path (NVRTC rejects the
    # warp-shuffle reductions). Install it here — NOT globally — so chunk /
    # chunk_bwd keep the proven NVRTC path (and their non-fast-math cumsum).
    patch_tilelang_use_nvcc()
    recurrent_base_cfg = {
        "H": args.num_key_heads,
        "HV": args.num_value_heads,
        "K": args.head_k_dim,
        "V": args.head_v_dim,
    }
    entries = []
    for (
        export_name,
        tokens_per_seq,
        disable_state_update,
        cache_intermediate,
        has_tree,
        block_dv,
        intermediate_bf16,
    ) in recurrent_exports:
        entries.append(
            export_tilelang_kernel(
                args.output_dir,
                export_name,
                compile_recurrent_update(
                    args.num_key_heads,
                    args.num_value_heads,
                    args.head_k_dim,
                    args.head_v_dim,
                    tokens_per_seq=tokens_per_seq,
                    disable_state_update=disable_state_update,
                    cache_intermediate_states=cache_intermediate,
                    has_tree_attention=has_tree,
                    block_dv=block_dv,
                    intermediate_bf16=intermediate_bf16,
                ),
                config_kwargs={
                    **recurrent_base_cfg,
                    "BV": block_dv,
                    "TOKENS_PER_SEQ": tokens_per_seq,
                    "DISABLE_STATE_UPDATE": int(disable_state_update),
                    "CACHE_INTERMEDIATE": int(cache_intermediate),
                    "TREE": int(has_tree),
                    # Baked scratch dtype of intermediate_states_buffer (0=f32,
                    # 1=bf16). The Rust loader validates this against the
                    # process's LING_DFLASH_VERIFY_SCRATCH_BF16 so a stale/
                    # mismatched cubin cannot silently read the scratch at the
                    # wrong element width.
                    "INTERMEDIATE_BF16": int(intermediate_bf16),
                },
            )
        )
    return entries


def compile_recurrent_decode_entries(args: argparse.Namespace):
    print("Compiling sm12x FlashQLA recurrent GDN decode kernel...")
    return _emit_recurrent_exports(args, [_recurrent_decode_export(args)])


def compile_recurrent_verify_entries(args: argparse.Namespace):
    print("Compiling sm12x FlashQLA recurrent GDN verify kernels...")
    return _emit_recurrent_exports(args, _recurrent_verify_exports(args))


def compile_recurrent_entries(args: argparse.Namespace):
    # Combined decode + verify + verify_tree (for `--mode recurrent`/`all`). The
    # production path uses the split recurrent_decode / recurrent_verify modes so
    # decode (vq-independent, prewarmed at boot) and verify (vq-keyed, loaded at
    # drafter-attach) are distinct modules.
    print("Compiling sm12x FlashQLA recurrent GDN kernels...")
    return _emit_recurrent_exports(
        args, [_recurrent_decode_export(args), *_recurrent_verify_exports(args)]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile sm12x FlashQLA TileLang GDN kernels for kernel-jit"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    # Root of the FlashQLA checkout (the dir containing `flash_qla/`). Lets an
    # out-of-tree caller (e.g. the Ling-RL kernel-jit runner, which embeds this
    # tree) point at it explicitly; defaults to this file's own repo root. When
    # given, it overrides _ROOT and is prepended to sys.path so the stage kernel
    # modules' `from flash_qla...` imports resolve regardless of cwd.
    parser.add_argument("--flashqla-path", type=Path, default=None)
    parser.add_argument("--head-k-dim", type=int, default=128)
    parser.add_argument("--head-v-dim", type=int, default=128)
    parser.add_argument("--num-key-heads", type=int, default=16)
    parser.add_argument("--num-value-heads", type=int, default=16)
    # DFlash verify row count (tree_budget + 1). The warp-specialized recurrent
    # kernel bakes tokens_per_seq at compile time, so verify/verify_tree cubins
    # are specialized to this; decode is always tokens_per_seq=1.
    parser.add_argument("--verify-q-len", type=int, default=17)
    # Storage dtype of the verify/verify_tree per-step intermediate-state
    # scratch (`intermediate_states_buffer`). bf16 halves the DFlash verify
    # scratch VRAM; the recurrence still accumulates fp32 in registers and the
    # persistent recurrent pool (`h0_source`) stays f32. Baked into the cubin
    # (and the INTERMEDIATE_BF16 manifest config), so the Ling-RL runner keys
    # its kernel cache on it. Decode is unaffected (no intermediate buffer).
    parser.add_argument(
        "--verify-scratch-dtype", choices=["f32", "bf16"], default="f32"
    )
    # nvcc cubin arch (e.g. "sm_121a" / "sm_120f"). The Ling-RL kernel-jit runner
    # passes its own family-suffix-resolved arch so the recurrent warp cubin is
    # named/targeted exactly like the wrapper-compiled kernels. Absent => derive
    # from the current torch device (applying the 120->120f / 12x->12xa rule).
    parser.add_argument("--cubin-arch", type=str, default=None)
    parser.add_argument(
        "--mode",
        choices=[
            "chunk",
            "chunk_bwd",
            "recurrent",
            "recurrent_decode",
            "recurrent_verify",
            "all",
        ],
        default="chunk",
    )
    parser.add_argument("--manifest-name")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--bwd-decomposed",
        action="store_true",
        help="emit the decomposed sm12x backward kernels (sm121-launchable, <=99KB "
        "shared) instead of the 228KB fused monolith; this is the production "
        "sm12x backward (see compile_chunk_bwd_decomposed_entries)",
    )
    return parser.parse_args()


def main() -> None:
    global _ROOT, _FLASHQLA_CUBIN_ARCH
    args = parse_args()
    if args.flashqla_path is not None:
        _ROOT = args.flashqla_path.expanduser().resolve()
    # Ensure `from flash_qla...` resolves for the load_source_module'd stage
    # kernels regardless of how/where this script is launched.
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_tilelang_env(args.output_dir)
    _FLASHQLA_CUBIN_ARCH = args.cubin_arch

    import torch

    torch.cuda.set_device(args.device)
    arch = require_sm12x()
    patch_tilelang_compat()
    patch_tilelang_nvrtc_scalar_params()
    patch_tilelang_use_nvcc()
    print(f"Compiling FlashQLA GDN kernels for {arch} from {_ROOT}")

    base_cfg = {
        "H": args.num_value_heads,
        "Hg": args.num_key_heads,
        "HV": args.num_value_heads,
        "K": args.head_k_dim,
        "V": args.head_v_dim,
        "BT": CHUNK_SIZE,
    }

    entries = []
    if args.mode in {"chunk", "all"}:
        entries.extend(compile_chunk_entries(args, base_cfg))
    if args.mode in {"chunk_bwd", "all"}:
        entries.extend(compile_chunk_bwd_entries(args, base_cfg))
    if args.mode == "recurrent_decode":
        entries.extend(compile_recurrent_decode_entries(args))
    if args.mode == "recurrent_verify":
        entries.extend(compile_recurrent_verify_entries(args))
    if args.mode in {"recurrent", "all"}:
        entries.extend(compile_recurrent_entries(args))

    manifest_name = args.manifest_name or {
        "chunk": "flashqla_gdn_chunk",
        "chunk_bwd": "flashqla_gdn_chunk_bwd",
        "recurrent": "flashqla_gdn_recurrent",
        "recurrent_decode": "flashqla_gdn_recurrent_decode",
        "recurrent_verify": "flashqla_gdn_recurrent_verify",
        "all": "flashqla_gdn",
    }[args.mode]
    write_manifest(args.output_dir, manifest_name, entries)


if __name__ == "__main__":
    main()
