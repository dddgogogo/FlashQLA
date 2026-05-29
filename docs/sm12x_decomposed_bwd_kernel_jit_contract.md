# sm12x decomposed GDN backward — kernel-jit deployment contract

The fused Hopper backward needs ~228 KB shared memory and cannot launch on
sm120/sm121 (99 KB cap). `flash_qla/compile_sm12x_gdn.py --bwd-decomposed`
therefore emits a **decomposed** backward manifest (`flashqla_gdn_chunk_bwd`)
whose every kernel fits ≤99 KB and is pure TileLang (no FLA). This documents
the contract so the Ling-RL Rust launcher (`crates/kernel-jit/src/flashqla_gdn.rs`
`FlashQlaGdnChunkBwd`) can orchestrate it. All kernels are bf16 q/k/v/a/b/w/h,
fp32 g/state; chunk_size (BT) = 64; C = 64; num_chunks = ceil(T/64).

## Kernels (manifest order), shared mem, threads, params, grid

| export | smem(KB) | warps | grid | notes |
|---|---|---|---|---|
| `flashqla_chunk_cumsum_fwd` | 24 | 4 | (existing cumsum grid) | g_raw→g_cumsum (REVERSE=0) |
| `flashqla_chunk_kkt_solve` | 42 | 8 | (existing) | k,g,b→a (A, [B,T,HV,64]) |
| `flashqla_chunk_fwd` | 88 | 16 | (existing fwd grid) | STORE_H=1, STORE_V_NEW=1, TRANSPOSE_STATE=1 → h,v_new |
| `flashqla_chunk_bwd_recompute_w` | 40 | 4 | (ceil(T/64), B·Hk), 128thr | →w [B,T,HV,DK] |
| `flashqla_chunk_bwd_dv_local` | 56 | 4 | (ceil(T/64), B·Hk), 128thr | →dv [B,T,HV,DV] |
| `flashqla_chunk_bwd_dhu` | 80 | 4 | (ceil(DV/32), B·HV), 128thr | →dh [B,nchunks·HV,DK,DV]; updates dv in place |
| `flashqla_chunk_bwd_dqkwg` | 93 | 8 | (ceil(T/64), B·Hk), 256thr | →dq, dk1, dw, dg1 (dq/dk1 already key-head-reduced) |
| `flashqla_chunk_bwd_wy` | 85 | 8 | (ceil(T/64), B·Hk), 256thr | →dk(final), dv(final), db, dg |
| `flashqla_chunk_cumsum_bwd` | 24 | 4 | (existing) | dg→dg (REVERSE=1) |

Param order per kernel (exact `param_names` from the manifest — the Rust
`build_params_from_names` binds by name, so order is informational):
- recompute_w: `a, beta, g, k, w` + `batch_size, num_tokens`
- dv_local:   `dv, g, k, q` + `batch_size, num_tokens`
- dhu:        `dh, dv, g, k, q, w` + `batch_size, num_tokens`
- dqkwg:      `dg, dh, dk, dq, dv, dw, g, h, k, q, v` + `batch_size, num_tokens`
- wy:         `Amat, beta, db, dg, dg1, dk, dk1, du, dv, dw, g, k, v` + `batch_size, num_tokens`

`batch_size`/`num_tokens` are i32 scalars; all others are u64* pointers.

## Launch sequence (one prefill backward)

```
cumsum_fwd(g_raw=g)                                   -> g_cumsum         [B,T,HV] f32
kkt_solve(k, g_cumsum, b)                             -> A                [B,T,HV,64] bf16
fwd(q,k,v,A,g_cumsum,b,h0; STORE_H,STORE_V_NEW)       -> h, v_new (o/ht throwaway)
recompute_w(A, b, g_cumsum, k)                        -> w                [B,T,HV,DK] bf16
dv_local(q, k, g_cumsum, do)                          -> dv               [B,T,HV,DV] bf16
dhu(q, k, w, g_cumsum, do, dv)                        -> dh [B,nchunks*HV,DK,DV]; dv updated
dqkwg(q, k, v_new, w, g_cumsum, h, dv, do, dh)        -> dq, dk1, dw, dg1
wy(k, v, b, A, g_cumsum, dw, du=dv, dk1, dg1)         -> dk, dv, db, dg
cumsum_bwd(g_raw=dg)                                  -> dg (reverse)
```
Final grads: `dq` (from dqkwg), `dk`/`dv`/`db` (from wy), `dg` (from cumsum_bwd).
GQA: dqkwg and wy reduce dq/dk to key-head internally (Hg=num_key_heads).
Production case: `initial_state=None`, `dht=None` (use_dht=False); `dh0` not produced.

## Intermediate buffers the Rust launcher must allocate
`g_cumsum[B,T,HV]f32, A[B,T,HV,64]bf16, h[B,nchunks,HV,DK,DV]bf16,
v_new[B,T,HV,DV]bf16, w[B,T,HV,DK]bf16, dv[B,T,HV,DV]bf16,
dh[B,nchunks*HV,DK,DV]bf16, dq/dk[B,T,Hk,DK]bf16, dw[B,T,HV,DK]bf16,
dg/db/dg1[B,T,HV]f32`. (`dh` is 4D in the manifest but is the same memory as the
5D [B,nchunks,HV,DK,DV] that dqkwg reads.)

## CRITICAL: dynamic shared-memory opt-in
fwd(88), dv_local(56), dhu(80), dqkwg(93), wy(85) exceed the 48 KB static
default. The launcher MUST call
`cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared)`
before `cuLaunchKernel` (the existing kernel-jit C wrapper already does this when
`shared>0`). TileLang's own NVRTC adapter does NOT opt in, which is why direct
`tilelang` NVRTC-backend launch crashes for these kernels — irrelevant to the
Rust path.

## Status / validation
All 9 kernels compile under NVRTC and fit ≤99 KB (verified on GB10/sm121). The
kernels are numerically validated vs the fp32 reference (`tests/test_bwd_sm12x.py`,
12/12 on GB10/sm121; `benchmark/bench_fla_vs_tilelang.py`): Qwen3.6-27B backward
is ~1.4× faster than FLA. The Hv=16 (2B/0.8B) `dg` race is fixed.

The pure backward is the ONLY sm12x backward now — the FLA-coupled
`blackwell/fused_bwd.py` was deleted; `chunk_gated_delta_rule_bwd` routes every
sm12x case (dht=None incl. packed/varlen, and dht!=None non-varlen) to the pure
TileLang path. There is no FLA on the sm12x compute path.

Capabilities now in the kernels (beyond this contract's original non-varlen
prefill scope — relevant when the Rust launcher is extended):
- varlen (cu_seqlens): every stage has a varlen prim_func (per-sequence state
  reset; dh packed via prepare_chunk_offsets; dqkwg/wy use chunk_indices). The
  packed-training launcher would pass cu_seqlens + chunk_indices/offsets.
- dht / dh0 (non-varlen): `dhu` has a use_dht mode that seeds the recurrence
  from dht ([B,HV,DK,DV]) and emits dh0 ([B,HV,DK,DV]); dht!=None together with
  varlen is unsupported and raises.
This contract's launch sequence/param tables describe the NON-varlen, dht=None
prefill path, which remains the production ling-rl case.
