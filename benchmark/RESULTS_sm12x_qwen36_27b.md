# FlashQLA sm12x (Blackwell sm120/sm121) — honest FLA vs pure-TileLang results

Measured on **NVIDIA GB10 (sm121)**, torch 2.11 / tilelang 0.1.9 / fla 0.5.0 /
triton 3.6, via `benchmark/bench_fla_vs_tilelang.py`. The TileLang side is
**pure TileLang with zero FLA at the compute layer** (no fla/qla mixing); the
FLA side is forced to its Triton backend (`FLA_TILELANG=0`). Every case is
checked for correctness against an fp32 reference (`tests/ref_gdr.py`) so speed
is never reported without accuracy.

Shape: **Qwen3.6-27B TP1 GDN** — `num_key_heads=16, num_value_heads=48,
head_k=head_v=128, chunk_size=64`, bf16 q/k/v, fp32 gate/state.

## Chunked prefill / training (forward + backward)

| tokens | fwd TileLang | fwd FLA | fwd speedup | bwd TileLang | bwd FLA | bwd speedup |
|-------:|-------------:|--------:|------------:|-------------:|--------:|------------:|
|   2048 |    0.97 ms   | 1.87 ms |  **1.93×**  |    3.61 ms   | 5.13 ms |  **1.42×**  |
|   4096 |    2.01 ms   | 3.68 ms |  **1.84×**  |    7.19 ms   | 9.98 ms |  **1.39×**  |
|   8192 |    3.97 ms   | 7.35 ms |  **1.85×**  |   14.54 ms   |20.92 ms |  **1.44×**  |
|  16384 |    7.79 ms   |16.16 ms |  **2.07×**  |   29.13 ms   |43.77 ms |  **1.50×**  |

Accuracy (max-abs-err vs fp32): forward TileLang ≈ 1.2–1.7e-3 (≤ FLA's
1.4–1.9e-3); backward grads at bf16 precision (dq ~2e-2, dv ~3e-3, dg ~1.6e-1),
matching the reference decomposition.

## Recurrent decode (vs FLA fused_recurrent), batch sweep, T=1

| batch | TileLang | FLA | speedup |
|------:|---------:|----:|--------:|
|     1 | 0.0076 ms| 0.0338 ms | **4.4×** |
|     4 | 0.0514 ms| 0.0765 ms | **1.49×** |
|    16 | 0.4623 ms| 0.4633 ms | 1.00× |
|    64 | 1.843 ms | 1.862 ms  | 1.01× |

## Qwen3.5 / 3.6 family coverage (Hk=16, K=V=128, GB10/sm121)

The kernels are validated and beat FLA across the whole family (forward +
backward, pure TileLang, correctness vs fp32 ref confirmed for all):

| Hv | models | fwd vs FLA | bwd vs FLA |
|---:|---|---|---|
| 16 | Qwen3.5 0.8B / 2B | 1.25–1.28× | 1.14–1.16× |
| 32 | Qwen3.5 4B / 9B / 35B | 1.11–1.20× | 1.41–1.47× |
| 48 | Qwen3.5-27B / Qwen3.6-27B | 1.8–2.1× | 1.4–1.5× |

(`tests/test_bwd_sm12x.py` validates the backward for Hv ∈ {16,32,48}; the
Hv=16/0.8B-2B `dg` race is fixed. Smaller-Hv forward speedups are lower because
there is less work to amortize launch/overhead, but TileLang still wins.)

## Key engineering notes

* The **backward must be decomposed** on sm120/sm121: the fused Hopper backward
  needs ~228 KB dynamic shared memory and cannot launch on the 99 KB sm12x cap.
  The pure-TileLang sm12x backward is six stage kernels — `recompute_w`,
  `dv_local`, `dhu` (reverse dh-recurrence, V-tiled), `dqkwg`, `wy_bwd`,
  `dg` reverse-cumsum — each ≤99 KB. See
  `flash_qla/ops/gated_delta_rule/chunk/blackwell/bwd_sm12x.py`.
* The runtime `chunk_gated_delta_rule_bwd` routes **every** sm12x case to this
  pure path: dht=None (incl. packed/varlen `cu_seqlens`) and dht!=None
  (non-varlen, recurrence seeded from dht, emits dh0). No FLA, no precision
  mixing. The FLA-coupled `blackwell/fused_bwd.py` was **deleted** — the sm12x
  backward is now 100% pure TileLang. Only dht/initial-state gradients *with*
  varlen raise NotImplementedError (no pure path; FLA is not used).
* Validated end to end vs the fp32 reference (`tests/test_bwd_sm12x.py`, 12/12
  on GB10/sm121): non-varlen + varlen (partial chunks) at Hv=16/32/48, the
  dht!=None dh0 path, and the public autograd Function round-trip.
* `flash_qla/compile_sm12x_gdn.py --bwd-decomposed` emits these kernels for the
  kernel-jit manifest (WIP: the Ling-RL Rust `FlashQlaGdnChunkBwd` multi-kernel
  launcher remains — blocked on a Ling-RL build).
