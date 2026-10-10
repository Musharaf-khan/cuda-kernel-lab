# What each optimization taught me

Numbers are from a Colab Tesla T4 unless marked *pending*. Each section follows the same loop: predict the bottleneck, change one thing, measure, check the counter that proves why.

---

## 01 · Vector add: the memory-bound baseline

`c[i] = a[i] + b[i]` does 1 FLOP per 12 bytes (two 4-byte reads, one write). The math units are idle; the only possible limit is DRAM bandwidth.

- Measured **262.6 GB/s = 82% of the 320 GB/s spec**. Well-written streaming kernels land at 80–90%; the rest is DRAM refresh, page switching and protocol overhead. You cannot make this kernel faster by computing smarter, only by moving fewer bytes.
- Habits set here and used everywhere after: bounds guard `if (i < n)`, block size a multiple of 32 (256 = 8 full warps), warm up before timing, time with CUDA events, never include `cudaMemcpy` in kernel time.

## 02 · Reduction: contention, then synchronization, then registers

| Version | GB/s | Why |
|---|---:|---|
| v1 `atomicAdd` per element | 1.9 | 16.7M atomics to **one address** serialize in the L2: thousands of cores become a queue. |
| v2 shared-memory tree | 107.7 | One atomic per block (65,536 instead of 16.7M). Still slow: each thread loads only 1 element, then waits at 8 `__syncthreads()` while half the threads go idle at every level. |
| v3 grid-stride + warp shuffle | 248.0 | ~320 blocks; each thread sums ~200 elements **in a register**, then warps combine with `__shfl_down_sync` (register-to-register, no shared memory, no barrier). Only ~320 atomics. |

**Lesson:** do as much as possible per thread in registers, reduce within warps, and touch shared memory, barriers and atomics as rarely as possible. 128x from the same hardware.

## 03 · Transpose: coalescing and bank conflicts, proven with counters

- **Naive (60.1 GB/s):** reads are coalesced but consecutive threads write addresses 4096 floats apart, so each warp's 32 writes become 32 transactions. `ncu`: DRAM only 39% busy.
- **Tiled `[32][32]` (105.7 GB/s):** stage a 32×32 tile in shared memory so both global read and write are row-wise. But reading a tile *column* puts all 32 threads in the same bank: `bank = (address / 4) mod 32`, and a row is exactly 32 floats wide.
- `ncu` reported **16,252,928** load bank conflicts. Derivation: 4096² / 32 = 524,288 warp-wide shared loads, each a 32-way conflict = 31 extra replays → 524,288 × 31 = 16,252,928. Exactly the counter.
- **Padded `[32][33]` (190.3 GB/s, 0 conflicts):** element `[r][c]` is now in bank `(33r + c) mod 32 = (r + c) mod 32`, so a column hits 32 different banks. One extra float per row, 1.8x.
- It reaches 80% of a plain copy (239 GB/s). The remaining gap is the extra shared-memory traffic and the barrier.

## 04 · SGEMM: every fix exposes the next bottleneck

| Version | GFLOPS (4096³) | % cuBLAS | Limiter |
|---|---:|---:|---|
| v1 naive, `threadIdx.x → row` | 61.8 | 1.7% | a warp reads A down a column: uncoalesced |
| v2 naive, `threadIdx.x → col` | 543.7 | 14.5% | global memory (helped by L1/L2 and A-broadcast) |
| v3 32×32 shared tiles | 838.0 | 22.4% | **shared-memory bandwidth**: `acc += As[ty][k] * Bs[k][tx]` is 1 FMA per 2 shared loads |
| v4 2D register tiling (128×128 block, 8×8 per thread) | 3,098.7 | **82.7%** | compute: per k-step, 8 + 8 values loaded into registers feed 64 FMAs (4 FMA/load) |
| v5 v4 + `float4` + transposed A tile | *pending* | | fewer load instructions: 128-bit global loads/stores and `LDS.128` from shared memory |
| cuBLAS | 3,744.9 | 100% | |

- **v1 → v2 (8.8x) is two swapped index lines.** Same math, same work.
- **v2 → v3 was only 1.5x even though global traffic fell 32x.** The bottleneck moved from DRAM to shared memory. Counting loads per FMA in the inner loop predicted it.
- **Global traffic math (whiteboard this):** naive does 2n loads per output → 2n³ total. A TILE×TILE shared tile reuses each loaded element TILE times → 2n³ / TILE. For TILE = 32: a 32x cut.
- **v4's trick:** registers are the only memory faster than shared memory. An 8×8 per-thread tile raises arithmetic intensity at the shared-memory level from 0.5 to 4 FMA per load.
- **v5:** storing the A tile transposed makes each thread's 8 values of A contiguous, so the compiler emits 2× `ld.shared.v4` instead of 8 scalar loads (checked in the PTX: 32 vector shared loads, 0 local memory, 125 registers, no spills).
- **Measurement honesty:** cuBLAS swung from 5,274 to 3,597 GFLOPS across runs as the GPU heated up and hit its 70 W cap, but the v4/cuBLAS ratio stayed at 82.5–83.5%. Report ratios from the same run, medians across runs.
- **Next steps not taken here:** double-buffered shared memory (load tile k+1 while computing tile k), warp tiling, and on newer GPUs Tensor Cores (WMMA/MMA), `cp.async`, TMA/WGMMA on Hopper. That is where cuBLAS's remaining lead comes from.

## 05 · Softmax: stability, then passes over memory

Softmax does a handful of FLOPs per element: it is memory-bound, so the game is how many times each row is read from DRAM.

- **Stability:** `exp(x)` overflows FP32 above x ≈ 88.7. The benchmark feeds logits in [90, 110] to a version with no max-subtraction and counts NaN/Inf outputs, then shows the stable version produces none. `softmax(x) = softmax(x − max)` mathematically, so subtracting the max is free correctness.
- **v1 thread-per-row:** neighbouring threads walk different rows → uncoalesced, and only 16 blocks for 40 SMs.
- **v2 block-per-row, 3 passes:** coalesced, but reads the row 3 times (max, sum, write).
- **v3 online softmax, 2 passes:** keep a running max `m` and normalizer `l`; when a new value `v > m` arrives, rescale `l ← l·exp(m − v) + 1`. Two partial states merge with `m = max(m1, m2)`, `l = l1·exp(m1 − m) + l2·exp(m2 − m)`. Because the merge is associative, it reduces in a warp-shuffle tree like any sum. This is the exact trick FlashAttention needs.
- **v4 register-cached, 1 pass:** a 4096-float row is 16 floats per thread for 256 threads, so the whole row fits in registers (`float4` loads). Read once, write once: the minimum possible traffic.
- Results: *pending* (vs `torch.softmax`).

## 06 · LayerNorm: stable statistics and fusion

- **v1 two-pass:** mean pass, variance pass, write pass → 3 reads.
- **Why not one pass with Σx and Σx²?** `var = E[x²] − E[x]²` subtracts two large, nearly equal numbers (catastrophic cancellation) when the mean is large relative to the spread. The benchmark deliberately uses inputs with mean ≈ 3.
- **v2 Welford:** a running `(count, mean, M2)` updated per element, merged across threads with Chan et al.'s formula `M2 = M2a + M2b + δ²·na·nb/n`. One stable statistics pass + one write pass → 2 reads.
- **v3 register-cached:** row in registers → exact two-pass on-chip, 1 read + 1 write.
- **Fusion:** in a transformer block `h = x + residual; y = LayerNorm(h)`. Unfused: the add kernel writes `h` to DRAM and LayerNorm reads it back (5 array-passes). Fused: read `x` and `residual`, write `h` and `y` (4 passes) and `h` never makes a round trip. For memory-bound ops, fewer passes = proportionally less time.
- Results: *pending* (vs `F.layer_norm` and PyTorch eager add + layer_norm).

## 07 · Attention: FlashAttention's idea in one kernel

`O = softmax(QKᵀ/√d)·V` with BH = 16 heads, d = 64.

- **Unfused (the textbook way):** cuBLAS writes S = QKᵀ (BH·N² floats) to DRAM, a softmax kernel reads and rewrites it, cuBLAS reads it again for P·V. At N = 8192 that is 4.3 GB of scores; at N = 16384 it needs 17.2 GB and **does not fit** on a 15 GB T4.
- **Fused:** each block owns 64 query rows (one per thread, `q` and the output accumulator `o` in registers). K and V stream through shared memory in tiles of 32 rows. Scores for the tile stay in registers; online softmax rescales `o` and `l` by `exp2(m_old − m_new)` whenever the running max grows, then adds `p·V`. The N×N matrix never exists: extra memory is O(N) instead of O(N²).
- Small tricks: fold `log2(e)/√d` into `q` once so every exponential is a single `exp2f`; 4 independent accumulators in the dot product for instruction-level parallelism; every thread reads the same K/V row at the same time, which is a shared-memory **broadcast**, so no bank conflicts.
- Honest scope: FP32, forward only, non-causal, one thread per query row (254 registers, no spills). Real FlashAttention-2 splits work across warps, uses Tensor Cores in FP16/BF16 and `cp.async` pipelining. PyTorch's FP32 SDPA on a T4 uses the memory-efficient (CUTLASS) kernel, which is a strong baseline.
- Results: *pending* (vs unfused cuBLAS path and `F.scaled_dot_product_attention`).
