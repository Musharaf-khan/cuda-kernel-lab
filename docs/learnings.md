# What each optimization taught me

All numbers are from a Colab Tesla T4 (CUDA 13.0) in the run stored in [`benchmarks/`](../benchmarks); a few are from an earlier session and say so. Byte counts come from Nsight Compute DRAM counters ([`benchmarks/ncu/`](../benchmarks/ncu)). Each section follows the same loop: predict the bottleneck, change one thing, measure, check the counter that proves why.

---

## 01 · Vector add: the memory-bound baseline

`c[i] = a[i] + b[i]` does 1 FLOP per 12 bytes (two 4-byte reads, one write). The math units are idle; the only possible limit is DRAM bandwidth.

- Measured **263.8 GB/s = 82% of the 320 GB/s spec**. Well-written streaming kernels land at 80–90%; the rest is DRAM refresh, page switching and protocol overhead. You cannot make this kernel faster by computing smarter, only by moving fewer bytes.
- Habits set here and used everywhere after: bounds guard `if (i < n)`, block size a multiple of 32 (256 = 8 full warps), warm up before timing, time with CUDA events, never include `cudaMemcpy` in kernel time.

## 02 · Reduction: contention, then synchronization, then registers

| Version | GB/s | Why |
|---|---:|---|
| v1 `atomicAdd` per element | 1.9 | 16.7M atomics to **one address** serialize in the L2: thousands of cores become a queue. |
| v2 shared-memory tree | 107.5 | One atomic per block (65,536 instead of 16.7M). Still slow: each thread loads only 1 element, then waits at 8 `__syncthreads()` while half the threads go idle at every level. |
| v3 grid-stride + warp shuffle | 247.8 | ~320 blocks; each thread sums ~200 elements **in a register**, then warps combine with `__shfl_down_sync` (register-to-register, no shared memory, no barrier). Only ~320 atomics. |

**Lesson:** do as much as possible per thread in registers, reduce within warps, and touch shared memory, barriers and atomics as rarely as possible. **133x** from the same hardware. Note that all three versions read the same ~82–94 MB from DRAM: v1 isn't slow because of memory traffic, it's slow because of serialization.

## 03 · Transpose: coalescing and bank conflicts, and why a counter isn't a stopwatch

| Version | This run | Earlier session |
|---|---:|---:|
| copy (ceiling) | 235.1 GB/s | 239.0 GB/s |
| naive | 84.8 GB/s | 60.1 GB/s |
| tiled `[32][32]` | 183.4 GB/s | 105.7 GB/s |
| tiled `[32][33]` | 193.9 GB/s | 190.3 GB/s |

- **Naive:** reads are coalesced but consecutive threads write addresses 4096 floats apart, so each warp's 32 writes become 32 transactions. `ncu`: DRAM only 39% busy.
- **Tiled `[32][32]`:** stage a 32×32 tile in shared memory so both global read and write are row-wise. But reading a tile *column* puts all 32 threads in the same bank: `bank = (address / 4) mod 32`, and a row is exactly 32 floats wide.
- `ncu` reports **16,252,928** load bank conflicts, identical in both sessions. Derivation: 4096² / 32 = 524,288 warp-wide shared loads, each a 32-way conflict = 31 extra replays → 524,288 × 31 = 16,252,928.
- **Padded `[32][33]`:** element `[r][c]` is now in bank `(33r + c) mod 32 = (r + c) mod 32`, so a column hits 32 different banks. **0 conflicts.** It reaches 82% of a plain copy; the rest is the extra shared-memory traffic and the barrier.
- **The surprise:** the conflict count is the same in both sessions, but its cost isn't. In the earlier session padding was worth 1.8x; in this run (newer CUDA 13 toolchain) only 1.06x. A bank conflict costs shared-memory replays, and whether that shows up in wall-clock time depends on whether there is other work (here, outstanding DRAM requests) to hide it behind. The likely explanation is different instruction scheduling between compiler versions, though I haven't confirmed it by diffing the SASS. **Lesson: a profiler counter tells you a problem exists; only timing tells you what it costs.**

## 04 · SGEMM: every fix exposes the next bottleneck

| Version | GFLOPS (4096³, median of 3) | % cuBLAS (same run) | DRAM read (2048³, ncu) | Limiter |
|---|---:|---:|---:|---|
| v1 naive, `threadIdx.x → row` | 61.7 | 1.7% | 2,215 MB | a warp reads A down a column: uncoalesced |
| v2 naive, `threadIdx.x → col` | 505.8 | 15.1% | 1,654 MB | global memory (L2 already catching much of the reuse) |
| v3 32×32 shared tiles | 776.0 | 23.2% | 1,584 MB | **shared-memory bandwidth**: 1 FMA per 2 shared loads |
| v4 2D register tiling (128×128 block, 8×8 per thread) | 2,841.4 | 85.1% | 201 MB | FP32 issue: 8 + 8 register loads feed 64 FMAs |
| v5 v4 + `float4` + transposed A tile | 3,101.6 | **91.6%** (90.0 / 92.9 / 91.6) | 185 MB | FP32 issue, fewer load instructions |
| cuBLAS | 3,339.8 | 100% | 275 MB | |

- **v1 → v2 (8.2x) is two swapped index lines.** Same math, same work.
- **v2 → v3 was only 1.5x even though tiling should cut global traffic 32x.** The counters show why: DRAM reads barely moved (1,654 → 1,584 MB), because in v2 the L2 cache was *already* serving most of the reuse. The bottleneck had moved from DRAM to shared memory; counting loads per FMA in the inner loop predicted it.
- **Global traffic math (whiteboard this):** naive does 2n loads per output → 2n³ total. A TILE×TILE shared tile reuses each loaded element TILE times → 2n³ / TILE. For TILE = 32: a 32x cut in *requests* (the L2 decides how many reach DRAM).
- **v4's trick:** registers are the only memory faster than shared memory. An 8×8 per-thread tile raises intensity at the shared-memory level from 0.5 to 4 FMA per load, and the larger 128×128 block tile cut DRAM reads 8x (1,584 → 201 MB). 3.7x faster.
- **v5 (+9% over v4):** storing the A tile transposed makes each thread's 8 values of A contiguous, so it can use `ld.shared.v4` (verified in the PTX: 32 vector shared loads, 0 local memory, 125 registers, no spills), and global loads/stores use `float4`.
- **Interesting: cuBLAS reads *more* DRAM (275 MB) yet is faster.** On the T4 it picked `volta_sgemm_128x64_nn`, a smaller tile, and hides the extra traffic with a better-scheduled inner loop. Fewer bytes isn't the goal once you're compute-bound.
- **Small sizes flip the ranking.** In the n = 256 correctness run, v4/v5 (538 GFLOPS) were *slower* than v3 (845): a 128×128 block tile gives only 4 blocks for 40 SMs, so 90% of the GPU idles. Tile size is a trade-off against parallelism.
- **Measurement honesty:** cuBLAS ranged 3,165–3,575 GFLOPS across three runs as the GPU heated up and hit its 70 W cap, while the v5/cuBLAS ratio stayed within 90–93%. Report same-run ratios and medians. Under Nsight (single launch at 2048³, GPU not yet throttled) the ratio is lower, ~68%; likely causes are that a 2048³ problem gives v5 only 256 blocks (a ragged last wave on 40 SMs) and that a sustained run throttles the more power-hungry cuBLAS kernel harder. I haven't isolated which dominates.
- **Next steps not taken here:** double-buffered shared memory (load tile k+1 while computing tile k), warp tiling, and on newer GPUs Tensor Cores (WMMA/MMA), `cp.async`, TMA/WGMMA on Hopper.

## 05 · Softmax: stability, then passes over memory

Softmax does a handful of FLOPs per element: it is memory-bound, so the game is how many bytes cross the DRAM bus.

| Version | Time | GB/s | DRAM read (ncu) |
|---|---:|---:|---:|
| v1 thread-per-row, 3 passes | 4.579 ms | 29.3 | 575 MB |
| v2 block-per-row, 3 passes | 0.746 ms | 180.0 | 93 MB |
| v3 online softmax, 2 passes | 0.751 ms | 178.8 | 124 MB |
| v4 register-cached, 1 pass | 0.573 ms | **234.1** | 76 MB |
| `torch.softmax` | 0.715 ms | 187.8 | |

The minimum read is 67 MB (the 4096×4096 input once).

- **Stability:** `exp(x)` overflows FP32 above x ≈ 88.7. On logits in [90, 110], the version without max-subtraction produced **262,144 / 262,144 NaN/Inf outputs**; the stable version produced 0. `softmax(x) = softmax(x − max)` mathematically, so subtracting the max is free correctness.
- **v1 thread-per-row:** neighbouring threads walk different rows → uncoalesced (8.6x the minimum DRAM reads), and only 16 blocks for 40 SMs.
- **v2 and v3 are a lesson in checking assumptions.** On paper v2 reads each row 3 times and online softmax (v3) 2 times. The counters say v2 actually read only 1.4x the minimum: a 16 KB row is still in L1/L2 when the next pass comes back for it. So online softmax gave **no speedup** here. It matters when the row is too large to stay cached, and it matters enormously in attention, where the "row" is computed on the fly and never stored at all.
- **Online softmax itself:** keep a running max `m` and normalizer `l`; when a new value `v > m` arrives, rescale `l ← l·exp(m − v) + 1`. Two partial states merge with `m = max(m1, m2)`, `l = l1·exp(m1 − m) + l2·exp(m2 − m)`. The merge is associative, so it reduces in a warp-shuffle tree like any sum.
- **v4 register-cached:** a 4096-float row is 16 floats per thread for 256 threads, so the whole row fits in registers (`float4` loads). One read, one write: the minimum possible traffic. **1.25x faster than `torch.softmax`, 8.0x faster than v1.**

## 06 · LayerNorm: stable statistics and fusion

| Version | Time | GB/s | DRAM read (ncu) |
|---|---:|---:|---:|
| v1 two-pass (mean, variance, write) | 0.870 ms | 154.3 | 95 MB |
| v2 Welford stats + write | 0.816 ms | 164.5 | 132 MB |
| v3 register-cached | 0.568 ms | **236.1** | 77 MB |
| `F.layer_norm` | 0.806 ms | 166.5 | |

- **Why not one pass with Σx and Σx²?** `var = E[x²] − E[x]²` subtracts two large, nearly equal numbers (catastrophic cancellation) when the mean is large relative to the spread. The benchmark deliberately uses inputs with mean ≈ 3.
- **v2 Welford:** a running `(count, mean, M2)` updated per element, merged across threads with Chan et al.'s formula `M2 = M2a + M2b + δ²·na·nb/n`. Only 1.07x over v1, for the same caching reason as softmax, plus a division per element.
- **v3 register-cached:** row in registers → exact two-pass on-chip, 1 read + 1 write. **1.42x faster than `F.layer_norm`.**
- **Fusion:** in a transformer block `h = x + residual; y = LayerNorm(h)`. Unfused: the add kernel writes `h` to DRAM and LayerNorm reads it back (5 array-passes). Fused: read `x` and `residual`, write `h` and `y` (4 passes). Measured DRAM traffic **375 MB → 304 MB** (the saved 71 MB is one 67 MB array plus noise), and time **1.370 → 1.127 ms (1.22x)**: almost exactly the 5 : 4 ratio theory predicts. **1.44x faster than PyTorch eager**, which runs them as separate kernels.

## 07 · Attention: FlashAttention's idea in one kernel

`O = softmax(QKᵀ/√d)·V` with batch 2 × 8 heads, d = 64, FP32.

| N | Unfused | Fused | Fused vs unfused | PyTorch SDPA | Fused vs SDPA |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 1.92 ms | 2.65 ms | 0.73x | 1.66 ms | 63% |
| 2,048 | 9.07 ms | 9.26 ms | 0.98x | 6.36 ms | 69% |
| 4,096 | 43.0 ms | 28.8 ms | **1.49x** | 25.6 ms | 89% |
| 8,192 | 196.6 ms | 112.2 ms | **1.75x** | 100.6 ms | 90% |
| 16,384 | out of memory (17.2 GB) | 453.6 ms | runs | 411.9 ms | 91% |

- **Unfused (the textbook way):** cuBLAS writes S = QKᵀ (16·N² floats) to DRAM, a softmax kernel reads and rewrites it, cuBLAS reads it again for P·V. At N = 2048 that moved **1,536 MB** through DRAM (ncu). At N = 16384 the score matrix needs 17.2 GB and does not fit on a 15 GB T4.
- **Fused:** each block owns 64 query rows (one per thread, `q` and the output accumulator `o` in registers). K and V stream through shared memory in tiles of 32 rows. Scores for the tile stay in registers; online softmax rescales `o` and `l` by `exp2(m_old − m_new)` whenever the running max grows, then adds `p·V`. The N×N matrix never exists. At N = 2048 it moved **59 MB: 26x less DRAM traffic.**
- **Why the crossover?** The unfused cost grows with the N² bytes of S (its throughput falls from 2.2 to 1.4 TFLOP/s as N grows); the fused kernel is compute-bound and climbs to a steady ~2.4 TFLOP/s from N = 4096 on. At small N, cuBLAS's GEMMs are simply more efficient than my one-thread-per-row FP32 loop, so the score matrix is cheap enough not to matter.
- **Versus PyTorch:** FP32 SDPA on a T4 runs a CUTLASS memory-efficient attention kernel; mine reaches ~90% of it at N ≥ 4096. The gap is the work split: one thread per query row means 254 registers per thread and low occupancy. FlashAttention-2 splits each block's rows across warps and uses Tensor Cores in FP16/BF16 with `cp.async` pipelining.
- Small tricks that mattered: fold `log2(e)/√d` into `q` once so every exponential is a single `exp2f`; 4 independent accumulators in the dot product for instruction-level parallelism; every thread reads the same K/V row at the same time, which is a shared-memory **broadcast**, so no bank conflicts.
