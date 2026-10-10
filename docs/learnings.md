# What each optimization taught me

All numbers are from a Colab Tesla T4 (CUDA 13.0, PyTorch 2.11, Triton 3.6) in the run stored in [`benchmarks/`](../benchmarks); a few are from an earlier session and say so. Byte counts come from Nsight Compute DRAM counters ([`benchmarks/ncu/`](../benchmarks/ncu)), instruction counts from `cuobjdump -sass` ([`benchmarks/sass/`](../benchmarks/sass)). Each section follows the same loop: predict the bottleneck, change one thing, measure, check the counter that proves why.

---

## 01 · Vector add: the memory-bound baseline

`c[i] = a[i] + b[i]` does 1 FLOP per 12 bytes (two 4-byte reads, one write). The math units are idle; the only possible limit is DRAM bandwidth.

- Measured **263.5 GB/s = 82% of the 320 GB/s spec**. Well-written streaming kernels land at 80–90%; the rest is DRAM refresh, page switching and protocol overhead. You cannot make this kernel faster by computing smarter, only by moving fewer bytes.
- Habits set here and used everywhere after: bounds guard `if (i < n)`, block size a multiple of 32 (256 = 8 full warps), warm up before timing, time with CUDA events, never include `cudaMemcpy` in kernel time.

## 02 · Reduction: contention, then synchronization, then registers

| Version | GB/s | Why |
|---|---:|---|
| v1 `atomicAdd` per element | 1.9 | 16.7M atomics to **one address** serialize in the L2: thousands of cores become a queue. |
| v2 shared-memory tree | 107.6 | One atomic per block (65,536 instead of 16.7M). Still slow: each thread loads only 1 element, then waits at 8 `__syncthreads()` while half the threads go idle at every level. |
| v3 grid-stride + warp shuffle | 248.3 | ~320 blocks; each thread sums ~200 elements **in a register**, then warps combine with `__shfl_down_sync` (register-to-register, no shared memory, no barrier). Only ~320 atomics. |

**Lesson:** do as much as possible per thread in registers, reduce within warps, and touch shared memory, barriers and atomics as rarely as possible. **131x** from the same hardware. All three versions read the same 81–93 MB from DRAM: v1 isn't slow because of memory traffic, it's slow because of serialization.

## 03 · Transpose: coalescing and bank conflicts, and why a counter isn't a stopwatch

| Version | This run | Earlier session |
|---|---:|---:|
| copy (ceiling) | 235.1 GB/s | 239.0 GB/s |
| naive | 88.1 GB/s | 60.1 GB/s |
| tiled `[32][32]` | 183.3 GB/s | 105.7 GB/s |
| tiled `[32][33]` | 194.8 GB/s | 190.3 GB/s |

- **Naive:** reads are coalesced but consecutive threads write addresses 4096 floats apart, so each warp's 32 writes become 32 transactions. `ncu`: DRAM only 39% busy.
- **Tiled `[32][32]`:** stage a 32×32 tile in shared memory so both global read and write are row-wise. But reading a tile *column* puts all 32 threads in the same bank: `bank = (address / 4) mod 32`, and a row is exactly 32 floats wide.
- `ncu` reports **16,252,928** load bank conflicts, identical in every session. Derivation: 4096² / 32 = 524,288 warp-wide shared loads, each a 32-way conflict = 31 extra replays → 524,288 × 31 = 16,252,928.
- **Padded `[32][33]`:** element `[r][c]` is now in bank `(33r + c) mod 32 = (r + c) mod 32`, so a column hits 32 different banks. **0 conflicts.** It reaches 83% of a plain copy; the rest is the extra shared-memory traffic and the barrier.
- **The surprise (still true in this run):** the conflict count is the same in every session, but its cost isn't. In the earlier session padding was worth 1.8x; with the CUDA 13 toolchain it is only **1.06x** (183.3 → 194.8 GB/s). A bank conflict costs shared-memory replays, and whether that shows up in wall-clock time depends on whether there is other work (outstanding DRAM requests) to hide it behind. The likely explanation is different instruction scheduling between compiler versions, though I haven't confirmed it by diffing the SASS. **Lesson: a profiler counter tells you a problem exists; only timing tells you what it costs.** The SGEMM work below hit the same lesson from the other side (B4).

## 04 · SGEMM: every fix exposes the next bottleneck

| Version | GFLOPS (4096³, median of 3) | % cuBLAS (same run, median) | DRAM read (2048³, ncu) | Limiter |
|---|---:|---:|---:|---|
| v1 naive, `threadIdx.x → row` | 61.8 | 1.5% | 2,223 MB | a warp reads A down a column: uncoalesced |
| v2 naive, `threadIdx.x → col` | 563.0 | 14.1% | 1,654 MB | global memory (L2 already catching much of the reuse) |
| v3 32×32 shared tiles | 875.2 | 21.8% | 1,567 MB | **shared-memory bandwidth**: 1 FMA per 2 shared loads |
| v4 2D register tiling (128×128 block, 8×8 per thread) | 3,215.1 | 81.8% | 201 MB | FP32 issue: 8 + 8 register loads feed 64 FMAs |
| v5 v4 + `float4` global loads/stores (transposed A tile) | 3,522.6 | 88.4% (88.7 / 87.7 / 88.4) | 182 MB | 2-way shared-load bank conflicts on the B tile (see B1) |
| **v6 autotuned** (warp tiling + double buffering + padded A tile) | **4,045.6** | **101.3%** (104.0 / 100.7 / 101.3) | 192 MB | FP32 issue (FMA pipe 92% busy) |
| cuBLAS | 3,999.5 | 100% | 277 MB | |

- **v1 → v2 (9.1x) is two swapped index lines.** Same math, same work.
- **v2 → v3 was only 1.55x even though tiling should cut global traffic 32x.** The counters show why: DRAM reads barely moved (1,654 → 1,567 MB), because in v2 the L2 cache was *already* serving most of the reuse. The bottleneck had moved from DRAM to shared memory; counting loads per FMA in the inner loop predicted it.
- **Global traffic math (whiteboard this):** naive does 2n loads per output → 2n³ total. A TILE×TILE shared tile reuses each loaded element TILE times → 2n³ / TILE. For TILE = 32: a 32x cut in *requests* (the L2 decides how many reach DRAM).
- **v4's trick:** registers are the only memory faster than shared memory. An 8×8 per-thread tile raises intensity at the shared-memory level from 0.5 to 4 FMA per load, and the larger 128×128 block tile cut DRAM reads 8x (1,567 → 201 MB). 3.7x faster.
- **v5 (+9.6% over v4), and a correction.** I originally credited v5's gain to the transposed A tile enabling vectorized `LDS.128` shared loads. **The SASS says otherwise:** ptxas already emitted `LDS.128` for v4; both hot loops contain exactly 32 `LDS.128`. What actually changed is the global side: `float4` loads/stores cut the hot loop's `LDG` from 8 to 2 and `STS` from 8 to 5 (563 vs 579 instructions per k-tile, 91% of them FFMA). Read the machine code before crediting an optimization.
- **cuBLAS reads *more* DRAM (277 MB) yet is fast.** On the T4 it picked `volta_sgemm_128x64_nn`, a smaller tile, and hides the extra traffic with a better-scheduled inner loop. Fewer bytes isn't the goal once you're compute-bound.

### Closing the last 10%: the attempt log

Goal: ≥95% of cuBLAS at 4096³, same-run methodology (cuBLAS timed before and after, median of 3 runs). Stop rule: reach 95%, or three consecutive changes that each gain <1%. To make one change at a time possible, v5 was re-expressed as one template, `sgemm_lab<BM, BN, BK, WM, WN, WNITER, TM, TN, DB, PADA, MINB>` ([`kernels/04_gemm/sgemm_lab.cuh`](../kernels/04_gemm/sgemm_lab.cuh)), and every attempt is a template instantiation run by [`gemm_lab.cu`](../kernels/04_gemm/gemm_lab.cu). Every attempt is correctness-checked against cuBLAS. Raw logs: `benchmarks/raw/04_gemm_attempts_run{1,2,3}.txt`; counters: `benchmarks/ncu/sgemm_attempts_*`.

The ncu columns come from one profiled launch each, at normal (unlocked) clocks and before the GPU heats up, so absolute durations are lower than the sustained runs; use the % column for speed claims and the counters for *why*. The first profiled kernel (A0) ran at 37.7 ms, slower than in sustained runs; I suspect the clocks were still ramping and don't use its duration for anything.

| # | Change (one at a time from A0, then cumulative) | % of cuBLAS (median of 3) | Shared-load bank conflicts | FMA pipe busy | Cycles / issued instr. | Verdict |
|---|---|---:|---:|---:|---:|---|
| A0 | v5 re-expressed in the template (baseline) | 90.1% | 268,435,456 | 66% | 10.94 | baseline |
| B1 | **warp tiling** (64×32 warp tile, 2 sub-tiles of 32×32, 8×4 per thread) | **99.4%** (98.7 / 99.4 / 100.4) | **0** | 82% | 8.75 | **win, crossed 95%** |
| B2 | register-prefetch double buffering on v5 | 88.0% | 268,435,456 | 67% | 10.58 | no gain alone |
| B3a | 128 threads (8×16 per thread) | 92.3% | 268,435,456 | 69% | 5.30 | small, noisy |
| B3b | 512 threads (4×8 per thread) | 81.5% | 0 | 56% | 12.03 | loss |
| B3c | `__launch_bounds__(256, 2)` | 88.8% | 268,435,456 | 66% | 10.94 | no change (already 2 blocks/SM) |
| B4 | pad the transposed A tile (stride 132) | 91.1% (94.5 / 89.8 / 91.1) | 268,435,456 (store conflicts 16.8M → 0) | 67% | 10.73 | noise-level |
| B5 | BK = 16 (half the barriers per FLOP) | 88.2% | 268,435,456 | 66% | 10.82 | no gain |
| C2 | B1 + double buffering | 97.6% | 0 | **93.5%** | 7.52 | better under ncu, not in sustained runs |
| C3 | C2 + padded A tile | **100.2%** | 52,720 (store: 210,752) | 92% | 7.66 | **best config → v6** |
| C4 | C3 + BK = 16 | 94.4% | 0 | 75% | 4.80 | loss (occupancy halves) |
| C4b | C4 without padding (32 KB smem) | 94.2% | 0 | 73% | 4.92 | loss |
| C5 | C4 + `__launch_bounds__(256, 2)` | 86.3% | 15 | 67% | 5.26 | loss (spills) |
| — | cuBLAS `volta_sgemm_128x64_nn` | 100% (3,808 GFLOPS in this ladder) | 118,904 | 97.6% | 6.92 | |

What each one taught:

- **B1, warp tiling: the real bottleneck was a bank conflict I hadn't seen.** In v5 each thread's 8 B values are two `LDS.128` loads at a 32-byte thread stride. A warp-wide 128-bit shared load is served in quarter-warp phases; with a 32-byte stride, 8 threads span 256 bytes, which is two passes over the 32 banks: a **2-way conflict on every B load**. ncu counted exactly **268,435,456** extra load wavefronts = 8,192 warps × 4,096 k-steps × 2 B loads × 4 extra wavefronts. Warp tiling rearranges which outputs each lane owns so that lanes read consecutive `float4`s (a 16-byte stride): conflicts went to **0**, shared-load wavefronts dropped from 671M to 403M, the FMA pipe went from 66% to 82% busy, and the kernel went from 90% to **99% of cuBLAS**. This was the single biggest change, and I'd have kept guessing without the counter.
- **B2 and C2, double buffering: it helps only after the conflicts are gone.** On sm_75 there is no `cp.async`, so "double buffering" means: load the next tile's global data into registers, compute the current tile from shared memory, then store the registers to the other shared buffer (one `__syncthreads()` per tile instead of two). On v5 (B2) nothing changed (88.0%): the kernel was limited by shared-memory replays, not by load latency. On top of warp tiling (C2), ncu shows the FMA pipe at 93.5% and the single-launch time dropping from 24.1 to 21.9 ms, but in the sustained, power-capped runs it measured 97.6% vs B1's 99.4%. My hypothesis is that once the T4 hits its 70 W cap, a kernel that keeps the FMA pipe busier just gets clocked down harder; I haven't measured clocks per kernel to prove it.
- **B3, threads per block:** 128 threads (8×16 per thread) raises FFMA per shared load from 16 to 21.3 (SASS) but needs 207 registers, so occupancy drops to 24.6%; it helped a little (92.3%). 512 threads (4×8 per thread) drops to 10.7 FFMA per shared load and 84 registers; 81.5%. The register tile size sets how much math each shared load buys.
- **B3c and C5, `__launch_bounds__`:** v5 already ran 2 blocks/SM at 123 registers, so asking for 2 changed nothing. On the BK = 16 kernel (C5), forcing 2 blocks/SM capped registers at 128, the SASS shows **23 local-memory (spill) instructions, 12 in the hot loop**, and shared memory still limited it to 1 block/SM: the worst result of the ladder (86.3%). Kept as a negative result.
- **B4, padding:** removed the A-tile **store** conflicts completely (16.8M → 0) but left the 268M **load** conflicts that mattered; 91.1% with a noisy spread (89.8–94.5). Same lesson as the transpose: a counter going to zero is not a speedup unless it was the bottleneck. In C3 the padding was worth keeping (97.6 → 100.2%), once the load conflicts were already gone.
- **B5, C4, C4b, BK = 16:** half the barriers per FLOP, but 140 registers → 1 block/SM by registers, and with 2 buffers × 16 KB the shared memory also limits to 1 block/SM even without padding (C4b, 32 KB). Occupancy 24.6%, FMA pipe 73–75%: loses 6 points.
- **Stop rule:** B1 reached ≥95% first; C2 → C3 was the last gain (+2.6 points); C4, C4b and C5 were three consecutive changes that each lost, so the search stopped there.
- **Versus cuBLAS, honestly:** in the sustained, same-run measurement v6 is at parity (101.3% median in the main benchmark, 99.3–107.6% across sizes in the autotuner). In a single unthrottled ncu launch, cuBLAS is still ~10% faster (19.8 ms vs 22.3 ms for C3 at 4096³; 6.9 vs 6.2 TFLOP/s on the roofline chart at 2048³). Its kernel runs 128 threads at 122 registers with 4 blocks/SM, keeps the FMA pipe 97.6% busy and issues 0.57 warps per scheduler per cycle vs our 0.51. I did not disassemble cuBLAS, so I can't say exactly which scheduling tricks close that last gap. "Matches cuBLAS on a power-capped T4" is the accurate claim; "beats cuBLAS" is not.

### The autotuner and the n = 256 fix

v5's 128×128 block tile is a 4096³ design. At n = 256 it launches only 4 blocks for 40 SMs, so 90% of the GPU idles: v5 ran at **540.6 GFLOPS (31.3% of cuBLAS)**, slower than the plain 32×32 tiled v3 (844.9).

[`scripts/autotune.py`](../scripts/autotune.py) sweeps 11 configurations of the same template (block tile 64×64 → 256×128, thread tile 4×8 → 16×8, 128/256/512 threads, BK 8/16) at n ∈ {256, 1024, 2048, 4096}. Each size is run 3 times (cuBLAS before and after in each run), it takes the median, discards any configuration that fails the correctness check, and writes the winners to [`configs/sgemm_autotune.json`](../configs/sgemm_autotune.json). `build/gemm` reads that file and runs the chosen configuration as **v6**.

| n | Chosen config | Autotune median | lab_v5 (= v5) in the same sweep |
|---:|---|---:|---:|
| 256 | `t64x64_4x8` (64×64 block, 128 threads, 4×8 per thread) | 1,168.6 GFLOPS (91.9%) | 398.1 GFLOPS |
| 1024 | `t128x128_8x8` (= C3) | 4,012.1 (97.1%) | 2,402.7 |
| 2048 | `t128x128_8x8` | 4,549.7 (107.6%) | 3,396.9 |
| 4096 | `t128x128_8x8` | 4,009.7 (99.3%) | 3,600.1 |

In the separate main GEMM benchmark (3 runs), **n = 256: v6 1,631.5 GFLOPS (94.1% of cuBLAS) vs v5 540.6 (31.3%): 3.0x**, and 1.9x over v3. The 64×64 tile launches 16 blocks; the 4×8 thread tile keeps registers low enough for many blocks per SM. The large configs (256×128, 128×256) were 15% of cuBLAS at n = 256 and 98% at 4096: there is no single best tile, which is why cuBLAS ships dozens of kernels and a heuristic to pick between them.

## 05 · Softmax: stability, then passes over memory

Softmax does a handful of FLOPs per element: it is memory-bound, so the game is how many bytes cross the DRAM bus.

| Version | Time | GB/s | DRAM read (ncu) |
|---|---:|---:|---:|
| v1 thread-per-row, 3 passes | 3.664 ms | 36.6 | 576 MB |
| v2 block-per-row, 3 passes | 0.697 ms | 192.5 | 93 MB |
| v3 online softmax, 2 passes | 0.760 ms | 176.6 | 124 MB |
| v4 register-cached, 1 pass | 0.584 ms | **229.7** | 77 MB |
| `torch.softmax` | 0.704 ms | 190.5 | |

The minimum read is 67 MB (the 4096×4096 input once).

- **Stability:** `exp(x)` overflows FP32 above x ≈ 88.7. On logits in [90, 110], the version without max-subtraction produced **262,144 / 262,144 NaN/Inf outputs**; the stable version produced 0. `softmax(x) = softmax(x − max)` mathematically, so subtracting the max is free correctness.
- **v1 thread-per-row:** neighbouring threads walk different rows → uncoalesced (8.6x the minimum DRAM reads), and only 16 blocks for 40 SMs.
- **v2 and v3 are a lesson in checking assumptions.** On paper v2 reads each row 3 times and online softmax (v3) 2 times. The counters say v2 actually read only 1.4x the minimum: a 16 KB row is still in L1/L2 when the next pass comes back for it. So online softmax gave **no speedup** here; in this run it was even **slower** (0.760 vs 0.697 ms) and read *more* DRAM (124 vs 93 MB). It matters when the row is too large to stay cached, and it matters enormously in attention, where the "row" is computed on the fly and never stored at all.
- **Online softmax itself:** keep a running max `m` and normalizer `l`; when a new value `v > m` arrives, rescale `l ← l·exp(m − v) + 1`. Two partial states merge with `m = max(m1, m2)`, `l = l1·exp(m1 − m) + l2·exp(m2 − m)`. The merge is associative, so it reduces in a warp-shuffle tree like any sum.
- **v4 register-cached:** a 4096-float row is 16 floats per thread for 256 threads, so the whole row fits in registers (`float4` loads). One read, one write: the minimum possible traffic. **1.21x faster than `torch.softmax`, 6.3x faster than v1.**

## 06 · LayerNorm: stable statistics and fusion

| Version | Time | GB/s | DRAM read (ncu) |
|---|---:|---:|---:|
| v1 two-pass (mean, variance, write) | 0.927 ms | 144.8 | 95 MB |
| v2 Welford stats + write | 0.870 ms | 154.3 | 132 MB |
| v3 register-cached | 0.568 ms | **236.3** | 77 MB |
| `F.layer_norm` | 0.805 ms | 166.8 | |

- **Why not one pass with Σx and Σx²?** `var = E[x²] − E[x]²` subtracts two large, nearly equal numbers (catastrophic cancellation) when the mean is large relative to the spread. The benchmark deliberately uses inputs with mean ≈ 3.
- **v2 Welford:** a running `(count, mean, M2)` updated per element, merged across threads with Chan et al.'s formula `M2 = M2a + M2b + δ²·na·nb/n`. Only 1.07x over v1, for the same caching reason as softmax, plus a division per element.
- **v3 register-cached:** row in registers → exact two-pass on-chip, 1 read + 1 write. **1.42x faster than `F.layer_norm`.**
- **Fusion:** in a transformer block `h = x + residual; y = LayerNorm(h)`. Unfused: the add kernel writes `h` to DRAM and LayerNorm reads it back (5 array-passes). Fused: read `x` and `residual`, write `h` and `y` (4 passes). Measured DRAM traffic **375 MB → 305 MB** (the saved 70 MB is one 67 MB array plus noise), and time **1.372 → 1.126 ms (1.22x)**: almost exactly the 5 : 4 ratio theory predicts. **1.44x faster than PyTorch eager**, which runs them as separate kernels.

## 07 · Attention: FlashAttention's idea in one kernel

`O = softmax(QKᵀ/√d)·V` with batch 2 × 8 heads, d = 64, FP32. SDPA times are from `scripts/torch_compare.py` on the same shapes.

| N | Unfused | Fused v1 | Fused v2 (2 thr/row, 128 rows) | v1 vs unfused | v2 vs unfused | PyTorch SDPA | v1 vs SDPA | **v2 vs SDPA** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.227 ms | 0.259 ms | 0.168 ms | 0.88x | 1.36x | — | — | — |
| 1,024 | 2.031 ms | 2.665 ms | 1.647 ms | **0.76x** | 1.23x | 1.507 ms | 57% | 91.5% |
| 2,048 | 7.584 ms | 7.950 ms | 6.179 ms | **0.95x** | 1.23x | 5.772 ms | 73% | 93.4% |
| 4,096 | 36.288 ms | 26.349 ms | 25.006 ms | 1.38x | 1.45x | 23.965 ms | 91% | **95.8%** |
| 8,192 | 169.464 ms | 101.002 ms | 100.086 ms | 1.68x | 1.69x | 94.570 ms | 94% | **94.5%** |
| 16,384 | out of memory (17.2 GB) | 417.135 ms | 406.694 ms | runs | runs | 385.086 ms | 92% | **94.7%** |

v2 numbers are one fixed configuration (2 threads per row, 128 rows per block, K/V tile 32) at every N, not the best variant per N. At N = 2,048 the 64-row variant was faster (5.829 ms, 99.0% of SDPA); all four variants are in `benchmarks/results.md`.

- **Unfused (the textbook way):** cuBLAS writes S = QKᵀ (16·N² floats) to DRAM, a softmax kernel reads and rewrites it, cuBLAS reads it again for P·V. At N = 2048 that moved **~1.5 GB** through DRAM (ncu). At N = 16384 the score matrix needs 17.2 GB and does not fit on a 15 GB T4.
- **Fused v1:** each block owns 64 query rows (one per thread, `q` and the output accumulator `o` in registers). K and V stream through shared memory in tiles of 32 rows. Scores for the tile stay in registers; online softmax rescales `o` and `l` by `exp2(m_old − m_new)` whenever the running max grows, then adds `p·V`. The N×N matrix never exists. At N = 2048 it moved **59 MB: 26x less DRAM traffic.**
- **Fused v1 still loses below ~2K tokens (0.76x at 1,024, 0.95x at 2,048).** The unfused cost grows with the N² bytes of S; the fused kernel is compute-bound. At small N, cuBLAS's GEMMs are simply more efficient than my one-thread-per-row FP32 loop, so the score matrix is cheap enough not to matter. This stays in the table: it's the honest crossover.
- **Why v1 was stuck at ~91% of SDPA:** one thread per query row holds a 64-float `q` and a 64-float `o` in registers: **254 registers per thread**, 4 blocks/SM by registers, **23% achieved occupancy** (ncu). Too few warps to hide shared-memory and FMA latency.

### Attention v2: split each row across threads

`flash_attention_split<HD, ROWS, TILE, SPLIT>` in [`kernels/07_attention/attention.cu`](../kernels/07_attention/attention.cu): each query row is owned by `SPLIT` adjacent lanes. Lane `sub` holds the interleaved `float4` chunks `c·SPLIT + sub` of `q` and `o` (so the K/V reads stay conflict-free broadcasts), computes a partial dot product, and the partial scores are combined with `__shfl_xor_sync` (1 shuffle for 2 lanes, 2 for 4). Online softmax state is identical in every lane of the row, so no other communication is needed.

ncu at N = 2,048 (one launch each):

| Variant | Threads/block | Registers | Achieved occupancy | Cycles between issues | v2 vs v1 (timed, N = 1,024 / 4,096) |
|---|---:|---:|---:|---|---|
| v1: 1 thread/row, 64 rows | 64 | 254 | 23.4% | 6.52 | baseline |
| v2: 2 threads/row, 64 rows, tile 32 | 128 | 159 | 35.3% | 8.41 (52% MIO-queue) | 1.21x / 1.04x |
| v2: 2 threads/row, 128 rows, tile 32 | 256 | 159 | 25.0% | **6.24** | **1.62x / 1.05x** |
| v2: 4 threads/row, 64 rows, tile 32 | 256 | 125 | 48.2% | 19.27 (76% MIO-queue, 14.6 cycles) | **0.62x / 0.60x** |
| v2: 4 threads/row, 64 rows, tile 64 | 256 | 148 | 25.0% | 10.64 | 0.86x / 0.59x |

- **2 threads/row is the sweet spot.** Registers drop from 254 to 159, so a block can hold more rows. With 128 rows per block, each K/V tile that is staged through shared memory feeds twice as many query rows, and warps wait the least between instructions (6.24 cycles). v2 is **1.62x faster than v1 at N = 1,024**, 1.29x at 2,048, and only 1–5% at N ≥ 4,096, where v1 was already compute-bound.
- **Occupancy wasn't the lever I expected.** The 64-row variant has the highest occupancy of the 2-thread variants (35%) but is slower than the 128-row one at 25%.
- **4 threads/row is a negative result worth knowing.** Two shuffles per score (per key, per row) go through the MIO pipe on Turing, the same queue as shared-memory loads. It has the best occupancy of all (48%), and it's the slowest: ncu shows warps waiting 14.6 cycles per instruction for that one queue. More warps can't help when every warp is waiting on the same unit.
- **Goal not met:** v2 is at **94.5–95.8% of PyTorch's FP32 SDPA for N ≥ 4,096**, not ≥100%. The gap shrank from ~9% (v1) to ~5%. SDPA on a T4 runs a CUTLASS memory-efficient attention kernel; doing better in FP32 would need the full FlashAttention-2 work split (one warp per 16 rows, with the K/V tile shared across warps and scores computed as a small register-tiled GEMM, like SGEMM v6) rather than more threads per row. Not attempted.
- Small tricks that mattered: fold `log2(e)/√d` into `q` once so every exponential is a single `exp2f`; 4 independent accumulators in the dot product for instruction-level parallelism; every thread reads the same K/V row at the same time, which is a shared-memory **broadcast**, so no bank conflicts (SASS: all shared loads are `LDS.128`, 4 FFMA per shared load in every variant).

## 08 · WMMA FP16 Tensor Core GEMM vs SIMT

`C = A·B` with FP16 inputs and FP32 accumulation through `nvcuda::wmma` (m16n16k16 fragments, which compile to Turing `HMMA.1688` instructions). Reference: `cublasGemmEx` with FP16 inputs, FP32 output and `CUBLAS_COMPUTE_32F`, i.e. the same math.

| n | v1 WMMA naive | v2 WMMA tiled | cublasGemmEx (FP16 in, FP32 acc.) | v2 % of cublasGemmEx | cublasHgemm (FP16 out) | cublasSgemm FP32 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 2.55 TFLOPS | 2.62 | 4.20 | 62.4% | 4.87 | 1.66 |
| 1024 | 3.62 | **22.97** | 33.74 | 68.1% | 34.59 | 5.84 |
| 2048 | 2.59 | **24.76** | 28.31 | 87.5% | 29.45 | 4.61 |
| 4096 | 2.09 | **15.59** | 22.66 | 68.8% | 31.15 | 4.03 |

- **Read the % column as a range, 68–88%.** cuBLAS itself drifted inside the same run as the T4 heated up: at 2048³ it measured 0.422 ms before our kernels and 0.792 ms after; at 4096³ 5.58 → 6.55 ms. The single-launch ncu profile at 2048³ gives 601 µs vs 426 µs = **71%**, which is the most trustworthy single number.
- **WMMA vs SIMT:** at 4096³ the tiled WMMA kernel does 15.6 TFLOP/s vs 4.0 for FP32 `cublasSgemm`: **3.9x** from the Tensor Cores, on the same chip. On the T4 the FP16 Tensor Core peak is 65 TFLOP/s vs 8.1 FP32, so a GEMM that keeps HMMA busy can't be matched by any SIMT kernel.
- **v1 naive (one warp per 16×16 output tile, fragments loaded straight from global memory)** is barely faster than FP32 SIMT (2–3.6 TFLOP/s). The Tensor Cores finish each `mma_sync` in a handful of cycles and then wait hundreds of cycles for the next fragment from DRAM: ncu shows 96% occupancy but only 18% SM throughput. Tensor Cores make the memory problem worse, not better.
- **v2 tiled:** a 128×128 block tile with BK = 32, 8 warps, each warp owning a 64×32 output (4×2 accumulator fragments, so each A fragment is reused 2x and each B fragment 4x from registers). A and B tiles are staged in shared memory padded by 8 halves per row (16 bytes) to stagger the rows across banks, with register-prefetch double buffering. 37.9 KB shared memory, 148 registers. **6–10x over v1.**
- **Fragment rules learned the hard way:** fragment layouts are opaque (you can't index a lane's elements to mean a fixed (row, col)); `load_matrix_sync` needs a 256-bit-aligned pointer and a leading dimension that's a multiple of 8 halves (16 bytes), which constrains the padding; and every lane of the warp must call `load/mma/store_sync` together, so no divergent branches around them.
- **Where cuBLAS is still ahead:** cuBLAS runs `turing_s1688gemm_fp16_128x128_ldg8`: the same 128×128 tile and similar occupancy (~24%), but 71% SM throughput vs our 48%. The SASS shows why: our WMMA fragment loads compile to **24 scalar `LDS.32` per k-tile, 2.7 HMMA per shared load, and no `LDSM`** (`ldmatrix`, the instruction that loads a whole fragment for 8 lanes at a time). Closing that gap means writing the `mma.sync` / `ldmatrix` PTX by hand (what CUTLASS does) instead of using WMMA.
- **Why `cublasHgemm` shows FAIL in the table:** it writes FP16 output, which rounds every result to 11 bits of mantissa, so its error against the FP32-accumulated reference is 0.019–0.39 (relative, worst element), above the 5e-2 check. That's a precision trade-off of FP16 output, not a bug, and it's the reason the FP32-output `cublasGemmEx` is the reference. It's also why `cublasHgemm` is faster at 4096³ (31 TFLOP/s): half the bytes written.

## 09 · Triton ports: the same kernels in a higher-level language

[`scripts/triton_ports.py`](../scripts/triton_ports.py) ports softmax, LayerNorm (+ fused residual add) and attention to Triton 3.6, checks each against PyTorch, and times them with CUDA events. The CUDA numbers below are from the separate CUDA benchmarks in the same run, so small differences (a few %) are within run-to-run noise; PyTorch timings in the Triton process drifted (torch.softmax 0.909 ms there vs 0.704 ms in the CUDA comparison), so I compare absolute times rather than ratios to PyTorch.

| Kernel (4096×4096) | Triton | My CUDA | Result |
|---|---:|---:|---|
| Softmax | 0.565 ms (237.7 GB/s) | v4 0.584 ms (229.7 GB/s) | Triton 3% faster |
| LayerNorm | 0.569 ms | v3 0.568 ms | tie |
| Residual add + LayerNorm | 1.152 ms | fused 1.126 ms | CUDA 2% faster |
| Attention N = 4,096 (FP32) | 37.088 ms | v2 25.006 ms | **CUDA 1.48x faster** |

- **For memory-bound row ops, Triton matches hand-written CUDA in ~15 lines.** Both keep the row in registers and hit ~74% of peak bandwidth: the same design, so the same roofline limit.
- **For attention it doesn't, at least not in FP32 on Turing.** With `tl.dot(..., input_precision="ieee")` (no TF32 on a T4) the Triton kernel ran at 58–68% of SDPA across N, best of 4 block shapes. My guess is that Triton's FP32 `tl.dot` codegen on sm_75 doesn't register-tile as aggressively as the hand-written loop; I didn't read its SASS to confirm.
