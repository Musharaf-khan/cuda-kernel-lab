# CUDA Kernel Lab

Hand-written CUDA kernels, each built from naive to optimized, benchmarked against cuBLAS and PyTorch, profiled with Nsight Compute, checked at the SASS level, and placed on a measured roofline. There are 25 kernel versions across 8 problems, plus a 13-step SGEMM attempt log, an autotuner and 3 Triton ports. Every result is correctness-checked before it is reported, and the results that didn't work are kept.

**GPU:** NVIDIA Tesla T4 (Turing, sm_75, 40 SMs, 320 GB/s GDDR6, 8.1 TFLOP/s FP32 / 65 TFLOP/s FP16 Tensor Core peak, 70 W cap) on Google Colab · CUDA 13.0 · PyTorch 2.11 · Triton 3.6.

## Highlights (all measured)

- **SGEMM at parity with cuBLAS:** autotuned v6 reaches **4,045.6 GFLOPS = 101.3% of cuBLAS** at 4096³ (median of 3 same-run ratios: 104.0 / 100.7 / 101.3), up from 61.8 GFLOPS for the naive kernel (**65x**). What closed the last 10% was **warp tiling, which removed 268M shared-memory bank conflicts** that Nsight counted on the v5 kernel. In a single unthrottled profiler launch cuBLAS is still ~10% faster; see [the attempt log](docs/learnings.md#closing-the-last-10-the-attempt-log).
- **Autotuner fixes small sizes:** at n = 256 the 4096-tuned kernel ran at 31% of cuBLAS; the autotuned 64×64 configuration runs at **94.1% (3.0x faster)**.
- **FP16 Tensor Core GEMM (WMMA):** **24.8 TFLOP/s** at 2048³, 3.9x the FP32 cuBLAS kernel at 4096³, and **68–88% of `cublasGemmEx`**.
- **FlashAttention-style fused attention** runs where the standard version can't: at N = 16,384 the unfused path needs 17.2 GB for the score matrix and runs out of memory. Fused v2 is **1.69x faster than unfused at N = 8,192**, moves **26x less DRAM traffic** (59 MB vs ~1.5 GB at N = 2,048), and reaches **94.5–95.8% of PyTorch's FP32 SDPA** at N ≥ 4,096.
- **Beats PyTorch on memory-bound ops:** softmax **1.21x** faster than `torch.softmax`, LayerNorm **1.42x** faster than `F.layer_norm`, fused residual-add + LayerNorm **1.44x** faster than PyTorch eager.
- **Reduction 131x faster** (1.9 → 248.3 GB/s) by replacing 16.7M atomics with a grid-stride loop and warp shuffles.
- **Bank conflicts proven with counters:** padding a transpose tile from `[32][32]` to `[32][33]` took Nsight's bank-conflict count from **16,252,928 to 0**. That number derives exactly as 524,288 warp loads × 31 replays.

## Writeup

[I wrote 21 CUDA kernels on a free GPU. Here's what the profiler taught me.](https://dev.to/musharaf_khanpathan_90de/i-wrote-21-cuda-kernels-on-a-free-gpu-heres-what-the-profiler-taught-me-4ed1) (dev.to) tells the story behind the numbers, including the results that didn't go as expected. It covers the first 21 versions; the SGEMM attempt log, the autotuner, WMMA and attention v2 came after it.

## Results

The full table, with every version, run count and check, is in [`benchmarks/results.md`](benchmarks/results.md). It is generated from the raw logs in [`benchmarks/raw/`](benchmarks/raw).

| Kernel | Version | Size | Time | Throughput | vs baseline | Limiter |
|---|---|---|---:|---:|---:|---|
| Vector add | naive | 16M floats | 0.764 ms | 263.5 GB/s | 82% of peak BW | memory |
| Reduction | v1 `atomicAdd` per element | 16M floats | 35.502 ms | 1.9 GB/s | 1x | contention on one address |
| Reduction | v2 shared-memory tree | 16M floats | 0.624 ms | 107.6 GB/s | 57x | barriers + idle threads |
| Reduction | v3 grid-stride + warp shuffle | 16M floats | 0.270 ms | 248.3 GB/s | **131x** | memory (78% of peak) |
| Transpose | copy (ceiling) | 4096² | 0.571 ms | 235.1 GB/s | — | memory |
| Transpose | naive | 4096² | 1.524 ms | 88.1 GB/s | 1x | uncoalesced writes |
| Transpose | smem tiled `[32][32]` | 4096² | 0.732 ms | 183.3 GB/s | 2.08x | 16.25M bank conflicts |
| Transpose | smem tiled `[32][33]` | 4096² | 0.689 ms | 194.8 GB/s | **2.21x** | memory (0 conflicts, 83% of copy) |
| SGEMM | v1 naive, uncoalesced | 4096³ | 2224.9 ms | 61.8 GFLOPS | 1.5% of cuBLAS | uncoalesced loads |
| SGEMM | v2 naive, coalesced | 4096³ | 244.1 ms | 563.0 GFLOPS | 14.1% | global memory |
| SGEMM | v3 shared-memory tiles 32×32 | 4096³ | 157.0 ms | 875.2 GFLOPS | 21.8% | shared-memory bandwidth |
| SGEMM | v4 2D register tiling 8×8/thread | 4096³ | 42.7 ms | 3,215.1 GFLOPS | 81.8% | FP32 issue |
| SGEMM | v5 v4 + `float4` global loads/stores | 4096³ | 39.0 ms | 3,522.6 GFLOPS | 88.4% | 2-way shared-load bank conflicts |
| SGEMM | **v6 autotuned**: warp tiling + double buffering + padding | 4096³ | 34.0 ms | **4,045.6 GFLOPS** | **101.3%** (100.7–104.0%) | FP32 issue |
| SGEMM | cuBLAS / `torch.mm` | 4096³ | 34.4 / 36.8 ms | 3,999.5 / 3,736.4 GFLOPS | 100% | |
| SGEMM | v5 / **v6 autotuned (64×64)** / cuBLAS | 256³ | 0.062 / 0.021 / 0.019 ms | 540.6 / **1,631.5** / 1,717.6 GFLOPS | 31.3% / **94.1%** / 100% | parallelism (4 vs 16 blocks) |
| Softmax | v1 thread-per-row, 3 passes | 4096² | 3.664 ms | 36.6 GB/s | 1x | uncoalesced, 16 blocks for 40 SMs |
| Softmax | v2 block-per-row, 3 passes | 4096² | 0.697 ms | 192.5 GB/s | 5.3x | memory |
| Softmax | v3 online softmax, 2 passes | 4096² | 0.760 ms | 176.6 GB/s | 4.8x (slower than v2) | memory |
| Softmax | v4 register-cached, 1 pass | 4096² | 0.584 ms | 229.7 GB/s | **6.3x** | memory (72% of peak) |
| Softmax | PyTorch `torch.softmax` | 4096² | 0.704 ms | 190.5 GB/s | v4 is **1.21x** faster | |
| LayerNorm | v1 two-pass | 4096² | 0.927 ms | 144.8 GB/s | 1x | memory |
| LayerNorm | v2 Welford single-pass stats | 4096² | 0.870 ms | 154.3 GB/s | 1.07x | memory |
| LayerNorm | v3 register-cached | 4096² | 0.568 ms | 236.3 GB/s | 1.63x | memory (74% of peak) |
| LayerNorm | PyTorch `F.layer_norm` | 4096² | 0.805 ms | 166.8 GB/s | v3 is **1.42x** faster | |
| Add + LayerNorm | unfused (2 kernels) | 4096² | 1.372 ms | 195.6 GB/s | 1x | extra DRAM round trip |
| Add + LayerNorm | fused (1 kernel) | 4096² | 1.126 ms | 238.5 GB/s | **1.22x** | memory |
| Add + LayerNorm | PyTorch eager | 4096² | 1.625 ms | 165.2 GB/s | fused is **1.44x** faster | |

### SGEMM: closing the gap to cuBLAS (4096³, % of cuBLAS in the same run, median of 3)

Each change was applied to v5 one at a time, then the winners were stacked, with Nsight counters for every step. The full log, with the reason behind each result, is in [`docs/learnings.md`](docs/learnings.md#closing-the-last-10-the-attempt-log).

| Attempt | % of cuBLAS | Shared-load bank conflicts (ncu) | FMA pipe busy |
|---|---:|---:|---:|
| A0 v5 baseline | 90.1% | 268,435,456 | 66% |
| B1 + warp tiling | **99.4%** | **0** | 82% |
| B2 + double buffering (alone) | 88.0% | 268,435,456 | 67% |
| B3a/b 128 / 512 threads | 92.3% / 81.5% | 268M / 0 | 69% / 56% |
| B3c `__launch_bounds__(256,2)` | 88.8% | 268,435,456 | 66% |
| B4 padded A tile (alone) | 91.1% | 268,435,456 (store conflicts → 0) | 67% |
| B5 BK = 16 | 88.2% | 268,435,456 | 66% |
| C2 warp tiling + double buffering | 97.6% | 0 | 93.5% |
| **C3 C2 + padded A tile (= v6)** | **100.2%** | 52,720 | 92% |
| C4 / C4b / C5 + BK = 16 variants | 94.4% / 94.2% / 86.3% | 0 | 67–75% (occupancy halves; C5 spills) |

The search stopped after C4, C4b and C5: three consecutive changes, each a loss. Under a single unthrottled ncu launch, cuBLAS (`volta_sgemm_128x64_nn`, FMA pipe 97.6% busy) is still ~10% faster than C3 (19.8 vs 22.3 ms). Parity holds in the sustained, power-capped runs that the table measures.

### FP16 Tensor Core GEMM (WMMA, FP16 in, FP32 accumulate)

| n | v1 WMMA naive | v2 WMMA tiled | `cublasGemmEx` (same math) | v2 % of cuBLAS | `cublasSgemm` FP32 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 3.6 TFLOP/s | **23.0** | 33.7 | 68.1% | 5.8 |
| 2048 | 2.6 | **24.8** | 28.3 | 87.5% | 4.6 |
| 4096 | 2.1 | **15.6** | 22.7 | 68.8% | 4.0 |

cuBLAS drifted heavily within this run (at 2048³: 0.42 ms before our kernels, 0.79 ms after), so read the ratio as a range. A single ncu launch at 2048³ gives **71%**. The remaining gap is in the SASS: WMMA's fragment loads compile to scalar `LDS.32`, while cuBLAS uses `ldmatrix`. In the full table `cublasHgemm` shows "FAIL" because its FP16 output differs from the FP32-accumulated reference by up to 0.39 (relative). That is an FP16-output precision trade-off, not a bug, and it is why the FP32-output `cublasGemmEx` is the reference.

### Attention (FP32, batch 2 × 8 heads, head dim 64, non-causal)

| Sequence length N | Unfused (cuBLAS + softmax) | Fused v1 (1 thread/row) | Fused v2 (2 threads/row) | v1 vs unfused | v2 vs unfused | PyTorch SDPA (FP32) | v1 vs SDPA | **v2 vs SDPA** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.227 ms | 0.259 ms | 0.168 ms | 0.88x | 1.36x | — | — | — |
| 1,024 | 2.031 ms | 2.665 ms | 1.647 ms | **0.76x** | 1.23x | 1.507 ms | 57% | 91.5% |
| 2,048 | 7.584 ms | 7.950 ms | 6.179 ms | **0.95x** | 1.23x | 5.772 ms | 73% | 93.4% |
| 4,096 | 36.288 ms | 26.349 ms | 25.006 ms | 1.38x | 1.45x | 23.965 ms | 91% | **95.8%** |
| 8,192 | 169.464 ms (4.3 GB of scores) | 101.002 ms | 100.086 ms | 1.68x | **1.69x** | 94.570 ms | 94% | **94.5%** |
| 16,384 | **out of memory** (needs 17.2 GB) | 417.135 ms | 406.694 ms | runs | runs | 385.086 ms | 92% | **94.7%** |

v2 uses one fixed configuration (128 rows per block, K/V tile 32) at every N. At N = 2,048 a 64-row variant was faster (5.829 ms, 99% of SDPA); all four variants are in `results.md`.

- **v1 still loses below ~2K tokens.** At small N, cuBLAS's GEMMs are simply more efficient than one-thread-per-row FP32 code, so the score matrix is cheap enough not to matter.
- **v2 halves v1's register count by splitting each row across 2 lanes** (254 → 159 registers). That makes it faster than unfused at every size measured, but it still sits **~5% short of SDPA** (PyTorch dispatches to a CUTLASS memory-efficient kernel). The goal of matching SDPA at N ≥ 4K was not met.
- **4 threads per row was slower** (0.6x of v1): the extra shuffles saturate the MIO queue (14.6 stall cycles per instruction, ncu).

### Triton ports

[`scripts/triton_ports.py`](scripts/triton_ports.py) ports three of the kernels to Triton, checks each against PyTorch and times them on the same shapes:

| Kernel | Triton | This repo's CUDA | Result |
|---|---:|---:|---|
| Softmax 4096² | 0.565 ms | 0.584 ms | Triton 3% faster |
| LayerNorm 4096² | 0.569 ms | 0.568 ms | tie |
| Add + LayerNorm 4096² | 1.152 ms | 1.126 ms | CUDA 2% faster |
| Attention N = 4,096 (FP32) | 37.088 ms | 25.006 ms | CUDA **1.48x** faster |

For memory-bound row ops, about 15 lines of Triton reach the same bandwidth roof as hand-written CUDA. For FP32 attention on Turing, the Triton kernel was 58–68% of SDPA.

## Roofline

![Roofline](benchmarks/roofline.png)

Every point is measured with Nsight Compute hardware counters, not estimated:
- FLOPs = `2·FFMA + FADD + FMUL` executed thread instructions.
- Bytes = real DRAM reads + writes after the caches.
- Time = `gpu__time_duration`.

The GEMM versions march up and to the right, from memory-bound to compute-bound, and the memory-bound ops sit on the bandwidth roof. [`roofline_memory.png`](benchmarks/roofline_memory.png) shows every version of the memory-bound kernels; the data is in [`roofline.csv`](benchmarks/roofline.csv). Tensor Core FLOPs (HMMA) are not counted by these FP32 metrics, so the WMMA kernels are not on the chart.

Caveat: profiling is one launch per kernel (GEMM and attention at n = 2048), with the GPU not yet power-throttled. Absolute GFLOP/s on the chart are therefore higher than the sustained numbers in the table. Use the table for speed claims and the roofline for *why*.

## What each kernel teaches

| # | Kernel | Versions | The lesson |
|---|---|---|---|
| 01 | [Vector add](kernels/01_vecadd) | naive | Thread indexing, bounds guards, CUDA-event timing, bandwidth as the metric for memory-bound work. |
| 02 | [Reduction](kernels/02_reduction) | atomic → shared tree → warp shuffle | Contention kills parallelism; do work in registers and reduce inside warps with shuffles. |
| 03 | [Transpose](kernels/03_transpose) | naive → tiled → padded | Coalescing, shared-memory staging, and bank conflicts with the `[32][33]` fix, proven with `ncu` counters. |
| 04 | [SGEMM](kernels/04_gemm) | 6 versions + cuBLAS, 13-step attempt log, autotuner | Each fix exposes the next bottleneck: DRAM → shared memory → registers → bank conflicts → instruction issue. No single tile size wins at every n. |
| 05 | [Softmax](kernels/05_softmax) | thread/row → 3-pass → online → registers | Numerical stability (overflow demo), online softmax, and cutting passes over memory. |
| 06 | [LayerNorm](kernels/06_layernorm) | two-pass → Welford → registers, + fusion | Stable single-pass variance, and why fusing the residual add saves a DRAM round trip. |
| 07 | [Attention](kernels/07_attention) | unfused (cuBLAS) vs fused v1 vs fused v2 | FlashAttention's core idea: tile K/V through shared memory, keep scores in registers, fold them in with online softmax. Memory O(N²) → O(N). Then trade registers for parallelism. |
| 08 | [HGEMM WMMA](kernels/08_hgemm_wmma) | naive WMMA → tiled + double-buffered | Tensor Cores: fragments, warp-wide `mma_sync`, and why they make the memory problem worse, not better. |

[`docs/learnings.md`](docs/learnings.md) covers what each optimization taught, with the profiler evidence.

## How correctness is verified

A fast wrong kernel is worthless, so every benchmark checks its output before it reports a number. In the last run, 181 checked results passed. This counts every repeated run and every attempt; PyTorch rows are timings only. There were 3 "FAIL"s, all `cublasHgemm`'s FP16 output (explained above), and the one non-result is the expected out-of-memory.

| Kernel | Reference | Tolerance | Worst measured |
|---|---|---|---|
| Vector add, reduction, transpose | exact expected values | bit-exact | exact |
| SGEMM (all versions, attempts, autotuner configs) | cuBLAS (and cuBLAS vs a double-precision CPU GEMM at n = 256) | max rel. error < 1e-3 | 4.6e-6 |
| HGEMM WMMA | `cublasGemmEx` (and it vs a double-precision CPU GEMM at n = 256) | max rel. error < 5e-2 | 5.2e-6 |
| Softmax | double-precision CPU softmax | max **true** relative error < 1e-4 | 4.3e-6 |
| LayerNorm, add + LayerNorm | double-precision CPU reference | \|a−b\| / (1+\|ref\|) < 1e-4 | 3.2e-7 |
| Attention v1 and v2 | double-precision CPU at N = 256, unfused GPU result above | < 1e-3 | 2.4e-7 |
| Triton ports | PyTorch (`torch.allclose`) | rtol 1e-4 (attention 1e-3) | pass |

NaN/Inf always counts as a failure (a plain `fmaxf` would silently ignore NaN and report "PASS"). At N = 16,384 there is no unfused reference, so v1 is checked for NaN/Inf and v2 against v1.

## Run it

**Colab, no local GPU needed:** open [`run_all.ipynb`](https://colab.research.google.com/github/Musharaf-khan/cuda-kernel-lab/blob/main/run_all.ipynb), choose the T4 runtime, and Run all (~30–40 min). It does the following:
- builds everything;
- runs the autotuner, every benchmark, the SGEMM attempt ladder, and the PyTorch and Triton baselines;
- collects the Nsight Compute counters and SASS instruction mix;
- writes `benchmarks/results.md`, the roofline charts and a `results.zip`.

**Any machine with an NVIDIA GPU and the CUDA toolkit:**

```bash
make                            # builds into build/ for sm_75; make ARCH=sm_86 etc. for other GPUs
bash scripts/run_all.sh         # the whole pipeline (SKIP_NCU=1 to skip profiling)
python3 scripts/autotune.py     # re-tune SGEMM for your GPU -> configs/sgemm_autotune.json
./build/gemm 256                # quick correctness run (also checks cuBLAS vs CPU)
./build/gemm_lab attempts 4096  # the SGEMM attempt ladder
./build/attention               # attention at N = 256 ... 16384
./build/hgemm                   # WMMA Tensor Core GEMM
```

Single files compile on their own too: `nvcc -O3 -arch=sm_75 kernels/04_gemm/gemm.cu -o gemm -lcublas`.

## Methodology

- **Kernel time only:** host↔device copies are never inside the timed region.
- **Timing:** one warm-up launch, then the average of 5–100 launches timed with CUDA events.
- **Same-run comparisons:** the T4 is power-capped, and its clocks drift as it heats up. cuBLAS ranged 3,808–3,999 GFLOPS across benchmarks in this run, and 0.42 → 0.79 ms for one FP16 shape. So comparisons are made **within the same run**: cuBLAS is timed before *and* after our kernels and averaged. GEMM and the attempt ladder run 3 times. The "% of cuBLAS" column is the median of the per-run ratios, and time/throughput come from the median-time run. Nothing is best-of-N.
- **Profiler clocks:** under Nsight Compute, `--clock-control none` keeps the normal clock behaviour instead of ncu's default lock to base clock. Single-launch profiles run before the GPU throttles, which is why ncu ratios can differ from sustained ones; both are reported.

## Layout

```
common/                   CHECK macro, timing, RESULT reporting, warp/block reductions
kernels/01_vecadd .. 08_hgemm_wmma    one .cu per problem: every version + benchmark + checks
kernels/04_gemm/sgemm_lab.cuh         templated SGEMM used by the attempt log, autotuner and v6
configs/sgemm_autotune.json           autotuner output, read by build/gemm
scripts/run_all.sh        the whole pipeline
scripts/autotune.py       SGEMM config sweep -> configs/sgemm_autotune.json
scripts/make_table.py     RESULT lines -> benchmarks/results.md
scripts/roofline.py       ncu counters -> roofline charts + csv
scripts/sass_stats.py     cuobjdump SASS -> instruction mix per kernel
scripts/torch_compare.py  PyTorch baselines on identical shapes
scripts/triton_ports.py   Triton softmax, LayerNorm, attention
benchmarks/               raw logs, ncu counters, SASS instruction mix, results table, roofline
docs/learnings.md         what each optimization taught, with the profiler evidence
```
