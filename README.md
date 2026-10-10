# CUDA Kernel Lab

Hand-written CUDA kernels, each built from naive to optimized, benchmarked against cuBLAS and PyTorch, profiled with Nsight Compute, and placed on a measured roofline. 21 kernel versions across 7 problems; every result is correctness-checked before it is reported.

**GPU:** NVIDIA Tesla T4 (Turing, sm_75, 40 SMs, 320 GB/s GDDR6, 8.1 TFLOP/s FP32 peak, 70 W cap) on Google Colab · CUDA 13.0 · PyTorch 2.11.

## Highlights (all measured)

- **SGEMM at 90–93% of cuBLAS** (median 91.6%, 3 runs at 4096³): 61.7 → 3,101.6 GFLOPS (**50x**) through coalescing, shared-memory tiling, 8×8 register tiling and `float4` vectorized loads.
- **FlashAttention-style fused attention runs where the standard version can't:** at N = 16,384 the unfused path needs 17.2 GB for the score matrix and runs out of memory; the fused kernel needs no extra memory. It is **1.75x faster than unfused at N = 8,192**, moves **26x less DRAM traffic** (59 MB vs 1.5 GB at N = 2,048, Nsight counters), and reaches **~90% of PyTorch's FP32 `scaled_dot_product_attention`** at N ≥ 4,096.
- **Beats PyTorch on memory-bound ops:** softmax **1.25x** faster than `torch.softmax` (234 GB/s), LayerNorm **1.42x** faster than `F.layer_norm` (236 GB/s), fused residual-add + LayerNorm **1.44x** faster than PyTorch eager.
- **Reduction 133x faster** (1.9 → 247.8 GB/s) by replacing 16.7M atomics with a grid-stride loop and warp shuffles.
- **Bank conflicts proven with counters:** padding a transpose tile from `[32][32]` to `[32][33]` took Nsight's shared-memory bank-conflict count from **16,252,928 to 0**, a number derived exactly as 524,288 warp loads × 31 replays.

## Writeup

[I wrote 21 CUDA kernels on a free GPU. Here's what the profiler taught me.](https://dev.to/musharaf_khanpathan_90de/i-wrote-21-cuda-kernels-on-a-free-gpu-heres-what-the-profiler-taught-me-4ed1) (dev.to): the story behind the numbers, including the results that didn't go as expected.

## Results

Full table with every version, run count and check: [`benchmarks/results.md`](benchmarks/results.md) (generated from the raw logs in [`benchmarks/raw/`](benchmarks/raw)).

| Kernel | Version | Size | Time | Throughput | vs baseline | Limiter |
|---|---|---|---:|---:|---:|---|
| Vector add | naive | 16M floats | 0.763 ms | 263.8 GB/s | 82% of peak BW | memory |
| Reduction | v1 `atomicAdd` per element | 16M floats | 36.037 ms | 1.9 GB/s | 1x | contention on one address |
| Reduction | v2 shared-memory tree | 16M floats | 0.625 ms | 107.5 GB/s | 58x | barriers + idle threads |
| Reduction | v3 grid-stride + warp shuffle | 16M floats | 0.271 ms | 247.8 GB/s | **133x** | memory (77% of peak) |
| Transpose | copy (ceiling) | 4096² | 0.571 ms | 235.1 GB/s | — | memory |
| Transpose | naive | 4096² | 1.583 ms | 84.8 GB/s | 1x | uncoalesced writes |
| Transpose | smem tiled `[32][32]` | 4096² | 0.732 ms | 183.4 GB/s | 2.16x | 16.25M bank conflicts |
| Transpose | smem tiled `[32][33]` | 4096² | 0.692 ms | 193.9 GB/s | **2.29x** | memory (0 conflicts, 82% of copy) |
| SGEMM | v1 naive, uncoalesced | 4096³ | 2226.3 ms | 61.7 GFLOPS | 1.7% of cuBLAS | uncoalesced loads |
| SGEMM | v2 naive, coalesced | 4096³ | 271.7 ms | 505.8 GFLOPS | 15.1% | global memory |
| SGEMM | v3 shared-memory tiles 32×32 | 4096³ | 177.1 ms | 776.0 GFLOPS | 23.2% | shared-memory bandwidth |
| SGEMM | v4 2D register tiling 8×8/thread | 4096³ | 48.4 ms | 2,841.4 GFLOPS | 85.1% | FP32 issue |
| SGEMM | v5 v4 + `float4` + transposed A tile | 4096³ | 44.3 ms | 3,101.6 GFLOPS | **91.6%** (90–93%) | FP32 issue |
| SGEMM | cuBLAS / `torch.mm` | 4096³ | 41.2 / 41.3 ms | 3,339.8 / 3,326.9 GFLOPS | 100% | |
| Softmax | v1 thread-per-row, 3 passes | 4096² | 4.579 ms | 29.3 GB/s | 1x | uncoalesced, 16 blocks for 40 SMs |
| Softmax | v2 block-per-row, 3 passes | 4096² | 0.746 ms | 180.0 GB/s | 6.1x | memory |
| Softmax | v3 online softmax, 2 passes | 4096² | 0.751 ms | 178.8 GB/s | 6.1x | memory |
| Softmax | v4 register-cached, 1 pass | 4096² | 0.573 ms | 234.1 GB/s | **8.0x** | memory (73% of peak) |
| Softmax | PyTorch `torch.softmax` | 4096² | 0.715 ms | 187.8 GB/s | v4 is **1.25x** faster | |
| LayerNorm | v1 two-pass | 4096² | 0.870 ms | 154.3 GB/s | 1x | memory |
| LayerNorm | v2 Welford single-pass stats | 4096² | 0.816 ms | 164.5 GB/s | 1.07x | memory |
| LayerNorm | v3 register-cached | 4096² | 0.568 ms | 236.1 GB/s | 1.53x | memory (74% of peak) |
| LayerNorm | PyTorch `F.layer_norm` | 4096² | 0.806 ms | 166.5 GB/s | v3 is **1.42x** faster | |
| Add + LayerNorm | unfused (2 kernels) | 4096² | 1.370 ms | 195.9 GB/s | 1x | extra DRAM round trip |
| Add + LayerNorm | fused (1 kernel) | 4096² | 1.127 ms | 238.2 GB/s | **1.22x** | memory |
| Add + LayerNorm | PyTorch eager | 4096² | 1.623 ms | 165.3 GB/s | fused is **1.44x** faster | |

### Attention (FP32, batch 2 × 8 heads, head dim 64, non-causal)

| Sequence length N | Unfused (cuBLAS + softmax) | Fused (this repo) | Fused vs unfused | PyTorch SDPA (FP32) | Fused vs SDPA |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 1.92 ms | 2.65 ms | 0.73x | 1.66 ms | 63% |
| 2,048 | 9.07 ms | 9.26 ms | 0.98x | 6.36 ms | 69% |
| 4,096 | 43.0 ms | 28.8 ms | **1.49x** | 25.6 ms | 89% |
| 8,192 | 196.6 ms (4.3 GB of scores) | 112.2 ms | **1.75x** | 100.6 ms | 90% |
| 16,384 | **out of memory** (needs 17.2 GB) | 453.6 ms | runs | 411.9 ms | 91% |

The fused kernel wins as N grows because the unfused path's cost is dominated by writing and re-reading the N×N score matrix. At small N, cuBLAS's tensor-sized GEMMs are simply more efficient than one-thread-per-row FP32 code. PyTorch's FP32 SDPA on a T4 dispatches to a CUTLASS memory-efficient attention kernel, a strong production baseline.

## Roofline

![Roofline](benchmarks/roofline.png)

Every point is measured with Nsight Compute hardware counters, not estimated: FLOPs = `2·FFMA + FADD + FMUL` executed thread instructions, bytes = real DRAM reads + writes after the caches, time = `gpu__time_duration`. The GEMM versions march up and to the right from memory-bound to compute-bound; the memory-bound ops sit on the bandwidth roof. All versions of the memory-bound kernels: [`roofline_memory.png`](benchmarks/roofline_memory.png). Data: [`roofline.csv`](benchmarks/roofline.csv).

Caveat: profiling is one launch per kernel (GEMM and attention at n = 2048), with the GPU not yet power-throttled, so absolute GFLOP/s on the chart are higher than the sustained numbers in the table. Use the table for speed claims and the roofline for *why*.

## What each kernel teaches

| # | Kernel | Versions | The lesson |
|---|---|---|---|
| 01 | [Vector add](kernels/01_vecadd) | naive | Thread indexing, bounds guards, CUDA-event timing, bandwidth as the metric for memory-bound work. |
| 02 | [Reduction](kernels/02_reduction) | atomic → shared tree → warp shuffle | Contention kills parallelism; do work in registers and reduce inside warps with shuffles. |
| 03 | [Transpose](kernels/03_transpose) | naive → tiled → padded | Coalescing, shared-memory staging, bank conflicts and the `[32][33]` fix, proven with `ncu` counters. |
| 04 | [SGEMM](kernels/04_gemm) | 5 versions + cuBLAS | Each fix exposes the next bottleneck: DRAM → shared memory → registers → instruction issue. |
| 05 | [Softmax](kernels/05_softmax) | thread/row → 3-pass → online → registers | Numerical stability (overflow demo), online softmax, and cutting passes over memory. |
| 06 | [LayerNorm](kernels/06_layernorm) | two-pass → Welford → registers, + fusion | Stable single-pass variance and why fusing the residual add saves a DRAM round trip. |
| 07 | [Attention](kernels/07_attention) | unfused (cuBLAS) vs fused | FlashAttention's core idea: tile K/V through shared memory, keep scores in registers, fold them in with online softmax. Memory O(N²) → O(N). |

What each optimization taught, with the profiler evidence: [`docs/learnings.md`](docs/learnings.md).

## How correctness is verified

A fast wrong kernel is worthless, so every benchmark checks its output before it reports a number. All 46 checked results in the last run passed (counting each of the 3 GEMM runs; the one non-result is the expected out-of-memory above). PyTorch baseline rows are timings only.

| Kernel | Reference | Tolerance | Worst measured |
|---|---|---|---|
| Vector add, reduction, transpose | exact expected values | bit-exact | exact |
| SGEMM | cuBLAS (and cuBLAS vs a double-precision CPU GEMM at n = 256) | max rel. error < 1e-3 | 4.6e-6 |
| Softmax | double-precision CPU softmax | max **true** relative error < 1e-4 | 4.3e-6 |
| LayerNorm, add + LayerNorm | double-precision CPU reference | \|a−b\| / (1+\|ref\|) < 1e-4 | 3.2e-7 |
| Attention | double-precision CPU at N = 256, unfused GPU result above | < 1e-3 | 2.4e-7 |

NaN/Inf always counts as a failure (a plain `fmaxf` would silently ignore NaN and report "PASS"). At N = 16,384 there is no unfused reference, so the fused output is checked for NaN/Inf only.

## Run it

**Colab, no local GPU needed:** open [`run_all.ipynb`](https://colab.research.google.com/github/Musharaf-khan/cuda-kernel-lab/blob/main/run_all.ipynb), choose the T4 runtime, Run all (~15 min). It builds everything, runs every benchmark, the PyTorch baselines and the Nsight Compute counters, then writes `benchmarks/results.md`, the roofline charts and a `results.zip`.

**Any machine with an NVIDIA GPU and the CUDA toolkit:**

```bash
make                       # builds into build/ for sm_75; make ARCH=sm_86 etc. for other GPUs
bash scripts/run_all.sh    # the whole pipeline
./build/gemm 256           # quick correctness run (also checks cuBLAS vs CPU)
./build/attention          # attention at N = 256 ... 16384
```

Single files compile on their own too: `nvcc -O3 -arch=sm_75 kernels/04_gemm/gemm.cu -o gemm -lcublas`.

## Methodology

- Kernel time only: host↔device copies are never inside the timed region.
- One warm-up launch, then the average of 5–100 launches timed with CUDA events.
- The T4 is power-capped and its clocks drift as it heats up (cuBLAS ranged 3,165–3,575 GFLOPS across three runs), so comparisons are made **within the same run**: cuBLAS is timed before *and* after our kernels and averaged, and GEMM is reported as the median of 3 runs.
- Under Nsight Compute, `--clock-control none` keeps the normal clock behaviour instead of ncu's default lock to base clock.

## Layout

```
common/                   CHECK macro, timing, RESULT reporting, warp/block reductions
kernels/01_vecadd .. 07_attention    one .cu per problem: every version + benchmark + checks
scripts/run_all.sh        the whole pipeline
scripts/make_table.py     RESULT lines -> benchmarks/results.md
scripts/roofline.py       ncu counters -> roofline charts + csv
scripts/torch_compare.py  PyTorch baselines on identical shapes
benchmarks/               raw logs, ncu counters and report, results table, roofline
docs/learnings.md         what each optimization taught, with the profiler evidence
```
