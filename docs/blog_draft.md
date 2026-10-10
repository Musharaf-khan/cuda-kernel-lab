# I wrote CUDA kernels from scratch on a free GPU. Here's what the profiler taught me.

*Draft. Lines marked [FILL] need numbers from `benchmarks/results.md` after running `run_all.ipynb`.*

## Why

Every GPU software posting I read asked about coalescing, occupancy and shared memory. I could recite the definitions but had never seen any of it happen. So I set myself a rule: write each kernel naively first, measure it, then make one change at a time and prove with Nsight Compute why it got faster. All of it ran on a free Colab Tesla T4 from a MacBook that has no NVIDIA GPU at all.

## The one idea that unlocked everything: the memory hierarchy

A GPU is mostly a machine for moving data. My first kernel, vector add, hit 262.6 GB/s, 82% of the T4's 320 GB/s spec, and could not go faster no matter what, because it does one addition per 12 bytes. Almost every speedup after that was the same move: get the data one level closer to the arithmetic, from DRAM to shared memory to registers.

- A reduction went from 1.9 GB/s to 248 GB/s (**128x**) by summing in registers and combining with warp shuffles instead of 16.7 million atomics on one address.
- In SGEMM, swapping two index lines so a warp reads consecutive addresses made it **8.8x** faster.

## The bug I'm proudest of finding: 16,252,928 bank conflicts

My shared-memory transpose was barely faster than the naive one. Nsight Compute said: 16,252,928 shared-memory bank conflicts. Reading a column of a `float[32][32]` tile puts all 32 threads of a warp in the same bank. And the count is exact: 4096² / 32 = 524,288 warp loads, each replayed 31 extra times. Padding the tile to `[32][33]` dropped the counter to **0** and the kernel got 1.8x faster. That was the moment profiling stopped being a chore and became the fun part.

## The GEMM ladder

| Version | GFLOPS | % of cuBLAS |
|---|---:|---:|
| naive | 61.8 | 1.7% |
| coalesced | 543.7 | 14.5% |
| shared-memory tiles | 838.0 | 22.4% |
| 8×8 register tiles | 3,098.7 | 82.7% |
| + float4, transposed A | [FILL] | [FILL] |

The surprise was the third row. Tiling cut global memory traffic 32x and the kernel only got 1.5x faster, because the bottleneck moved: the inner loop did one multiply-add per two shared-memory loads. Register tiling fixed that by making each loaded value feed 8 multiply-adds.

Also humbling: cuBLAS measured 5,274 GFLOPS in one run and 3,597 in another. Same code, a hotter GPU. The ratio between my kernel and cuBLAS stayed within one percent, which is why every number I report is a same-run ratio or a median.

## Softmax, LayerNorm and fusion

- Softmax: [FILL: v1 → v4 GB/s]. Online softmax cut the passes over memory from 3 to 2, and keeping the row in registers cut it to 1.
- LayerNorm: [FILL]. Fusing the residual add saved [FILL]% because the intermediate never goes back to DRAM.

## Attention without the N×N matrix

The textbook implementation writes all the attention scores to memory: 4.3 GB at 8K tokens, and at 16K tokens it doesn't fit on the GPU at all. My fused kernel streams K and V through shared memory and folds each tile into the output with online softmax, so the score matrix never exists. [FILL: fused vs unfused times, N=16384 result, comparison with PyTorch SDPA, and an honest note on where it is slower.]

## The roofline

[roofline.png] Every point is measured with hardware counters. The GEMM versions march up and to the right toward the compute roof; the softmax and LayerNorm kernels sit on the memory roof, which is exactly where a memory-bound kernel should be.

## What's next

Tensor Cores (WMMA), double buffering, and plugging these kernels into a small LLM inference server.

*Code, numbers and profiler reports: github.com/Musharaf-khan/cuda-kernel-lab*
