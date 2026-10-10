# Resume bullets

Every number below is measured and traceable to [`benchmarks/results.md`](../benchmarks/results.md) or the Nsight Compute counters in [`benchmarks/ncu/`](../benchmarks/ncu). Expect an interviewer to pick any one and ask "how?": the answers are in [`learnings.md`](learnings.md).

## Project entry (pick 3–4 bullets)

**CUDA Kernel Lab** · CUDA C++, cuBLAS, Nsight Compute, PyTorch · [github.com/Musharaf-khan/cuda-kernel-lab](https://github.com/Musharaf-khan/cuda-kernel-lab)

- Implemented 21 CUDA kernel versions across 7 problems (reduction, transpose, SGEMM, softmax, LayerNorm, fused attention), each benchmarked on a Tesla T4 and verified against cuBLAS/CPU references with stated tolerances.
- Optimized FP32 SGEMM from 62 to 3,100 GFLOPS (50x) via coalescing, shared-memory tiling, 8×8 register tiling and float4 vectorization, reaching 92% of cuBLAS at 4096³.
- Built a FlashAttention-style fused attention kernel using tiled K/V and online softmax: 1.75x faster than unfused cuBLAS attention at 8K tokens, 26x less DRAM traffic, runs 16K-token sequences where the unfused version runs out of memory, and reaches 90% of PyTorch SDPA.
- Wrote softmax and LayerNorm kernels 1.25x and 1.42x faster than PyTorch, and a fused residual-add + LayerNorm kernel 1.44x faster than PyTorch eager, by keeping rows in registers and cutting DRAM passes.
- Profiled every kernel with Nsight Compute and plotted measured roofline charts; diagnosed and eliminated 16.25M shared-memory bank conflicts in a tiled transpose (verified 0 via hardware counters).

## One-line versions (for a skills-heavy resume)

- CUDA Kernel Lab: 21 hand-written CUDA kernels; SGEMM at 92% of cuBLAS; fused FlashAttention-style kernel 1.75x faster than unfused with 26x less memory traffic; softmax/LayerNorm 1.25–1.42x faster than PyTorch; Nsight Compute roofline analysis.

## Keywords this project lets you list truthfully

CUDA C/C++ · GPU kernel optimization · cuBLAS · Nsight Compute · roofline analysis · memory coalescing · shared memory / bank conflicts · warp-level primitives · register tiling · kernel fusion · FlashAttention / online softmax · PyTorch · performance profiling · benchmarking

## Notes

- "92% of cuBLAS" is the median of three runs (90.0 / 92.9 / 91.6%). If asked, say "90 to 93 percent, same-run ratio, median of three". If you want the most conservative wording, write "over 90% of cuBLAS".
- Don't round attention up: it is *slower* than unfused below about 2K tokens and ~10% slower than PyTorch SDPA. Saying so in an interview reads as understanding, not weakness.
- All numbers are FP32 on a T4. Don't imply Tensor Cores, FP16 or newer GPUs.
