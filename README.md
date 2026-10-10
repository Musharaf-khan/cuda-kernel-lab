# CUDA Kernel Lab

Hand-written CUDA kernels, from naive to optimized, benchmarked and profiled.

**GPU:** NVIDIA Tesla T4 (Turing, sm_75, 320 GB/s peak memory bandwidth) on Google Colab

## Results

| Kernel | Version | Size | Time | Throughput | % of peak | Bound |
|---|---|---|---|---|---|---|
| Vector add | naive | 16M floats | 0.767 ms | 262.6 GB/s | 82% | memory |
| Reduction (sum) | v1 atomic | 16M floats | 34.672 ms | 1.9 GB/s | 1% | memory (atomic contention) |
| Reduction (sum) | v2 shared-mem tree | 16M floats | 0.623 ms | 107.7 GB/s | 34% | memory (sync overhead) |
| Reduction (sum) | v3 grid-stride + warp shuffle | 16M floats | 0.271 ms | 248.0 GB/s | 78% | memory |
   | Transpose 4096² | copy (ceiling) | 64 MB | 0.561 ms | 239.0 GB/s | 75% | memory |
   | Transpose 4096² | naive | 64 MB | 2.234 ms | 60.1 GB/s | 19% | memory (uncoalesced writes) |
   | Transpose 4096² | tiled, smem [32][32] | 64 MB | 1.270 ms | 105.7 GB/s | 33% | shared-mem bank conflicts (16.25M) |
   | Transpose 4096² | tiled, smem [32][33] | 64 MB | 0.705 ms | 190.3 GB/s | 59% | memory (0 bank conflicts) |

   ## Key learnings
   - **Reduction:** replacing per-element atomics with grid-stride + warp shuffle gave 128x (1.9 → 248 GB/s).
   - **Transpose:** padding the shared-memory tile from [32][32] to [32][33] eliminated 16,252,928 bank conflicts (= 524,288 warp loads × 31 replays, verified with Nsight Compute) and gave 1.8x.

## Build and run
```
nvcc -arch=sm_75 kernels/01_vecadd/vecadd_bench.cu -o vecadd_bench && ./vecadd_bench
```