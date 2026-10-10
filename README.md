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

## Build and run
```
nvcc -arch=sm_75 kernels/01_vecadd/vecadd_bench.cu -o vecadd_bench && ./vecadd_bench
```