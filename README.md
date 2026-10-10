# CUDA Kernel Lab

Hand-written CUDA kernels, from naive to optimized, benchmarked and profiled.

**GPU:** NVIDIA Tesla T4 (Turing, sm_75, 320 GB/s peak memory bandwidth) on Google Colab

## Results

| Kernel | Version | Size | Time | Throughput | % of peak | Bound |
|---|---|---|---|---|---|---|
| Vector add | naive | 16M floats | 0.767 ms | 262.6 GB/s | 82% | memory |

## Build and run
```
nvcc -arch=sm_75 kernels/01_vecadd/vecadd_bench.cu -o vecadd_bench && ./vecadd_bench
```