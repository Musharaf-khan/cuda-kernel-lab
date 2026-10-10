| Kernel | Size | Version | Time (ms) | Throughput | Speedup | % of | Check |
|---|---|---|---:|---:|---:|---:|---|
| Vector add | 16M floats | naive | 0.763 | 263.8 GB/s | 1.00x | 82.4% of peak BW | PASS |
| Reduction (sum) | 16M floats | v1 atomicAdd per element | 36.037 | 1.9 GB/s | 1.00x | 0.6% of peak BW | PASS |
| Reduction (sum) | 16M floats | v2 shared-mem tree | 0.625 | 107.5 GB/s | 57.71x | 33.6% of peak BW | PASS |
| Reduction (sum) | 16M floats | v3 grid-stride + warp shuffle | 0.271 | 247.8 GB/s | 133.08x | 77.4% of peak BW | PASS |
| Transpose | 4096x4096 | copy (ceiling) | 0.571 | 235.1 GB/s | 2.77x | 73.5% of peak BW | PASS |
| Transpose | 4096x4096 | naive (uncoalesced writes) | 1.583 | 84.8 GB/s | 1.00x | 26.5% of peak BW | PASS |
| Transpose | 4096x4096 | smem tiled [32][32] (bank conflicts) | 0.732 | 183.4 GB/s | 2.16x | 57.3% of peak BW | PASS |
| Transpose | 4096x4096 | smem tiled [32][33] (padded) | 0.692 | 193.9 GB/s | 2.29x | 60.6% of peak BW | PASS |
| SGEMM | 4096^3 | v1 naive (uncoalesced) | 2226.330 | 61.7 GFLOPS | 1.00x | 1.7% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | v2 naive (coalesced) | 271.708 | 505.8 GFLOPS | 8.19x | 15.1% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | v3 shared-mem tiled 32x32 | 177.117 | 776.0 GFLOPS | 12.57x | 23.2% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | v4 2D register tiling 8x8/thread | 48.369 | 2,841.4 GFLOPS | 46.03x | 85.1% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | v5 v4 + float4 + transposed A tile | 44.312 | 3,101.6 GFLOPS | 50.24x | 92.9% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | cuBLAS (cublasSgemm) | 41.152 | 3,339.8 GFLOPS | 54.10x | 100.0% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | PyTorch torch.mm (cuBLAS) | 41.312 | 3,326.9 GFLOPS | 53.89x | - | baseline (timing only) |
| Softmax | 4096x4096 | v1 thread-per-row, 3 passes | 4.579 | 29.3 GB/s | 1.00x | 9.2% of peak BW | PASS |
| Softmax | 4096x4096 | v2 block-per-row, 3 passes | 0.746 | 180.0 GB/s | 6.14x | 56.3% of peak BW | PASS |
| Softmax | 4096x4096 | v3 online softmax, 2 passes | 0.751 | 178.8 GB/s | 6.10x | 55.9% of peak BW | PASS |
| Softmax | 4096x4096 | v4 register-cached, 1 pass (float4) | 0.573 | 234.1 GB/s | 7.99x | 73.2% of peak BW | PASS |
| Softmax | 4096x4096 | PyTorch torch.softmax | 0.715 | 187.8 GB/s | 6.41x | 58.7% of peak BW | baseline (timing only) |
| LayerNorm | 4096x4096 | v1 two-pass (3 reads) | 0.870 | 154.3 GB/s | 1.00x | 48.2% of peak BW | PASS |
| LayerNorm | 4096x4096 | v2 Welford single-pass stats (2 reads) | 0.816 | 164.5 GB/s | 1.07x | 51.4% of peak BW | PASS |
| LayerNorm | 4096x4096 | v3 register-cached (1 read, float4) | 0.568 | 236.1 GB/s | 1.53x | 73.8% of peak BW | PASS |
| LayerNorm | 4096x4096 | PyTorch F.layer_norm | 0.806 | 166.5 GB/s | 1.08x | 52.0% of peak BW | baseline (timing only) |
| Residual add + LayerNorm | 4096x4096 | unfused: add kernel + LayerNorm kernel | 1.370 | 195.9 GB/s | 1.00x | 61.2% of peak BW | PASS |
| Residual add + LayerNorm | 4096x4096 | fused: add + LayerNorm in one kernel | 1.127 | 238.2 GB/s | 1.22x | 74.4% of peak BW | PASS |
| Residual add + LayerNorm | 4096x4096 | PyTorch eager (x + res, F.layer_norm) | 1.623 | 165.3 GB/s | 0.84x | 51.7% of peak BW | baseline (timing only) |
| Attention (BH=16, d=64) | N=256 | unfused: cuBLAS QK^T + softmax + cuBLAS PV | 0.212 | 1,265.1 GFLOPS | 1.00x | 100.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=256 | fused: FlashAttention-style (online softmax, tiled K/V) | 0.236 | 1,138.0 GFLOPS | 0.90x | 89.9% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=1024 | unfused: cuBLAS QK^T + softmax + cuBLAS PV | 1.923 | 2,233.4 GFLOPS | 1.00x | 100.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=1024 | fused: FlashAttention-style (online softmax, tiled K/V) | 2.648 | 1,621.8 GFLOPS | 0.73x | 72.6% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=1024 | PyTorch F.scaled_dot_product_attention (FP32) | 1.657 | 2,592.8 GFLOPS | 1.16x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=1024 | PyTorch naive (matmul, softmax, matmul) | 2.345 | 1,831.5 GFLOPS | 0.82x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=2048 | unfused: cuBLAS QK^T + softmax + cuBLAS PV | 9.073 | 1,893.5 GFLOPS | 1.00x | 100.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=2048 | fused: FlashAttention-style (online softmax, tiled K/V) | 9.260 | 1,855.3 GFLOPS | 0.98x | 98.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=2048 | PyTorch F.scaled_dot_product_attention (FP32) | 6.356 | 2,702.7 GFLOPS | 1.43x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=2048 | PyTorch naive (matmul, softmax, matmul) | 10.438 | 1,645.8 GFLOPS | 0.87x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=4096 | unfused: cuBLAS QK^T + softmax + cuBLAS PV | 43.006 | 1,597.9 GFLOPS | 1.00x | 100.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=4096 | fused: FlashAttention-style (online softmax, tiled K/V) | 28.811 | 2,385.2 GFLOPS | 1.49x | 149.3% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=4096 | PyTorch F.scaled_dot_product_attention (FP32) | 25.637 | 2,680.4 GFLOPS | 1.68x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=4096 | PyTorch naive (matmul, softmax, matmul) | 45.831 | 1,499.4 GFLOPS | 0.94x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=8192 | unfused: cuBLAS QK^T + softmax + cuBLAS PV | 196.597 | 1,398.2 GFLOPS | 1.00x | 100.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=8192 | fused: FlashAttention-style (online softmax, tiled K/V) | 112.180 | 2,450.3 GFLOPS | 1.75x | 175.3% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=8192 | PyTorch F.scaled_dot_product_attention (FP32) | 100.558 | 2,733.5 GFLOPS | 1.96x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=8192 | PyTorch naive (matmul, softmax, matmul) | 197.297 | 1,393.2 GFLOPS | 1.00x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=16384 | unfused: cuBLAS QK^T + softmax + cuBLAS PV | - | - | - | - | OOM (needs 17.2 GB for S) |
| Attention (BH=16, d=64) | N=16384 | fused: FlashAttention-style (online softmax, tiled K/V) | 453.550 | 2,424.2 GFLOPS | - | - | PASS |
| Attention (BH=16, d=64) | N=16384 | PyTorch F.scaled_dot_product_attention (FP32) | 411.922 | 2,669.2 GFLOPS | - | - | baseline (timing only) |
