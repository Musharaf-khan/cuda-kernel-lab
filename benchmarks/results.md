| Kernel | Size | Version | Time (ms) | Throughput | Speedup | % of | Check |
|---|---|---|---:|---:|---:|---:|---|
| Vector add | 16M floats | naive | 0.764 | 263.5 GB/s | 1.00x | 82.3% of peak BW | PASS |
| Reduction (sum) | 16M floats | v1 atomicAdd per element | 35.502 | 1.9 GB/s | 1.00x | 0.6% of peak BW | PASS |
| Reduction (sum) | 16M floats | v2 shared-mem tree | 0.624 | 107.6 GB/s | 56.94x | 33.6% of peak BW | PASS |
| Reduction (sum) | 16M floats | v3 grid-stride + warp shuffle | 0.270 | 248.3 GB/s | 131.39x | 77.6% of peak BW | PASS |
| Transpose | 4096x4096 | copy (ceiling) | 0.571 | 235.1 GB/s | 2.67x | 73.5% of peak BW | PASS |
| Transpose | 4096x4096 | naive (uncoalesced writes) | 1.524 | 88.1 GB/s | 1.00x | 27.5% of peak BW | PASS |
| Transpose | 4096x4096 | smem tiled [32][32] (bank conflicts) | 0.732 | 183.3 GB/s | 2.08x | 57.3% of peak BW | PASS |
| Transpose | 4096x4096 | smem tiled [32][33] (padded) | 0.689 | 194.8 GB/s | 2.21x | 60.9% of peak BW | PASS |
| SGEMM | 256^3 | v1 naive (uncoalesced) | 0.691 | 48.5 GFLOPS | 1.00x | 2.8% of cuBLAS | PASS (median of 3) |
| SGEMM | 256^3 | v2 naive (coalesced) | 0.049 | 678.9 GFLOPS | 13.99x | 39.3% of cuBLAS | PASS (median of 3) |
| SGEMM | 256^3 | v3 shared-mem tiled 32x32 | 0.040 | 844.9 GFLOPS | 17.41x | 48.9% of cuBLAS | PASS (median of 3) |
| SGEMM | 256^3 | v4 2D register tiling 8x8/thread | 0.074 | 455.1 GFLOPS | 9.38x | 26.3% of cuBLAS | PASS (median of 3) |
| SGEMM | 256^3 | v5 v4 + float4 + transposed A tile | 0.062 | 540.6 GFLOPS | 11.13x | 31.3% of cuBLAS | PASS (median of 3) |
| SGEMM | 256^3 | v6 autotuned (t64x64_4x8) | 0.021 | 1,631.5 GFLOPS | 33.56x | 94.1% of cuBLAS | PASS (median of 3) |
| SGEMM | 256^3 | cuBLAS (cublasSgemm) | 0.019 | 1,717.6 GFLOPS | 35.45x | 100.0% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | v1 naive (uncoalesced) | 2224.913 | 61.8 GFLOPS | 1.00x | 1.5% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | v2 naive (coalesced) | 244.126 | 563.0 GFLOPS | 9.11x | 14.1% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | v3 shared-mem tiled 32x32 | 157.030 | 875.2 GFLOPS | 14.17x | 21.8% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | v4 2D register tiling 8x8/thread | 42.748 | 3,215.1 GFLOPS | 52.05x | 81.8% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | v5 v4 + float4 + transposed A tile | 39.017 | 3,522.6 GFLOPS | 57.02x | 88.4% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | v6 autotuned (t128x128_8x8) | 33.972 | 4,045.6 GFLOPS | 65.49x | 101.3% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | cuBLAS (cublasSgemm) | 34.364 | 3,999.5 GFLOPS | 64.75x | 100.0% of cuBLAS | PASS (median of 3) |
| SGEMM | 4096^3 | PyTorch torch.mm (cuBLAS) | 36.784 | 3,736.4 GFLOPS | 60.49x | - | baseline (timing only) |
| SGEMM attempts | 4096^3 | A0 v5 re-expressed in the lab kernel (baseline) | 40.058 | 3,431.0 GFLOPS | 1.00x | 90.1% of cuBLAS | PASS (median of 3) |
| SGEMM attempts | 4096^3 | B1 v5 + warp tiling (64x32 warp tile, 2 sub-tiles) | 36.298 | 3,786.4 GFLOPS | 1.10x | 99.4% of cuBLAS | PASS (median of 3) |
| SGEMM attempts | 4096^3 | B2 v5 + register-prefetch double buffering | 41.089 | 3,344.9 GFLOPS | 0.97x | 88.0% of cuBLAS | PASS (median of 3) |
| SGEMM attempts | 4096^3 | B3a v5 with 128 threads (8x16 per thread) | 39.113 | 3,513.9 GFLOPS | 1.02x | 92.3% of cuBLAS | PASS (median of 3) |
| SGEMM attempts | 4096^3 | B3b v5 with 512 threads (4x8 per thread) | 44.268 | 3,104.7 GFLOPS | 0.90x | 81.5% of cuBLAS | PASS (median of 3) |
| SGEMM attempts | 4096^3 | B3c v5 + __launch_bounds__(256, 2) | 40.633 | 3,382.4 GFLOPS | 0.99x | 88.8% of cuBLAS | PASS (median of 3) |
| SGEMM attempts | 4096^3 | B4 v5 + padded transposed A tile (stride 132) | 40.141 | 3,423.9 GFLOPS | 1.00x | 91.1% of cuBLAS | PASS (median of 3) |
| SGEMM attempts | 4096^3 | B5 v5 + BK=16 (half the barriers per FLOP) | 40.898 | 3,360.5 GFLOPS | 0.98x | 88.2% of cuBLAS | PASS (median of 3) |
| SGEMM attempts | 4096^3 | C2 warp tiling + double buffering | 37.057 | 3,708.8 GFLOPS | 1.08x | 97.6% of cuBLAS | PASS (median of 3) |
| SGEMM attempts | 4096^3 | C3 C2 + padded A tile | 35.971 | 3,820.8 GFLOPS | 1.11x | 100.2% of cuBLAS | PASS (median of 3) |
| SGEMM attempts | 4096^3 | C4 C3 + BK=16 | 37.854 | 3,630.8 GFLOPS | 1.06x | 94.4% of cuBLAS | PASS (median of 3) |
| SGEMM attempts | 4096^3 | C4b C4 without padding (32 KB smem fits 2 blocks/SM) | 38.313 | 3,587.3 GFLOPS | 1.05x | 94.2% of cuBLAS | PASS (median of 3) |
| SGEMM attempts | 4096^3 | C5 C4 + __launch_bounds__(256, 2) | 41.824 | 3,286.1 GFLOPS | 0.96x | 86.3% of cuBLAS | PASS (median of 3) |
| SGEMM attempts | 4096^3 | cuBLAS (same run) | 36.089 | 3,808.3 GFLOPS | 1.11x | 100.0% of cuBLAS | PASS (median of 3) |
| Softmax | 4096x4096 | v1 thread-per-row, 3 passes | 3.664 | 36.6 GB/s | 1.00x | 11.4% of peak BW | PASS |
| Softmax | 4096x4096 | v2 block-per-row, 3 passes | 0.697 | 192.5 GB/s | 5.25x | 60.1% of peak BW | PASS |
| Softmax | 4096x4096 | v3 online softmax, 2 passes | 0.760 | 176.6 GB/s | 4.82x | 55.2% of peak BW | PASS |
| Softmax | 4096x4096 | v4 register-cached, 1 pass (float4) | 0.584 | 229.7 GB/s | 6.27x | 71.8% of peak BW | PASS |
| Softmax | 4096x4096 | PyTorch torch.softmax | 0.704 | 190.5 GB/s | 5.20x | 59.5% of peak BW | baseline (timing only) |
| Softmax | 4096x4096 | Triton (row in registers) | 0.565 | 237.7 GB/s | 6.49x | 74.3% of peak BW | PASS |
| Softmax | 4096x4096 | PyTorch torch.softmax (same process as Triton) | 0.909 | 147.7 GB/s | 4.03x | - | baseline (timing only) |
| LayerNorm | 4096x4096 | v1 two-pass (3 reads) | 0.927 | 144.8 GB/s | 1.00x | 45.2% of peak BW | PASS |
| LayerNorm | 4096x4096 | v2 Welford single-pass stats (2 reads) | 0.870 | 154.3 GB/s | 1.07x | 48.2% of peak BW | PASS |
| LayerNorm | 4096x4096 | v3 register-cached (1 read, float4) | 0.568 | 236.3 GB/s | 1.63x | 73.9% of peak BW | PASS |
| LayerNorm | 4096x4096 | PyTorch F.layer_norm | 0.805 | 166.8 GB/s | 1.15x | 52.1% of peak BW | baseline (timing only) |
| LayerNorm | 4096x4096 | Triton (row in registers) | 0.569 | 235.7 GB/s | 1.63x | 73.7% of peak BW | PASS |
| LayerNorm | 4096x4096 | PyTorch F.layer_norm (same process as Triton) | 0.810 | 165.8 GB/s | 1.14x | - | baseline (timing only) |
| Residual add + LayerNorm | 4096x4096 | unfused: add kernel + LayerNorm kernel | 1.372 | 195.6 GB/s | 1.00x | 61.1% of peak BW | PASS |
| Residual add + LayerNorm | 4096x4096 | fused: add + LayerNorm in one kernel | 1.126 | 238.5 GB/s | 1.22x | 74.5% of peak BW | PASS |
| Residual add + LayerNorm | 4096x4096 | PyTorch eager (x + res, F.layer_norm) | 1.625 | 165.2 GB/s | 0.84x | 51.6% of peak BW | baseline (timing only) |
| Residual add + LayerNorm | 4096x4096 | Triton fused add + LayerNorm | 1.151 | 233.1 GB/s | 1.19x | 72.8% of peak BW | PASS |
| Attention (BH=16, d=64) | N=256 | unfused: cuBLAS QK^T + softmax + cuBLAS PV | 0.227 | 1,180.8 GFLOPS | 1.00x | 100.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=256 | fused: FlashAttention-style (online softmax, tiled K/V) | 0.259 | 1,035.6 GFLOPS | 0.88x | 87.7% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=256 | fused v2: 2 threads/row, 64 rows, tile 32 (128 thr) | 0.172 | 1,561.8 GFLOPS | 1.32x | 132.3% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=256 | fused v2: 4 threads/row, 64 rows, tile 32 (256 thr) | 0.322 | 832.6 GFLOPS | 0.71x | 70.5% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=256 | fused v2: 4 threads/row, 64 rows, tile 64 (256 thr) | 0.332 | 809.1 GFLOPS | 0.69x | 68.5% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=256 | fused v2: 2 threads/row, 128 rows, tile 32 (256 thr) | 0.168 | 1,601.4 GFLOPS | 1.36x | 135.6% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=1024 | unfused: cuBLAS QK^T + softmax + cuBLAS PV | 2.031 | 2,114.7 GFLOPS | 1.00x | 100.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=1024 | fused: FlashAttention-style (online softmax, tiled K/V) | 2.665 | 1,611.5 GFLOPS | 0.76x | 76.2% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=1024 | fused v2: 2 threads/row, 64 rows, tile 32 (128 thr) | 2.207 | 1,946.5 GFLOPS | 0.92x | 92.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=1024 | fused v2: 4 threads/row, 64 rows, tile 32 (256 thr) | 4.317 | 994.9 GFLOPS | 0.47x | 47.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=1024 | fused v2: 4 threads/row, 64 rows, tile 64 (256 thr) | 3.096 | 1,387.1 GFLOPS | 0.66x | 65.6% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=1024 | fused v2: 2 threads/row, 128 rows, tile 32 (256 thr) | 1.647 | 2,608.1 GFLOPS | 1.23x | 123.3% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=1024 | PyTorch F.scaled_dot_product_attention (FP32) | 1.507 | 2,850.2 GFLOPS | 1.35x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=1024 | PyTorch naive (matmul, softmax, matmul) | 2.289 | 1,876.0 GFLOPS | 0.89x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=1024 | Triton FlashAttention-style (best of 4 block shapes: 64x32) | 4.388 | 978.9 GFLOPS | 0.46x | - | PASS |
| Attention (BH=16, d=64) | N=1024 | PyTorch SDPA (same process as Triton) | 2.523 | 1,702.2 GFLOPS | 0.80x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=2048 | unfused: cuBLAS QK^T + softmax + cuBLAS PV | 7.584 | 2,265.2 GFLOPS | 1.00x | 100.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=2048 | fused: FlashAttention-style (online softmax, tiled K/V) | 7.950 | 2,161.0 GFLOPS | 0.95x | 95.4% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=2048 | fused v2: 2 threads/row, 64 rows, tile 32 (128 thr) | 5.829 | 2,947.1 GFLOPS | 1.30x | 130.1% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=2048 | fused v2: 4 threads/row, 64 rows, tile 32 (256 thr) | 10.983 | 1,564.3 GFLOPS | 0.69x | 69.1% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=2048 | fused v2: 4 threads/row, 64 rows, tile 64 (256 thr) | 11.077 | 1,550.9 GFLOPS | 0.68x | 68.5% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=2048 | fused v2: 2 threads/row, 128 rows, tile 32 (256 thr) | 6.179 | 2,780.3 GFLOPS | 1.23x | 122.7% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=2048 | PyTorch F.scaled_dot_product_attention (FP32) | 5.772 | 2,976.6 GFLOPS | 1.31x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=2048 | PyTorch naive (matmul, softmax, matmul) | 9.659 | 1,778.6 GFLOPS | 0.79x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=2048 | Triton FlashAttention-style (best of 4 block shapes: 64x64) | 9.244 | 1,858.6 GFLOPS | 0.82x | - | PASS |
| Attention (BH=16, d=64) | N=2048 | PyTorch SDPA (same process as Triton) | 5.464 | 3,144.0 GFLOPS | 1.39x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=4096 | unfused: cuBLAS QK^T + softmax + cuBLAS PV | 36.288 | 1,893.7 GFLOPS | 1.00x | 100.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=4096 | fused: FlashAttention-style (online softmax, tiled K/V) | 26.349 | 2,608.1 GFLOPS | 1.38x | 137.7% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=4096 | fused v2: 2 threads/row, 64 rows, tile 32 (128 thr) | 25.305 | 2,715.6 GFLOPS | 1.43x | 143.4% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=4096 | fused v2: 4 threads/row, 64 rows, tile 32 (256 thr) | 43.799 | 1,569.0 GFLOPS | 0.83x | 82.9% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=4096 | fused v2: 4 threads/row, 64 rows, tile 64 (256 thr) | 44.914 | 1,530.0 GFLOPS | 0.81x | 80.8% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=4096 | fused v2: 2 threads/row, 128 rows, tile 32 (256 thr) | 25.006 | 2,748.1 GFLOPS | 1.45x | 145.1% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=4096 | PyTorch F.scaled_dot_product_attention (FP32) | 23.965 | 2,867.4 GFLOPS | 1.51x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=4096 | PyTorch naive (matmul, softmax, matmul) | 41.261 | 1,665.5 GFLOPS | 0.88x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=4096 | Triton FlashAttention-style (best of 4 block shapes: 64x64) | 37.088 | 1,852.9 GFLOPS | 0.98x | - | PASS |
| Attention (BH=16, d=64) | N=4096 | PyTorch SDPA (same process as Triton) | 24.064 | 2,855.7 GFLOPS | 1.51x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=8192 | unfused: cuBLAS QK^T + softmax + cuBLAS PV | 169.464 | 1,622.0 GFLOPS | 1.00x | 100.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=8192 | fused: FlashAttention-style (online softmax, tiled K/V) | 101.002 | 2,721.5 GFLOPS | 1.68x | 167.8% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=8192 | fused v2: 2 threads/row, 64 rows, tile 32 (128 thr) | 101.448 | 2,709.5 GFLOPS | 1.67x | 167.0% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=8192 | fused v2: 4 threads/row, 64 rows, tile 32 (256 thr) | 174.859 | 1,572.0 GFLOPS | 0.97x | 96.9% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=8192 | fused v2: 4 threads/row, 64 rows, tile 64 (256 thr) | 179.778 | 1,529.0 GFLOPS | 0.94x | 94.3% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=8192 | fused v2: 2 threads/row, 128 rows, tile 32 (256 thr) | 100.086 | 2,746.4 GFLOPS | 1.69x | 169.3% of unfused speed | PASS |
| Attention (BH=16, d=64) | N=8192 | PyTorch F.scaled_dot_product_attention (FP32) | 94.570 | 2,906.6 GFLOPS | 1.79x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=8192 | PyTorch naive (matmul, softmax, matmul) | 184.300 | 1,491.5 GFLOPS | 0.92x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=8192 | Triton FlashAttention-style (best of 4 block shapes: 64x64) | 146.706 | 1,873.7 GFLOPS | 1.16x | - | PASS |
| Attention (BH=16, d=64) | N=8192 | PyTorch SDPA (same process as Triton) | 96.238 | 2,856.2 GFLOPS | 1.76x | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=16384 | unfused: cuBLAS QK^T + softmax + cuBLAS PV | - | - | - | - | OOM (needs 17.2 GB for S) |
| Attention (BH=16, d=64) | N=16384 | fused: FlashAttention-style (online softmax, tiled K/V) | 417.135 | 2,635.9 GFLOPS | - | - | PASS |
| Attention (BH=16, d=64) | N=16384 | fused v2: 2 threads/row, 64 rows, tile 32 (128 thr) | 407.177 | 2,700.3 GFLOPS | - | - | PASS |
| Attention (BH=16, d=64) | N=16384 | fused v2: 4 threads/row, 64 rows, tile 32 (256 thr) | 698.513 | 1,574.1 GFLOPS | - | - | PASS |
| Attention (BH=16, d=64) | N=16384 | fused v2: 4 threads/row, 64 rows, tile 64 (256 thr) | 715.152 | 1,537.5 GFLOPS | - | - | PASS |
| Attention (BH=16, d=64) | N=16384 | fused v2: 2 threads/row, 128 rows, tile 32 (256 thr) | 406.694 | 2,703.5 GFLOPS | - | - | PASS |
| Attention (BH=16, d=64) | N=16384 | PyTorch F.scaled_dot_product_attention (FP32) | 385.086 | 2,855.2 GFLOPS | - | - | baseline (timing only) |
| Attention (BH=16, d=64) | N=16384 | Triton FlashAttention-style (best of 4 block shapes: 64x64) | 591.747 | 1,858.1 GFLOPS | - | - | PASS |
| Attention (BH=16, d=64) | N=16384 | PyTorch SDPA (same process as Triton) | 401.744 | 2,736.8 GFLOPS | - | - | baseline (timing only) |
| HGEMM FP16 Tensor Cores | 256^3 | v1 WMMA naive (warp per 16x16 tile, global loads) | 0.013 | 2,548.2 GFLOPS | 1.00x | 60.6% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 256^3 | v2 WMMA tiled (128x128, smem, double-buffered) | 0.013 | 2,624.6 GFLOPS | 1.03x | 62.4% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 256^3 | cublasGemmEx (FP16 in, FP32 accumulate) | 0.008 | 4,204.4 GFLOPS | 1.65x | 100.0% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 256^3 | cublasHgemm (FP16 in/out) | 0.007 | 4,869.9 GFLOPS | 1.91x | 115.8% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 256^3 | cublasSgemm FP32 (no Tensor Cores, for scale) | 0.020 | 1,660.7 GFLOPS | 0.65x | 39.5% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 1024^3 | v1 WMMA naive (warp per 16x16 tile, global loads) | 0.593 | 3,621.0 GFLOPS | 1.00x | 10.7% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 1024^3 | v2 WMMA tiled (128x128, smem, double-buffered) | 0.093 | 22,974.9 GFLOPS | 6.34x | 68.1% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 1024^3 | cublasGemmEx (FP16 in, FP32 accumulate) | 0.064 | 33,741.7 GFLOPS | 9.33x | 100.0% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 1024^3 | cublasHgemm (FP16 in/out) | 0.062 | 34,587.6 GFLOPS | 9.55x | 102.5% of cublasGemmEx | FAIL |
| HGEMM FP16 Tensor Cores | 1024^3 | cublasSgemm FP32 (no Tensor Cores, for scale) | 0.368 | 5,836.2 GFLOPS | 1.61x | 17.3% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 2048^3 | v1 WMMA naive (warp per 16x16 tile, global loads) | 6.631 | 2,591.0 GFLOPS | 1.00x | 9.2% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 2048^3 | v2 WMMA tiled (128x128, smem, double-buffered) | 0.694 | 24,762.7 GFLOPS | 9.56x | 87.5% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 2048^3 | cublasGemmEx (FP16 in, FP32 accumulate) | 0.607 | 28,306.4 GFLOPS | 10.93x | 100.0% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 2048^3 | cublasHgemm (FP16 in/out) | 0.583 | 29,450.4 GFLOPS | 11.37x | 104.0% of cublasGemmEx | FAIL |
| HGEMM FP16 Tensor Cores | 2048^3 | cublasSgemm FP32 (no Tensor Cores, for scale) | 3.727 | 4,610.0 GFLOPS | 1.78x | 16.3% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 4096^3 | v1 WMMA naive (warp per 16x16 tile, global loads) | 65.645 | 2,093.7 GFLOPS | 1.00x | 9.2% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 4096^3 | v2 WMMA tiled (128x128, smem, double-buffered) | 8.818 | 15,586.6 GFLOPS | 7.44x | 68.8% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 4096^3 | cublasGemmEx (FP16 in, FP32 accumulate) | 6.065 | 22,662.2 GFLOPS | 10.82x | 100.0% of cublasGemmEx | PASS |
| HGEMM FP16 Tensor Cores | 4096^3 | cublasHgemm (FP16 in/out) | 4.412 | 31,154.1 GFLOPS | 14.88x | 137.5% of cublasGemmEx | FAIL |
| HGEMM FP16 Tensor Cores | 4096^3 | cublasSgemm FP32 (no Tensor Cores, for scale) | 34.086 | 4,032.1 GFLOPS | 1.93x | 17.8% of cublasGemmEx | PASS |
