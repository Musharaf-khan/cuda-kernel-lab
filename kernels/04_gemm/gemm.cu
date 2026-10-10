// SGEMM: C = A x B (row-major, square n x n, FP32), five versions vs cuBLAS.
//
//   v1 naive, uncoalesced     one thread per output, warp walks down a column
//   v2 naive, coalesced       same math, threadIdx.x -> column
//   v3 shared-memory tiled    32x32 tiles of A and B staged in shared memory
//   v4 2D register tiling     each thread computes an 8x8 block of C in registers
//   v5 vectorized             v4 + float4 global loads/stores + transposed A tile
//
// Usage:  ./gemm [n] [profile]      n must be a multiple of 128 (default 2048)
//         ./gemm 256                small run, also checks cuBLAS against a CPU reference
//         ./gemm 2048 profile       run each kernel once (for Nsight Compute)
#include "../../common/common.h"
#include <cublas_v2.h>
#include <vector>

const int T = 32;   // tile size for v1-v3 / block is T x T = 1024 threads

// v1: naive, threadIdx.x -> ROW  (a warp walks down a column: uncoalesced)
__global__ void sgemm_naive_uncoalesced(int n, const float* A, const float* B, float* C) {
    int row = blockIdx.x * T + threadIdx.x;
    int col = blockIdx.y * T + threadIdx.y;
    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; k++) acc += A[row * n + k] * B[k * n + col];
        C[row * n + col] = acc;
    }
}

// v2: naive, threadIdx.x -> COL  (a warp reads consecutive B / writes consecutive C)
__global__ void sgemm_naive_coalesced(int n, const float* A, const float* B, float* C) {
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; k++) acc += A[row * n + k] * B[k * n + col];
        C[row * n + col] = acc;
    }
}

// v3: shared-memory tiling (assumes n is a multiple of 32)
__global__ void sgemm_tiled(int n, const float* A, const float* B, float* C) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty;
    int col = blockIdx.x * T + tx;

    float acc = 0.0f;
    for (int t = 0; t < n; t += T) {
        As[ty][tx] = A[row * n + (t + tx)];      // coalesced load of an A tile
        Bs[ty][tx] = B[(t + ty) * n + col];      // coalesced load of a B tile
        __syncthreads();                         // tiles fully loaded
        for (int k = 0; k < T; k++)
            acc += As[ty][k] * Bs[k][tx];        // 1 FMA per 2 shared-memory loads
        __syncthreads();                         // done reading before next load
    }
    C[row * n + col] = acc;
}

// v4: 2D register tiling. Block computes BM x BN of C; each thread computes TM x TN.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_2d_regtile(int n, const float* A, const float* B, float* C) {
    const int numThreads = (BM * BN) / (TM * TN);     // 256
    const int threadCol = threadIdx.x % (BN / TN);    // 0..15
    const int threadRow = threadIdx.x / (BN / TN);    // 0..15

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move pointers to this block's starting tile
    A += blockIdx.y * BM * n;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * n + blockIdx.x * BN;

    // Which element each thread loads into shared memory
    const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK, strideA = numThreads / BK;
    const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN, strideB = numThreads / BN;

    float threadResults[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int bk = 0; bk < n; bk += BK) {
        for (int off = 0; off < BM; off += strideA)
            As[(innerRowA + off) * BK + innerColA] = A[(innerRowA + off) * n + innerColA];
        for (int off = 0; off < BK; off += strideB)
            Bs[(innerRowB + off) * BN + innerColB] = B[(innerRowB + off) * n + innerColB];
        __syncthreads();
        A += BK;          // slide right along A's rows
        B += BK * n;      // slide down B's columns

        #pragma unroll
        for (int dot = 0; dot < BK; dot++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) regM[i] = As[(threadRow * TM + i) * BK + dot];
            #pragma unroll
            for (int i = 0; i < TN; i++) regN[i] = Bs[dot * BN + threadCol * TN + i];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    threadResults[m * TN + j] += regM[m] * regN[j];   // 64 FMAs from 16 loads
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; m++)
        for (int j = 0; j < TN; j++)
            C[(threadRow * TM + m) * n + threadCol * TN + j] = threadResults[m * TN + j];
}

// v5: v4 + vectorized memory access.
//  - Global loads/stores use float4 (one 128-bit instruction instead of four 32-bit ones).
//  - The A tile is stored TRANSPOSED in shared memory (As[k][m]), so in the inner loop each
//    thread reads its TM values of A contiguously - the compiler can then use 128-bit
//    shared-memory loads (LDS.128) for both A and B.
// Requires n % 4 == 0 (float4 alignment) and n % 128 == 0 (tiling).
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_vectorized(int n, const float* A, const float* B, float* C) {
    static_assert((BM * BN) / (TM * TN) * 4 == BM * BK, "one float4 of A per thread");
    static_assert((BM * BN) / (TM * TN) * 4 == BK * BN, "one float4 of B per thread");

    const int threadCol = threadIdx.x % (BN / TN);    // 0..15
    const int threadRow = threadIdx.x / (BN / TN);    // 0..15

    __shared__ float As[BK * BM];                     // transposed: As[k * BM + m]
    __shared__ float Bs[BK * BN];                     //             Bs[k * BN + n]

    A += blockIdx.y * BM * n;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * n + blockIdx.x * BN;

    // A tile: BM x BK = 128 x 8 floats = 256 float4 -> one per thread
    const int innerRowA = threadIdx.x / (BK / 4);     // 0..127
    const int innerColA = threadIdx.x % (BK / 4);     // 0..1   (which float4 in the row)
    // B tile: BK x BN = 8 x 128 floats = 256 float4 -> one per thread
    const int innerRowB = threadIdx.x / (BN / 4);     // 0..7
    const int innerColB = threadIdx.x % (BN / 4);     // 0..31

    float threadResults[TM * TN] = {0.0f};
    alignas(16) float regM[TM];
    alignas(16) float regN[TN];

    for (int bk = 0; bk < n; bk += BK) {
        // Load 4 consecutive floats of A with one instruction, scatter them transposed.
        float4 a = reinterpret_cast<const float4*>(&A[innerRowA * n + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = a.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = a.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = a.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = a.w;
        // B is already in the right layout: straight float4 copy.
        reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4*>(&B[innerRowB * n + innerColB * 4])[0];
        __syncthreads();
        A += BK;
        B += BK * n;

        #pragma unroll
        for (int dot = 0; dot < BK; dot++) {
            // TM (and TN) consecutive floats -> explicit 128-bit shared-memory loads.
            #pragma unroll
            for (int i = 0; i < TM; i += 4)
                *reinterpret_cast<float4*>(&regM[i]) =
                    *reinterpret_cast<const float4*>(&As[dot * BM + threadRow * TM + i]);
            #pragma unroll
            for (int i = 0; i < TN; i += 4)
                *reinterpret_cast<float4*>(&regN[i]) =
                    *reinterpret_cast<const float4*>(&Bs[dot * BN + threadCol * TN + i]);
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    threadResults[m * TN + j] += regM[m] * regN[j];
        }
        __syncthreads();
    }

    // Write C with float4 stores.
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            float4 r = make_float4(threadResults[m * TN + j + 0], threadResults[m * TN + j + 1],
                                   threadResults[m * TN + j + 2], threadResults[m * TN + j + 3]);
            reinterpret_cast<float4*>(&C[(threadRow * TM + m) * n + threadCol * TN + j])[0] = r;
        }
    }
}

// CPU reference in double precision (only used for small n, it is O(n^3)).
static void cpu_gemm(int n, const float* A, const float* B, float* C) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int k = 0; k < n; k++) acc += (double)A[i * n + k] * B[k * n + j];
            C[i * n + j] = (float)acc;
        }
}

int main(int argc, char** argv) {
    bool profile = profile_mode(argc, argv);
    int n = 2048;
    for (int i = 1; i < argc; i++) if (atoi(argv[i]) > 0) n = atoi(argv[i]);
    if (n % 128 != 0) { printf("n must be a multiple of 128 (got %d)\n", n); return 1; }

    size_t count = (size_t)n * n, bytes = count * sizeof(float);
    double flops = 2.0 * n * n * n;

    std::vector<float> hA(count), hB(count), hC(count), hRef(count);
    srand(42);
    for (size_t i = 0; i < count; i++) {
        hA[i] = rand() / (float)RAND_MAX * 2 - 1;
        hB[i] = rand() / (float)RAND_MAX * 2 - 1;
    }

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes)); CHECK(cudaMalloc(&dB, bytes)); CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    // cuBLAS is column-major. Row-major C = A*B  <=>  column-major C^T = B^T * A^T,
    // so we pass B first, then A.
    cublasHandle_t h; cublasCreate(&h);
    float alpha = 1.0f, beta = 0.0f;
    auto run_cublas = [&]() {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dC, n);
    };

    dim3 block(T, T), grid(n / T, n / T);
    std::function<void()> launches[] = {
        [&]() { sgemm_naive_uncoalesced<<<grid, block>>>(n, dA, dB, dC); },
        [&]() { sgemm_naive_coalesced<<<grid, block>>>(n, dA, dB, dC); },
        [&]() { sgemm_tiled<<<grid, block>>>(n, dA, dB, dC); },
        [&]() { sgemm_2d_regtile<128, 128, 8, 8, 8><<<dim3(n / 128, n / 128), 256>>>(n, dA, dB, dC); },
        [&]() { sgemm_vectorized<128, 128, 8, 8, 8><<<dim3(n / 128, n / 128), 256>>>(n, dA, dB, dC); },
    };
    const int NK = 5;
    const char* names[] = {"v1 naive uncoalesced", "v2 naive coalesced  ", "v3 smem tiled       ",
                           "v4 2D register tile ", "v5 vectorized float4"};
    const char* versions[] = {"v1 naive (uncoalesced)", "v2 naive (coalesced)", "v3 shared-mem tiled 32x32",
                              "v4 2D register tiling 8x8/thread", "v5 v4 + float4 + transposed A tile"};

    if (profile) {                               // each kernel once, for Nsight Compute
        for (int k = 0; k < NK; k++) { launches[k](); CHECK(cudaGetLastError()); }
        run_cublas();
        CHECK(cudaDeviceSynchronize());
        cublasDestroy(h);
        return 0;
    }

    // Reference result from cuBLAS (and, for small n, check cuBLAS itself against the CPU).
    run_cublas(); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hRef.data(), dC, bytes, cudaMemcpyDeviceToHost));
    if (n <= 512) {
        std::vector<float> hCpu(count);
        cpu_gemm(n, hA.data(), hB.data(), hCpu.data());
        float e = max_rel_err(hRef.data(), hCpu.data(), count);
        printf("cuBLAS vs CPU (double) reference: max rel err %.1e %s\n", e, e < 1e-3 ? "PASS" : "FAIL");
    }

    // GPU clocks drift with temperature/power, so time cuBLAS before AND after our kernels
    // and compare against the average. Ratios from the same run are what we report.
    int iters = n >= 4096 ? 5 : 10;
    float cublas_ms_before = time_ms(run_cublas, 20);

    printf("SGEMM %dx%dx%d\n", n, n, n);
    float ms_k[NK], err_k[NK];
    for (int k = 0; k < NK; k++) {
        CHECK(cudaMemset(dC, 0, bytes));
        ms_k[k] = time_ms(launches[k], k == 0 ? (n >= 4096 ? 2 : 5) : iters);
        CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        err_k[k] = max_rel_err(hC.data(), hRef.data(), count);
    }

    float cublas_ms_after = time_ms(run_cublas, 20);
    float cublas_ms = 0.5f * (cublas_ms_before + cublas_ms_after);
    double cublas_gf = flops / (cublas_ms / 1e3) / 1e9;

    char size[32]; snprintf(size, sizeof size, "%d^3", n);
    for (int k = 0; k < NK; k++) {
        double gf = flops / (ms_k[k] / 1e3) / 1e9;
        bool ok = err_k[k] < 1e-3f;
        printf("%s | %8.2f ms | %7.1f GFLOPS | %4.1f%% of peak | %5.1f%% of cuBLAS | err %.1e %s\n",
               names[k], ms_k[k], gf, gf / T4_PEAK_GFLOPS * 100, gf / cublas_gf * 100, err_k[k],
               ok ? "PASS" : "FAIL");
        report("SGEMM", versions[k], size, ms_k[k], gf, "GFLOPS", gf / cublas_gf * 100, "of cuBLAS", ok);
    }
    printf("cuBLAS               | %8.2f ms | %7.1f GFLOPS | %4.1f%% of peak | (before %.2f ms, after %.2f ms)\n",
           cublas_ms, cublas_gf, cublas_gf / T4_PEAK_GFLOPS * 100, cublas_ms_before, cublas_ms_after);
    report("SGEMM", "cuBLAS (cublasSgemm)", size, cublas_ms, cublas_gf, "GFLOPS", 100.0, "of cuBLAS", true);

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
