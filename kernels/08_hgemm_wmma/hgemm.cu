// FP16 GEMM on Tensor Cores with WMMA (nvcuda::wmma), FP32 accumulate:  C(fp32) = A(fp16) x B(fp16)
//
//   v1 WMMA naive    one warp per 16x16 output tile, fragments loaded straight from global memory
//   v2 WMMA tiled    128x128 block tile, BK = 32, 8 warps each owning a 64x32 warp tile (4x2 fragments),
//                    A/B staged in padded shared memory, register-prefetch double buffering
//
// Baselines (same run, same inputs):
//   cublasGemmEx    FP16 inputs, FP32 accumulate + FP32 output  <- apples-to-apples with v1/v2
//   cublasHgemm     FP16 inputs, FP16 output (cuBLAS may accumulate in FP16; reported, not the target)
//   cublasSgemm     FP32 SIMT, for "what do Tensor Cores buy on a T4"
//
// Usage: ./hgemm [n] [profile]        n multiple of 128 (default: 1024, 2048, 4096)
#include "../../common/common.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <string>
#include <vector>

using namespace nvcuda;

// ---------------------------------------------------------------- v1: naive WMMA
// Block = 4 warps stacked along M; each warp owns one 16x16 tile of C and walks K in steps of 16.
__global__ void hgemm_wmma_naive(int n, const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C) {
    const int warp = threadIdx.x / 32;
    const int tm = blockIdx.y * 4 + warp, tn = blockIdx.x;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int k = 0; k < n; k += 16) {
        wmma::load_matrix_sync(a, A + (size_t)tm * 16 * n + k, n);
        wmma::load_matrix_sync(b, B + (size_t)k * n + tn * 16, n);
        wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(C + (size_t)tm * 16 * n + tn * 16, acc, n, wmma::mem_row_major);
}

// ---------------------------------------------------------------- v2: tiled WMMA
constexpr int HBM = 128, HBN = 128, HBK = 32;   // block tile
constexpr int HWM = 64, HWN = 32;               // warp tile -> 2 x 4 = 8 warps
constexpr int HNT = (HBM / HWM) * (HBN / HWN) * 32;
constexpr int PAD = 8;                          // halves (16 bytes): breaks the bank pattern of ldmatrix-style loads
constexpr int LDA_S = HBK + PAD;                // As: [HBM][HBK + PAD], k contiguous
constexpr int LDB_S = HBN + PAD;                // Bs: [HBK][HBN + PAD], n contiguous

__global__ void __launch_bounds__(HNT)
hgemm_wmma_tiled(int n, const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C) {
    __shared__ __align__(32) half As[2][HBM * LDA_S];
    __shared__ __align__(32) half Bs[2][HBK * LDB_S];

    const int tid = threadIdx.x, warp = tid / 32;
    const int warpM = warp / (HBN / HWN), warpN = warp % (HBN / HWN);
    A += (size_t)blockIdx.y * HBM * n;
    B += (size_t)blockIdx.x * HBN;

    // One tile = 128*32 halves = 512 x 16-byte chunks -> 2 per thread, for A and for B.
    constexpr int CH = HBM * HBK / 8 / HNT;
    uint4 ra[CH], rb[CH];
    auto load_global = [&](int k0) {
        #pragma unroll
        for (int i = 0; i < CH; i++) {
            const int f = tid + i * HNT;
            const int ar = f / (HBK / 8), ac = f % (HBK / 8);           // A: 4 chunks per row
            ra[i] = *reinterpret_cast<const uint4*>(A + (size_t)ar * n + k0 + ac * 8);
            const int br = f / (HBN / 8), bc = f % (HBN / 8);           // B: 16 chunks per row
            rb[i] = *reinterpret_cast<const uint4*>(B + (size_t)(k0 + br) * n + bc * 8);
        }
    };
    auto store_shared = [&](int s) {
        #pragma unroll
        for (int i = 0; i < CH; i++) {
            const int f = tid + i * HNT;
            const int ar = f / (HBK / 8), ac = f % (HBK / 8);
            *reinterpret_cast<uint4*>(&As[s][ar * LDA_S + ac * 8]) = ra[i];
            const int br = f / (HBN / 8), bc = f % (HBN / 8);
            *reinterpret_cast<uint4*>(&Bs[s][br * LDB_S + bc * 8]) = rb[i];
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[HWM / 16][HWN / 16];
    #pragma unroll
    for (int i = 0; i < HWM / 16; i++)
        #pragma unroll
        for (int j = 0; j < HWN / 16; j++) wmma::fill_fragment(acc[i][j], 0.0f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[HWM / 16];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[HWN / 16];

    load_global(0);
    store_shared(0);
    __syncthreads();
    int s = 0;
    for (int k0 = 0; k0 < n; k0 += HBK) {
        const bool more = k0 + HBK < n;
        if (more) load_global(k0 + HBK);        // next tile's global loads overlap this tile's math
        #pragma unroll
        for (int kk = 0; kk < HBK; kk += 16) {
            #pragma unroll
            for (int i = 0; i < HWM / 16; i++)
                wmma::load_matrix_sync(a[i], &As[s][(warpM * HWM + i * 16) * LDA_S + kk], LDA_S);
            #pragma unroll
            for (int j = 0; j < HWN / 16; j++)
                wmma::load_matrix_sync(b[j], &Bs[s][kk * LDB_S + warpN * HWN + j * 16], LDB_S);
            #pragma unroll
            for (int i = 0; i < HWM / 16; i++)
                #pragma unroll
                for (int j = 0; j < HWN / 16; j++) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        if (more) store_shared(s ^ 1);
        __syncthreads();
        s ^= 1;
    }

    float* Cb = C + (size_t)(blockIdx.y * HBM + warpM * HWM) * n + blockIdx.x * HBN + warpN * HWN;
    #pragma unroll
    for (int i = 0; i < HWM / 16; i++)
        #pragma unroll
        for (int j = 0; j < HWN / 16; j++)
            wmma::store_matrix_sync(Cb + (size_t)i * 16 * n + j * 16, acc[i][j], n, wmma::mem_row_major);
}

// ---------------------------------------------------------------- host
static void cpu_gemm(int n, const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double acc = 0;
            for (int k = 0; k < n; k++) acc += (double)A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = (float)acc;
        }
}

static void run_size(int n, bool profile, cublasHandle_t h) {
    size_t count = (size_t)n * n;
    double flops = 2.0 * n * n * n;
    std::vector<half> hA(count), hB(count);
    std::vector<float> fA(count), fB(count), hC(count), hRef(count);
    srand(99 + n);
    for (size_t i = 0; i < count; i++) {
        hA[i] = __float2half(rand() / (float)RAND_MAX * 2 - 1);
        hB[i] = __float2half(rand() / (float)RAND_MAX * 2 - 1);
        fA[i] = __half2float(hA[i]);                 // the exact values the GPU sees
        fB[i] = __half2float(hB[i]);
    }
    half *dA, *dB, *dC16;
    float *dC, *dAf, *dBf;
    CHECK(cudaMalloc(&dA, count * 2)); CHECK(cudaMalloc(&dB, count * 2)); CHECK(cudaMalloc(&dC16, count * 2));
    CHECK(cudaMalloc(&dC, count * 4)); CHECK(cudaMalloc(&dAf, count * 4)); CHECK(cudaMalloc(&dBf, count * 4));
    CHECK(cudaMemcpy(dA, hA.data(), count * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB.data(), count * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dAf, fA.data(), count * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dBf, fB.data(), count * 4, cudaMemcpyHostToDevice));

    // Row-major C = A*B is column-major C^T = B^T A^T: pass B first.
    const float one = 1.0f, zero = 0.0f;
    const half hone = __float2half(1.0f), hzero = __float2half(0.0f);
    auto gemmex = [&]() {
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, dB, CUDA_R_16F, n, dA, CUDA_R_16F, n,
                     &zero, dC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };
    auto hgemm = [&]() { cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &hone, dB, n, dA, n, &hzero, dC16, n); };
    auto sgemm = [&]() { cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, dBf, n, dAf, n, &zero, dC, n); };
    auto v1 = [&]() { hgemm_wmma_naive<<<dim3(n / 16, n / 64), 128>>>(n, dA, dB, dC); };
    auto v2 = [&]() { hgemm_wmma_tiled<<<dim3(n / HBN, n / HBM), HNT>>>(n, dA, dB, dC); };

    if (profile) {
        v1(); v2(); gemmex(); hgemm();
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    } else {
        gemmex(); CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hRef.data(), dC, count * 4, cudaMemcpyDeviceToHost));
        if (n <= 256) {
            std::vector<float> cpu(count);
            cpu_gemm(n, fA, fB, cpu);
            float e = max_rel_err(hRef.data(), cpu.data(), count);
            printf("cublasGemmEx vs CPU (double, same fp16 inputs): max rel err %.1e %s\n", e, e < 1e-3 ? "PASS" : "FAIL");
        }
        int iters = n >= 4096 ? 20 : 50;
        float ex_before = time_ms(gemmex, iters);
        struct K { const char* name; std::function<void()> f; };
        std::vector<K> ks = {{"v1 WMMA naive (warp per 16x16 tile, global loads)", v1},
                             {"v2 WMMA tiled (128x128, smem, double-buffered)", v2}};
        std::vector<float> ms(ks.size()), err(ks.size());
        for (size_t k = 0; k < ks.size(); k++) {
            CHECK(cudaMemset(dC, 0, count * 4));
            ms[k] = time_ms(ks[k].f, iters);
            CHECK(cudaMemcpy(hC.data(), dC, count * 4, cudaMemcpyDeviceToHost));
            err[k] = max_rel_err(hC.data(), hRef.data(), count);
        }
        float hg_ms = time_ms(hgemm, iters);
        std::vector<half> h16(count);
        CHECK(cudaMemcpy(h16.data(), dC16, count * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; i++) hC[i] = __half2float(h16[i]);
        float hg_err = max_rel_err(hC.data(), hRef.data(), count);
        float sg_ms = time_ms(sgemm, n >= 4096 ? 5 : 20);
        float ex_after = time_ms(gemmex, iters);
        float ex_ms = 0.5f * (ex_before + ex_after);
        double ex_gf = flops / (ex_ms / 1e3) / 1e9;

        char size[32]; snprintf(size, sizeof size, "%d^3", n);
        printf("HGEMM %d^3 (FP16 in, FP32 accumulate)   cublasGemmEx %.3f ms = %.1f TFLOPS (before %.3f, after %.3f)\n",
               n, ex_ms, ex_gf / 1e3, ex_before, ex_after);
        for (size_t k = 0; k < ks.size(); k++) {
            double gf = flops / (ms[k] / 1e3) / 1e9;
            bool ok = err[k] < 1e-3f;
            printf("%-50s | %8.3f ms | %6.2f TFLOPS | %5.1f%% of cublasGemmEx | err %.1e %s\n",
                   ks[k].name, ms[k], gf / 1e3, gf / ex_gf * 100, err[k], ok ? "PASS" : "FAIL");
            report("HGEMM FP16 Tensor Cores", ks[k].name, size, ms[k], gf, "GFLOPS", gf / ex_gf * 100,
                   "of cublasGemmEx", ok);
        }
        report("HGEMM FP16 Tensor Cores", "cublasGemmEx (FP16 in, FP32 accumulate)", size, ex_ms, ex_gf, "GFLOPS",
               100.0, "of cublasGemmEx", true);
        double hg_gf = flops / (hg_ms / 1e3) / 1e9;
        printf("%-50s | %8.3f ms | %6.2f TFLOPS | %5.1f%% of cublasGemmEx | err %.1e (FP16 output)\n",
               "cublasHgemm (FP16 output)", hg_ms, hg_gf / 1e3, hg_gf / ex_gf * 100, hg_err);
        report("HGEMM FP16 Tensor Cores", "cublasHgemm (FP16 in/out)", size, hg_ms, hg_gf, "GFLOPS",
               hg_gf / ex_gf * 100, "of cublasGemmEx", hg_err < 5e-2f);
        double sg_gf = flops / (sg_ms / 1e3) / 1e9;
        printf("%-50s | %8.3f ms | %6.2f TFLOPS | %5.1f%% of cublasGemmEx (FP32 SIMT, no Tensor Cores)\n",
               "cublasSgemm (FP32)", sg_ms, sg_gf / 1e3, sg_gf / ex_gf * 100);
        report("HGEMM FP16 Tensor Cores", "cublasSgemm FP32 (no Tensor Cores, for scale)", size, sg_ms, sg_gf,
               "GFLOPS", sg_gf / ex_gf * 100, "of cublasGemmEx", true);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC16); cudaFree(dC); cudaFree(dAf); cudaFree(dBf);
}

int main(int argc, char** argv) {
    bool profile = profile_mode(argc, argv);
    std::vector<int> sizes;
    for (int i = 1; i < argc; i++) if (atoi(argv[i]) > 0) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = profile ? std::vector<int>{2048} : std::vector<int>{256, 1024, 2048, 4096};
    cublasHandle_t h; cublasCreate(&h);
    for (int n : sizes) {
        if (n % 128) { printf("n must be a multiple of 128\n"); return 1; }
        run_size(n, profile, h);
    }
    cublasDestroy(h);
}
