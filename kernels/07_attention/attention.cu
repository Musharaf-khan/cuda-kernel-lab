// Scaled dot-product attention, forward pass, FP32, non-causal:
//     O = softmax(Q K^T / sqrt(d)) V        Q, K, V, O: [BH heads][N tokens][d = 64]
//
//   unfused   the textbook way: S = Q K^T (cuBLAS, written to HBM as BH x N x N floats),
//             softmax over S (in place), O = P V (cuBLAS). Memory: O(N^2).
//   fused     FlashAttention-style: each block owns 64 query rows; K/V stream through shared
//             memory in tiles of 32; scores live in registers and are folded into the output
//             with ONLINE SOFTMAX (running max + rescale). The N x N matrix never exists.
//             Extra memory: O(1) per row (just the output).
//
// Usage: ./attention [profile]
#include "../../common/common.h"
#include "../../common/reduce.cuh"
#include <cublas_v2.h>
#include <vector>

constexpr int D = 64;          // head dimension
constexpr int BH = 16;         // batch 2 x 8 heads
constexpr int BR = 64;         // query rows per block (one thread per row)
constexpr int BC = 32;         // key/value rows per shared-memory tile

// ---------------------------------------------------------------- fused kernel
// scale_log2 = log2(e) / sqrt(d): folding log2(e) into Q lets us use exp2f (one hardware op).
template <int HD, int ROWS, int TILE>
__global__ void __launch_bounds__(ROWS)
flash_attention_fwd(const float* __restrict__ Q, const float* __restrict__ K,
                    const float* __restrict__ V, float* __restrict__ O, int n, float scale_log2) {
    __shared__ float4 Ks[TILE * HD / 4];
    __shared__ float4 Vs[TILE * HD / 4];

    const size_t head = (size_t)blockIdx.y * n * HD;
    const int row = blockIdx.x * ROWS + threadIdx.x;              // n is a multiple of ROWS
    const float4* Kh = reinterpret_cast<const float4*>(K + head);
    const float4* Vh = reinterpret_cast<const float4*>(V + head);

    // This thread's query row and output accumulator live in registers for the whole kernel.
    float q[HD], o[HD];
    const float4* q4 = reinterpret_cast<const float4*>(Q + head + (size_t)row * HD);
    #pragma unroll
    for (int c = 0; c < HD / 4; c++) {
        float4 t = q4[c];
        q[4 * c + 0] = t.x * scale_log2; q[4 * c + 1] = t.y * scale_log2;
        q[4 * c + 2] = t.z * scale_log2; q[4 * c + 3] = t.w * scale_log2;
    }
    #pragma unroll
    for (int c = 0; c < HD; c++) o[c] = 0.0f;
    float m = -INFINITY, l = 0.0f;                                 // online-softmax state

    for (int kv = 0; kv < n; kv += TILE) {
        // Cooperative, coalesced float4 load of one K tile and one V tile into shared memory.
        for (int f = threadIdx.x; f < TILE * HD / 4; f += ROWS) {
            Ks[f] = Kh[(size_t)kv * HD / 4 + f];
            Vs[f] = Vh[(size_t)kv * HD / 4 + f];
        }
        __syncthreads();

        // Scores for this tile (log2 domain). Every thread reads the same K row at the same
        // time -> shared-memory broadcast, no bank conflicts.
        float s[TILE];
        float tile_max = -INFINITY;
        #pragma unroll
        for (int j = 0; j < TILE; j++) {
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;              // 4 chains for ILP
            #pragma unroll
            for (int c = 0; c < HD / 4; c++) {
                float4 k = Ks[j * HD / 4 + c];
                a0 += q[4 * c + 0] * k.x; a1 += q[4 * c + 1] * k.y;
                a2 += q[4 * c + 2] * k.z; a3 += q[4 * c + 3] * k.w;
            }
            s[j] = (a0 + a1) + (a2 + a3);
            tile_max = fmaxf(tile_max, s[j]);
        }

        // Online softmax: if the max went up, shrink everything accumulated so far.
        const float m_new = fmaxf(m, tile_max);
        const float corr = exp2f(m - m_new);                       // = 0 on the first tile
        l *= corr;
        #pragma unroll
        for (int c = 0; c < HD; c++) o[c] *= corr;

        #pragma unroll
        for (int j = 0; j < TILE; j++) {
            const float p = exp2f(s[j] - m_new);
            l += p;
            #pragma unroll
            for (int c = 0; c < HD / 4; c++) {
                float4 v = Vs[j * HD / 4 + c];
                o[4 * c + 0] += p * v.x; o[4 * c + 1] += p * v.y;
                o[4 * c + 2] += p * v.z; o[4 * c + 3] += p * v.w;
            }
        }
        m = m_new;
        __syncthreads();                                           // before the next tile overwrites Ks/Vs
    }

    const float inv = 1.0f / l;
    float4* o4 = reinterpret_cast<float4*>(O + head + (size_t)row * HD);
    #pragma unroll
    for (int c = 0; c < HD / 4; c++)
        o4[c] = make_float4(o[4 * c] * inv, o[4 * c + 1] * inv, o[4 * c + 2] * inv, o[4 * c + 3] * inv);
}

// ---------------------------------------------------------------- unfused building block
// Online softmax over each row (same as kernels/05_softmax v3), safe to run in place.
__global__ void softmax_rows_inplace(float* x, int n) {
    float* row = x + (size_t)blockIdx.x * n;
    float m = -INFINITY, l = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row[i];
        if (v > m) { l = l * __expf(m - v) + 1.0f; m = v; } else { l += __expf(v - m); }
    }
    block_online_merge(m, l);
    float inv = 1.0f / l;
    for (int i = threadIdx.x; i < n; i += blockDim.x) row[i] = __expf(row[i] - m) * inv;
}

// ---------------------------------------------------------------- CPU reference (small N only)
static void cpu_attention(const float* Q, const float* K, const float* V, float* O, int bh, int n) {
    std::vector<double> s(n);
    const double scale = 1.0 / sqrt((double)D);
    for (int h = 0; h < bh; h++)
        for (int i = 0; i < n; i++) {
            const float* q = Q + ((size_t)h * n + i) * D;
            double mx = -INFINITY;
            for (int j = 0; j < n; j++) {
                const float* k = K + ((size_t)h * n + j) * D;
                double a = 0; for (int c = 0; c < D; c++) a += (double)q[c] * k[c];
                s[j] = a * scale; mx = fmax(mx, s[j]);
            }
            double sum = 0; for (int j = 0; j < n; j++) { s[j] = exp(s[j] - mx); sum += s[j]; }
            for (int c = 0; c < D; c++) {
                double acc = 0;
                for (int j = 0; j < n; j++) acc += s[j] * V[((size_t)h * n + j) * D + c];
                O[((size_t)h * n + i) * D + c] = (float)(acc / sum);
            }
        }
}

int main(int argc, char** argv) {
    bool profile = profile_mode(argc, argv);
    std::vector<int> seqs = profile ? std::vector<int>{2048} : std::vector<int>{256, 1024, 2048, 4096, 8192, 16384};

    cublasHandle_t h; cublasCreate(&h);
    const float scale = 1.0f / sqrtf((float)D);
    const float scale_log2 = scale * 1.4426950408889634f;
    printf("Attention forward, FP32, BH=%d (batch 2 x 8 heads), d=%d, non-causal\n", BH, D);

    for (int n : seqs) {
        size_t qkv = (size_t)BH * n * D, qkv_bytes = qkv * sizeof(float);
        size_t s_bytes = (size_t)BH * n * n * sizeof(float);
        double flops = 4.0 * BH * (double)n * n * D;                // QK^T + PV, 2 FLOPs per MAC

        std::vector<float> hQ(qkv), hK(qkv), hV(qkv), hO(qkv), hRef(qkv);
        srand(1234 + n);
        auto rnd = []() { return rand() / (float)RAND_MAX * 2 - 1; };
        for (size_t i = 0; i < qkv; i++) { hQ[i] = rnd(); hK[i] = rnd(); hV[i] = rnd(); }

        float *dQ, *dK, *dV, *dO;
        CHECK(cudaMalloc(&dQ, qkv_bytes)); CHECK(cudaMalloc(&dK, qkv_bytes));
        CHECK(cudaMalloc(&dV, qkv_bytes)); CHECK(cudaMalloc(&dO, qkv_bytes));
        CHECK(cudaMemcpy(dQ, hQ.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dK, hK.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dV, hV.data(), qkv_bytes, cudaMemcpyHostToDevice));

        // The unfused path needs the full score matrix. At large N it simply does not fit.
        float* dS = nullptr;
        bool unfused_fits = cudaMalloc(&dS, s_bytes) == cudaSuccess;
        if (!unfused_fits) { cudaGetLastError(); dS = nullptr; }      // clear the error

        auto fused = [&]() {
            flash_attention_fwd<D, BR, BC><<<dim3(n / BR, BH), BR>>>(dQ, dK, dV, dO, n, scale_log2);
        };
        auto unfused = [&]() {
            const float one = 1.0f, zero = 0.0f;
            // Row-major S = Q K^T  <=>  column-major S^T = K Q^T (see README for the derivation).
            cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, D, &scale,
                                      dK, D, (long long)n * D, dQ, D, (long long)n * D,
                                      &zero, dS, n, (long long)n * n, BH);
            softmax_rows_inplace<<<BH * n, 256>>>(dS, n);
            // Row-major O = P V  <=>  column-major O^T = V^T P^T.
            cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, D, n, n, &one,
                                      dV, D, (long long)n * D, dS, n, (long long)n * n,
                                      &zero, dO, D, (long long)n * D, BH);
        };

        if (profile) {
            fused(); if (unfused_fits) unfused();
            CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
            cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); if (dS) cudaFree(dS);
            continue;
        }

        char size[32]; snprintf(size, sizeof size, "N=%d", n);
        int iters = n >= 8192 ? 3 : 10;

        // Reference: CPU (double) for the smallest size, otherwise the unfused GPU result.
        bool have_ref = false;
        if (n <= 256) { cpu_attention(hQ.data(), hK.data(), hV.data(), hRef.data(), BH, n); have_ref = true; }

        float unfused_ms = 0.0f;
        if (unfused_fits) {
            CHECK(cudaMemset(dO, 0, qkv_bytes));
            unfused_ms = time_ms(unfused, iters);
            CHECK(cudaMemcpy(hO.data(), dO, qkv_bytes, cudaMemcpyDeviceToHost));
            float err = have_ref ? max_rel_err(hO.data(), hRef.data(), qkv) : 0.0f;
            if (!have_ref) { hRef = hO; have_ref = true; }
            bool ok = err < 1e-3f;
            double gf = flops / (unfused_ms / 1e3) / 1e9;
            printf("N=%-6d unfused (cuBLAS + softmax) | %9.3f ms | %7.1f GFLOPS | score matrix %7.1f MB | %s\n",
                   n, unfused_ms, gf, s_bytes / 1e6, ok ? "PASS" : "FAIL");
            report("Attention (BH=16, d=64)", "unfused: cuBLAS QK^T + softmax + cuBLAS PV", size,
                   unfused_ms, gf, "GFLOPS", 100.0, "of unfused speed", ok);
        } else {
            printf("N=%-6d unfused (cuBLAS + softmax) | OUT OF MEMORY: score matrix needs %.1f GB\n",
                   n, s_bytes / 1e9);
            printf("RESULT|Attention (BH=16, d=64)|unfused: cuBLAS QK^T + softmax + cuBLAS PV|%s|"
                   "0|0|GFLOPS|0|of unfused speed|OOM (needs %.1f GB for S)\n", size, s_bytes / 1e9);
        }

        CHECK(cudaMemset(dO, 0, qkv_bytes));
        float fused_ms = time_ms(fused, iters);
        CHECK(cudaMemcpy(hO.data(), dO, qkv_bytes, cudaMemcpyDeviceToHost));
        float err = have_ref ? max_rel_err(hO.data(), hRef.data(), qkv) : 0.0f;
        bool finite = max_rel_err(hO.data(), hO.data(), qkv) == 0.0f;   // NaN/Inf check
        bool ok = finite && err < 1e-3f;
        double gf = flops / (fused_ms / 1e3) / 1e9;
        double vs = unfused_fits ? unfused_ms / fused_ms : 0.0;
        printf("N=%-6d fused (FlashAttention-style)| %9.3f ms | %7.1f GFLOPS | extra memory     0.0 MB | "
               "%s%.2fx vs unfused | err %.1e %s%s\n",
               n, fused_ms, gf, unfused_fits ? "" : "(n/a) ", vs, err, ok ? "PASS" : "FAIL",
               have_ref ? "" : " (no reference: finite-check only)");
        report("Attention (BH=16, d=64)", "fused: FlashAttention-style (online softmax, tiled K/V)", size,
               fused_ms, gf, "GFLOPS", unfused_fits ? 100.0 * unfused_ms / fused_ms : 0.0, "of unfused speed", ok);

        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); if (dS) cudaFree(dS);
    }
    cublasDestroy(h);
}
