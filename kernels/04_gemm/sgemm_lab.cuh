// One parameterized FP32 SGEMM kernel that can express v5 and every optimization tried after it,
// so each experiment is a single config change measured in the same run.
//
//   BM x BN   block tile (C elements per block)          BK      k-depth of one shared-memory tile
//   WM x WN   warp tile (C elements per warp)             WNITER  warp sub-tiles along N
//   TM x TN   thread tile inside one warp sub-tile         DB      register-prefetch double buffering
//   PADA      padding (floats) on the transposed A tile    MINB    __launch_bounds__ min blocks per SM
//
// threads per block = (BM/WM) * (BN/WN) * 32. Each thread computes (WMITER*TM) x (WNITER*TN) outputs.
// v5 itself is BM=BN=128, BK=8, WM=16, WN=128, WNITER=1, TM=TN=8, no DB, no padding: one "warp tile"
// covering 2 rows of 16 threads. Requires n % BM == 0, n % BN == 0, n % BK == 0 (and n % 4 == 0).
//
// sm_75 (T4) has no cp.async, so double buffering is done by hand: the next tile is loaded from
// global memory into registers before computing on the current shared-memory stage, then stored into
// the other stage; one __syncthreads per k-tile instead of two.
#pragma once
#include <cstring>

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, bool DB, int PADA, int MINB>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32, MINB)
sgemm_lab(int n, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    constexpr int NT = (BM / WM) * (BN / WN) * 32;
    constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER, WSUBN = WN / WNITER;
    constexpr int TCOLS = WSUBN / TN;                       // threads along N in a warp sub-tile
    static_assert(WMITER >= 1 && WMITER * 32 * TM * TN * WNITER == WM * WN, "warp tile must split evenly");
    static_assert(TCOLS * (WSUBM / TM) == 32, "a warp sub-tile must hold exactly 32 threads");
    static_assert(TM % 4 == 0 && TN % 4 == 0 && BK % 4 == 0 && PADA % 4 == 0, "float4 alignment");
    constexpr int LDA = BM + PADA;                          // As is stored transposed: [BK][LDA]
    constexpr int STAGES = DB ? 2 : 1;
    constexpr int A4 = BM * BK / 4, B4 = BK * BN / 4;       // float4s per tile
    constexpr int A_PER = (A4 + NT - 1) / NT, B_PER = (B4 + NT - 1) / NT;

    __shared__ __align__(16) float As[STAGES][BK * LDA];
    __shared__ __align__(16) float Bs[STAGES][BK * BN];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warpCol = warp % (BN / WN), warpRow = warp / (BN / WN);
    const int tcol = lane % TCOLS, trow = lane / TCOLS;

    A += (size_t)blockIdx.y * BM * n;
    B += (size_t)blockIdx.x * BN;
    C += (size_t)blockIdx.y * BM * n + (size_t)blockIdx.x * BN;

    float4 ra[A_PER], rb[B_PER];
    auto load_global = [&](int k0) {
        #pragma unroll
        for (int i = 0; i < A_PER; i++) {
            const int f = tid + i * NT;
            if (A4 % NT == 0 || f < A4) {
                const int r = f / (BK / 4), c = f % (BK / 4);
                ra[i] = *reinterpret_cast<const float4*>(&A[(size_t)r * n + k0 + c * 4]);
            }
        }
        #pragma unroll
        for (int i = 0; i < B_PER; i++) {
            const int f = tid + i * NT;
            if (B4 % NT == 0 || f < B4) {
                const int r = f / (BN / 4), c = f % (BN / 4);
                rb[i] = *reinterpret_cast<const float4*>(&B[(size_t)(k0 + r) * n + c * 4]);
            }
        }
    };
    auto store_shared = [&](int s) {
        #pragma unroll
        for (int i = 0; i < A_PER; i++) {
            const int f = tid + i * NT;
            if (A4 % NT == 0 || f < A4) {
                const int r = f / (BK / 4), c = f % (BK / 4);
                As[s][(c * 4 + 0) * LDA + r] = ra[i].x;    // transpose A while staging it
                As[s][(c * 4 + 1) * LDA + r] = ra[i].y;
                As[s][(c * 4 + 2) * LDA + r] = ra[i].z;
                As[s][(c * 4 + 3) * LDA + r] = ra[i].w;
            }
        }
        #pragma unroll
        for (int i = 0; i < B_PER; i++) {
            const int f = tid + i * NT;
            if (B4 % NT == 0 || f < B4) {
                const int r = f / (BN / 4), c = f % (BN / 4);
                *reinterpret_cast<float4*>(&Bs[s][r * BN + c * 4]) = rb[i];
            }
        }
    };

    float acc[WMITER * TM][WNITER * TN];
    #pragma unroll
    for (int m = 0; m < WMITER * TM; m++)
        #pragma unroll
        for (int j = 0; j < WNITER * TN; j++) acc[m][j] = 0.0f;
    alignas(16) float regM[WMITER * TM];
    alignas(16) float regN[WNITER * TN];

    auto compute = [&](int s) {
        #pragma unroll
        for (int d = 0; d < BK; d++) {
            #pragma unroll
            for (int wm = 0; wm < WMITER; wm++)
                #pragma unroll
                for (int i = 0; i < TM; i += 4)
                    *reinterpret_cast<float4*>(&regM[wm * TM + i]) = *reinterpret_cast<const float4*>(
                        &As[s][d * LDA + warpRow * WM + wm * WSUBM + trow * TM + i]);
            #pragma unroll
            for (int wn = 0; wn < WNITER; wn++)
                #pragma unroll
                for (int j = 0; j < TN; j += 4)
                    *reinterpret_cast<float4*>(&regN[wn * TN + j]) = *reinterpret_cast<const float4*>(
                        &Bs[s][d * BN + warpCol * WN + wn * WSUBN + tcol * TN + j]);
            #pragma unroll
            for (int m = 0; m < WMITER * TM; m++)
                #pragma unroll
                for (int j = 0; j < WNITER * TN; j++) acc[m][j] += regM[m] * regN[j];
        }
    };

    load_global(0);
    store_shared(0);
    __syncthreads();
    int s = 0;
    for (int k0 = 0; k0 < n; k0 += BK) {
        const bool more = k0 + BK < n;
        if (DB && more) load_global(k0 + BK);     // global loads in flight while we compute
        compute(s);
        if (DB) {
            if (more) store_shared(s ^ 1);         // other stage: nobody reads it this iteration
            __syncthreads();
            s ^= 1;
        } else {
            __syncthreads();
            if (more) { load_global(k0 + BK); store_shared(0); __syncthreads(); }
        }
    }

    #pragma unroll
    for (int wm = 0; wm < WMITER; wm++)
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int wn = 0; wn < WNITER; wn++)
                #pragma unroll
                for (int j = 0; j < TN; j += 4) {
                    const int row = warpRow * WM + wm * WSUBM + trow * TM + i;
                    const int col = warpCol * WN + wn * WSUBN + tcol * TN + j;
                    const float* a = &acc[wm * TM + i][wn * TN + j];
                    *reinterpret_cast<float4*>(&C[(size_t)row * n + col]) = make_float4(a[0], a[1], a[2], a[3]);
                }
}

// ---------------------------------------------------------------------------------------------
// Every configuration that is compiled. X(name, BM, BN, BK, WM, WN, WNITER, TM, TN, DB, PADA, MINB)
// threads = (BM/WM)*(BN/WN)*32 is noted in the comment.
#define SGEMM_LAB_CONFIGS(X)                                                                      \
    /* --- Part B: v5 re-expressed, then one change at a time from v5 ------------------------ */ \
    X(lab_v5,          128, 128,  8,  16, 128, 1,  8,  8, false, 0, 1) /* 256 thr: = v5       */  \
    X(v5_warptile,     128, 128,  8,  64,  32, 2,  8,  4, false, 0, 1) /* 256 thr             */  \
    X(v5_dbuf,         128, 128,  8,  16, 128, 1,  8,  8, true,  0, 1) /* 256 thr             */  \
    X(v5_128thr,       128, 128,  8,  64,  64, 1,  8, 16, false, 0, 1) /* 128 thr, 8x16/thread*/  \
    X(v5_512thr,       128, 128,  8,  32,  32, 1,  4,  8, false, 0, 1) /* 512 thr, 4x8/thread */  \
    X(v5_lb2,          128, 128,  8,  16, 128, 1,  8,  8, false, 0, 2) /* launch_bounds(256,2)*/  \
    X(v5_padA,         128, 128,  8,  16, 128, 1,  8,  8, false, 4, 1) /* As stride 132       */  \
    X(v5_bk16,         128, 128, 16,  16, 128, 1,  8,  8, false, 0, 1) /* half the barriers   */  \
    /* --- Part B: cumulative ladder ------------------------------------------------------- */ \
    X(c2_warp_dbuf,    128, 128,  8,  64,  32, 2,  8,  4, true,  0, 1)                            \
    X(c3_warp_dbuf_pad,128, 128,  8,  64,  32, 2,  8,  4, true,  4, 1)                            \
    X(c4_wdp_bk16,     128, 128, 16,  64,  32, 2,  8,  4, true,  4, 1)                            \
    X(c4b_wd_bk16_nopad, 128, 128, 16, 64,  32, 2,  8,  4, true,  0, 1) /* 32 KB smem: 2 blk/SM */ \
    X(c5_wdp_bk16_lb2, 128, 128, 16,  64,  32, 2,  8,  4, true,  4, 2)                            \
    /* --- Part C: autotuner search space (double-buffered, padded) -------------------------- */ \
    X(t64x64_8x8,       64,  64,  8,  32,  64, 1,  8,  8, true,  4, 1) /*  64 thr             */  \
    X(t64x64_4x8,       64,  64,  8,  32,  32, 1,  4,  8, true,  4, 1) /* 128 thr             */  \
    X(t64x128_8x8,      64, 128,  8,  32,  64, 1,  8,  8, true,  4, 1) /* 128 thr             */  \
    X(t128x64_8x8,     128,  64,  8,  64,  32, 1,  8,  8, true,  4, 1) /* 128 thr             */  \
    X(t128x128_8x8,    128, 128,  8,  64,  32, 2,  8,  4, true,  4, 1) /* 256 thr (= c3)      */  \
    X(t128x128_8x16,   128, 128,  8,  64,  64, 1,  8, 16, true,  4, 1) /* 128 thr             */  \
    X(t128x128_16x8,   128, 128,  8,  64,  64, 1, 16,  8, true,  4, 1) /* 128 thr             */  \
    X(t128x256_8x8,    128, 256,  8,  32,  64, 1,  8,  8, true,  4, 1) /* 512 thr             */  \
    X(t256x128_8x8,    256, 128,  8,  64,  32, 1,  8,  8, true,  4, 1) /* 512 thr             */  \
    X(t128x256_8x16,   128, 256,  8,  64,  64, 1,  8, 16, true,  4, 1) /* 256 thr             */  \
    X(t128x128_8x8_bk16, 128, 128, 16, 64, 32, 2,  8,  4, true,  4, 1) /* 256 thr (= c4)      */

struct LabConfig { const char* name; int bm, bn, threads; };

#define SGEMM_LAB_INFO(name, BM, BN, BK, WM, WN, WNI, TM, TN, DB, PADA, MINB) \
    {#name, BM, BN, (BM / WM) * (BN / WN) * 32},
static const LabConfig kLabConfigs[] = {SGEMM_LAB_CONFIGS(SGEMM_LAB_INFO)};
constexpr int kNumLabConfigs = sizeof(kLabConfigs) / sizeof(kLabConfigs[0]);
#undef SGEMM_LAB_INFO

// Launch configuration `name` on n x n x n. Returns false if the name is unknown or n doesn't fit.
inline bool launch_lab(const char* name, int n, const float* A, const float* B, float* C) {
#define SGEMM_LAB_LAUNCH(cname, BM, BN, BK, WM, WN, WNI, TM, TN, DB, PADA, MINB)                    \
    if (strcmp(name, #cname) == 0) {                                                                 \
        if (n % BM || n % BN || n % BK) return false;                                                \
        sgemm_lab<BM, BN, BK, WM, WN, WNI, TM, TN, DB, PADA, MINB>                                   \
            <<<dim3(n / BN, n / BM), (BM / WM) * (BN / WN) * 32>>>(n, A, B, C);                      \
        return true;                                                                                 \
    }
    SGEMM_LAB_CONFIGS(SGEMM_LAB_LAUNCH)
#undef SGEMM_LAB_LAUNCH
    return false;
}
