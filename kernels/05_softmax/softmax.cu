// Row-wise softmax over a [M x N] matrix (M = N = 4096, FP32):  y = exp(x - max) / sum(exp(x - max))
//
//   v1 thread-per-row, 3 passes   each thread walks its own row: uncoalesced
//   v2 block-per-row,  3 passes   256 threads share a row: coalesced; reads x 3 times
//   v3 online softmax, 2 passes   running (max, sum) in ONE pass, then write: reads x 2 times
//   v4 register-cached, 1 pass    row held in registers (float4 loads): reads x once
//
// Softmax is memory-bound (a handful of FLOPs per element), so the score is effective
// bandwidth = (bytes read once + bytes written once) / time, i.e. the minimum possible traffic.
//
// Usage: ./softmax [profile]
#include "../../common/common.h"
#include "../../common/reduce.cuh"
#include <vector>

constexpr int M = 4096, N = 4096;
constexpr int THREADS = 256;

// v0: textbook formula with NO max subtraction. Only used to demonstrate overflow.
__global__ void softmax_unsafe(const float* x, float* y, int n) {
    const float* row = x + (size_t)blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += __expf(row[i]);
    s = block_sum(s);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[(size_t)blockIdx.x * n + i] = __expf(row[i]) / s;
}

// v1: one thread per row. Neighbouring threads touch addresses n floats apart -> uncoalesced.
__global__ void softmax_thread_per_row(const float* x, float* y, int m, int n) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;
    const float* row = x + (size_t)r * n;
    float* out = y + (size_t)r * n;
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) mx = fmaxf(mx, row[i]);                  // pass 1: max
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += __expf(row[i] - mx);                // pass 2: sum
    float inv = 1.0f / s;
    for (int i = 0; i < n; i++) out[i] = __expf(row[i] - mx) * inv;      // pass 3: write
}

// v2: one block per row, strided loops (coalesced), block reductions for max and sum.
__global__ void softmax_block_3pass(const float* x, float* y, int n) {
    const float* row = x + (size_t)blockIdx.x * n;
    float* out = y + (size_t)blockIdx.x * n;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, row[i]);
    mx = block_max(mx);
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += __expf(row[i] - mx);
    s = block_sum(s);
    float inv = 1.0f / s;
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = __expf(row[i] - mx) * inv;
}

// v3: online softmax. Max and normalizer are computed together in ONE pass:
//   when a new element raises the max, rescale the running sum by exp(m_old - m_new).
// This is the exact trick FlashAttention uses to avoid ever storing the full score row.
__global__ void softmax_online(const float* x, float* y, int n) {
    const float* row = x + (size_t)blockIdx.x * n;
    float* out = y + (size_t)blockIdx.x * n;
    float m = -INFINITY, l = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row[i];
        if (v > m) { l = l * __expf(m - v) + 1.0f; m = v; }
        else       { l += __expf(v - m); }
    }
    block_online_merge(m, l);                 // combine per-thread (m, l) pairs
    float inv = 1.0f / l;
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = __expf(row[i] - m) * inv;
}

// v4: the whole row lives in registers: each of 256 threads holds 16 floats (4 x float4).
// One global read, one global write: the minimum traffic softmax can possibly do.
template <int COLS>
__global__ void __launch_bounds__(THREADS) softmax_registers(const float* x, float* y) {
    constexpr int VEC = COLS / 4 / THREADS;   // float4s per thread (4 for COLS = 4096)
    static_assert(COLS % (4 * THREADS) == 0, "COLS must be a multiple of 1024");
    const float4* row = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * COLS);
    float4* out = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * COLS);

    float4 v[VEC];
    float mx = -INFINITY;
    #pragma unroll
    for (int j = 0; j < VEC; j++) {
        v[j] = row[j * THREADS + threadIdx.x];             // coalesced 128-bit loads
        mx = fmaxf(mx, fmaxf(fmaxf(v[j].x, v[j].y), fmaxf(v[j].z, v[j].w)));
    }
    mx = block_max(mx);
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < VEC; j++) {
        v[j].x = __expf(v[j].x - mx); v[j].y = __expf(v[j].y - mx);
        v[j].z = __expf(v[j].z - mx); v[j].w = __expf(v[j].w - mx);
        s += v[j].x + v[j].y + v[j].z + v[j].w;
    }
    s = block_sum(s);
    float inv = 1.0f / s;
    #pragma unroll
    for (int j = 0; j < VEC; j++) {
        v[j].x *= inv; v[j].y *= inv; v[j].z *= inv; v[j].w *= inv;
        out[j * THREADS + threadIdx.x] = v[j];
    }
}

// Softmax outputs are tiny (~1/4096), so use a true relative error, not |a-b|/(1+|ref|).
static float max_true_rel_err(const float* got, const float* ref, size_t count) {
    float m = 0.0f;
    for (size_t i = 0; i < count; i++) {
        if (!std::isfinite(got[i])) return INFINITY;
        m = fmaxf(m, fabsf(got[i] - ref[i]) / fmaxf(fabsf(ref[i]), 1e-30f));
    }
    return m;
}

static void cpu_softmax(const float* x, float* y, int m, int n) {
    for (int r = 0; r < m; r++) {
        const float* row = x + (size_t)r * n;
        double mx = -INFINITY, s = 0.0;
        for (int i = 0; i < n; i++) mx = fmax(mx, (double)row[i]);
        for (int i = 0; i < n; i++) s += exp(row[i] - mx);
        for (int i = 0; i < n; i++) y[(size_t)r * n + i] = (float)(exp(row[i] - mx) / s);
    }
}

int main(int argc, char** argv) {
    bool profile = profile_mode(argc, argv);
    size_t count = (size_t)M * N, bytes = count * sizeof(float);

    std::vector<float> hx(count), hy(count), href(count);
    srand(7);
    for (size_t i = 0; i < count; i++) hx[i] = (rand() / (float)RAND_MAX * 2 - 1) * 5.0f;   // logits in [-5, 5]

    float *dx, *dy;
    CHECK(cudaMalloc(&dx, bytes)); CHECK(cudaMalloc(&dy, bytes));
    CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));

    std::function<void()> launches[] = {
        [&]() { softmax_thread_per_row<<<(M + THREADS - 1) / THREADS, THREADS>>>(dx, dy, M, N); },
        [&]() { softmax_block_3pass<<<M, THREADS>>>(dx, dy, N); },
        [&]() { softmax_online<<<M, THREADS>>>(dx, dy, N); },
        [&]() { softmax_registers<N><<<M, THREADS>>>(dx, dy); },
    };
    const int NK = 4;
    const char* versions[] = {"v1 thread-per-row, 3 passes", "v2 block-per-row, 3 passes",
                              "v3 online softmax, 2 passes", "v4 register-cached, 1 pass (float4)"};

    if (profile) {
        for (int k = 0; k < NK; k++) launches[k]();
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        return 0;
    }

    cpu_softmax(hx.data(), href.data(), M, N);

    // --- Numerical stability demo: logits around 100 overflow exp() in FP32 (max ~3.4e38 = e^88.7).
    {
        const int sm = 64;
        std::vector<float> big((size_t)sm * N), out((size_t)sm * N);
        for (auto& v : big) v = 90.0f + (rand() / (float)RAND_MAX) * 20.0f;   // logits in [90, 110]
        float *db, *dout;
        CHECK(cudaMalloc(&db, big.size() * 4)); CHECK(cudaMalloc(&dout, big.size() * 4));
        CHECK(cudaMemcpy(db, big.data(), big.size() * 4, cudaMemcpyHostToDevice));
        softmax_unsafe<<<sm, THREADS>>>(db, dout, N);
        CHECK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
        size_t bad = 0; for (float v : out) if (!std::isfinite(v)) bad++;
        softmax_registers<N><<<sm, THREADS>>>(db, dout);
        CHECK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
        size_t bad2 = 0; for (float v : out) if (!std::isfinite(v)) bad2++;
        printf("Stability check, logits in [90,110]: no max-subtraction -> %zu/%zu NaN/Inf outputs; "
               "stable v4 -> %zu NaN/Inf\n", bad, out.size(), bad2);
        cudaFree(db); cudaFree(dout);
    }

    printf("Softmax %d x %d (FP32)\n", M, N);
    float base_ms = 0;
    for (int k = 0; k < NK; k++) {
        CHECK(cudaMemset(dy, 0, bytes));
        float ms = time_ms(launches[k], k == 0 ? 5 : 50);
        if (k == 0) base_ms = ms;
        CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
        float err = max_true_rel_err(hy.data(), href.data(), count);
        bool ok = err < 1e-4f;                    // relative, vs a double-precision CPU reference
        double gbps = 2.0 * bytes / (ms / 1e3) / 1e9;
        printf("%-38s | %8.3f ms | %6.1f GB/s | %3.0f%% of peak | %6.1fx vs v1 | err %.1e %s\n",
               versions[k], ms, gbps, gbps / T4_PEAK_GBS * 100, base_ms / ms, err, ok ? "PASS" : "FAIL");
        report("Softmax", versions[k], "4096x4096", ms, gbps, "GB/s", gbps / T4_PEAK_GBS * 100, "of peak BW", ok);
    }

    cudaFree(dx); cudaFree(dy);
}
