// LayerNorm over the last dimension of a [M x N] matrix (M = N = 4096, FP32):
//     y = (x - mean) / sqrt(var + eps) * gamma + beta
//
//   v1 two-pass           pass 1 mean, pass 2 variance, pass 3 write: reads x 3 times
//   v2 Welford            mean AND variance in one numerically stable pass, then write: reads x 2 times
//   v3 register-cached    row held in registers (float4): reads x once, writes y once
//
// Fusion experiment (pre-LN transformer block: h = x + residual; y = LayerNorm(h)):
//   unfused  add kernel (writes h to HBM) + v3 LayerNorm (reads h back)
//   fused    one kernel: read x and residual, write h and y; h never round-trips through HBM
//
// Usage: ./layernorm [profile]
#include "../../common/common.h"
#include "../../common/reduce.cuh"
#include <vector>

constexpr int M = 4096, N = 4096;
constexpr int THREADS = 256;
constexpr float EPS = 1e-5f;

// v1: classic two-pass statistics.
__global__ void layernorm_two_pass(const float* x, const float* gamma, const float* beta, float* y, int n) {
    const float* row = x + (size_t)blockIdx.x * n;
    float* out = y + (size_t)blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += row[i];
    const float mean = block_sum(s) / n;
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float d = row[i] - mean; v += d * d; }
    const float rstd = rsqrtf(block_sum(v) / n + EPS);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = (row[i] - mean) * rstd * gamma[i] + beta[i];
}

// ---- Welford: running (count, mean, M2) where M2 = sum of squared deviations from the mean.
// Unlike sum/sum-of-squares, it does not suffer catastrophic cancellation.
// Two partial results merge with Chan et al.'s parallel formula:
__device__ __forceinline__ void welford_merge(float& n, float& mean, float& m2, float nb, float meanb, float m2b) {
    float nn = n + nb;
    if (nn == 0.0f) return;
    float delta = meanb - mean;
    mean += delta * (nb / nn);
    m2 += m2b + delta * delta * (n * nb / nn);
    n = nn;
}

__device__ __forceinline__ void block_welford(float& n, float& mean, float& m2) {
    __shared__ float sn[32], smean[32], sm2[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        welford_merge(n, mean, m2, __shfl_xor_sync(FULL_MASK, n, o),
                      __shfl_xor_sync(FULL_MASK, mean, o), __shfl_xor_sync(FULL_MASK, m2, o));
    if (lane == 0) { sn[warp] = n; smean[warp] = mean; sm2[warp] = m2; }
    __syncthreads();
    if (warp == 0) {
        n    = (lane < nwarps) ? sn[lane]    : 0.0f;
        mean = (lane < nwarps) ? smean[lane] : 0.0f;
        m2   = (lane < nwarps) ? sm2[lane]   : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            welford_merge(n, mean, m2, __shfl_xor_sync(FULL_MASK, n, o),
                          __shfl_xor_sync(FULL_MASK, mean, o), __shfl_xor_sync(FULL_MASK, m2, o));
        if (lane == 0) { sn[0] = n; smean[0] = mean; sm2[0] = m2; }
    }
    __syncthreads();
    n = sn[0]; mean = smean[0]; m2 = sm2[0];
    __syncthreads();
}

// v2: single-pass statistics with Welford, then one pass to write.
__global__ void layernorm_welford(const float* x, const float* gamma, const float* beta, float* y, int n) {
    const float* row = x + (size_t)blockIdx.x * n;
    float* out = y + (size_t)blockIdx.x * n;
    float cnt = 0.0f, mean = 0.0f, m2 = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row[i];
        cnt += 1.0f;
        float d = v - mean;
        mean += d / cnt;
        m2 += d * (v - mean);
    }
    block_welford(cnt, mean, m2);
    const float rstd = rsqrtf(m2 / n + EPS);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = (row[i] - mean) * rstd * gamma[i] + beta[i];
}

// v3: row in registers. One HBM read, one HBM write. With the data already on-chip,
// an exact two-pass (mean, then variance) costs no extra memory traffic.
template <int COLS>
__global__ void __launch_bounds__(THREADS)
layernorm_registers(const float* x, const float* gamma, const float* beta, float* y) {
    constexpr int VEC = COLS / 4 / THREADS;
    static_assert(COLS % (4 * THREADS) == 0, "COLS must be a multiple of 1024");
    const float4* row = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * COLS);
    float4* out = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * COLS);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);

    float4 v[VEC];
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < VEC; j++) {
        v[j] = row[j * THREADS + threadIdx.x];
        s += v[j].x + v[j].y + v[j].z + v[j].w;
    }
    const float mean = block_sum(s) / COLS;
    float var = 0.0f;
    #pragma unroll
    for (int j = 0; j < VEC; j++) {
        v[j].x -= mean; v[j].y -= mean; v[j].z -= mean; v[j].w -= mean;
        var += v[j].x * v[j].x + v[j].y * v[j].y + v[j].z * v[j].z + v[j].w * v[j].w;
    }
    const float rstd = rsqrtf(block_sum(var) / COLS + EPS);
    #pragma unroll
    for (int j = 0; j < VEC; j++) {
        const int idx = j * THREADS + threadIdx.x;
        float4 g = g4[idx], b = b4[idx], o;
        o.x = v[j].x * rstd * g.x + b.x; o.y = v[j].y * rstd * g.y + b.y;
        o.z = v[j].z * rstd * g.z + b.z; o.w = v[j].w * rstd * g.w + b.w;
        out[idx] = o;
    }
}

// Unfused step 1: h = x + residual (elementwise, float4).
__global__ void residual_add(const float4* x, const float4* res, float4* h, size_t n4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) { float4 a = x[i], b = res[i]; h[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
}

// Fused: h = x + residual; y = LayerNorm(h). Writes h (needed by the next residual
// connection) and y, but never reads h back from HBM.
template <int COLS>
__global__ void __launch_bounds__(THREADS)
add_layernorm_fused(const float* x, const float* res, const float* gamma, const float* beta,
                    float* h, float* y) {
    constexpr int VEC = COLS / 4 / THREADS;
    const size_t base = (size_t)blockIdx.x * COLS;
    const float4* x4 = reinterpret_cast<const float4*>(x + base);
    const float4* r4 = reinterpret_cast<const float4*>(res + base);
    float4* h4 = reinterpret_cast<float4*>(h + base);
    float4* y4 = reinterpret_cast<float4*>(y + base);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);

    float4 v[VEC];
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < VEC; j++) {
        const int idx = j * THREADS + threadIdx.x;
        float4 a = x4[idx], b = r4[idx];
        v[j] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        h4[idx] = v[j];
        s += v[j].x + v[j].y + v[j].z + v[j].w;
    }
    const float mean = block_sum(s) / COLS;
    float var = 0.0f;
    #pragma unroll
    for (int j = 0; j < VEC; j++) {
        v[j].x -= mean; v[j].y -= mean; v[j].z -= mean; v[j].w -= mean;
        var += v[j].x * v[j].x + v[j].y * v[j].y + v[j].z * v[j].z + v[j].w * v[j].w;
    }
    const float rstd = rsqrtf(block_sum(var) / COLS + EPS);
    #pragma unroll
    for (int j = 0; j < VEC; j++) {
        const int idx = j * THREADS + threadIdx.x;
        float4 g = g4[idx], b = b4[idx], o;
        o.x = v[j].x * rstd * g.x + b.x; o.y = v[j].y * rstd * g.y + b.y;
        o.z = v[j].z * rstd * g.z + b.z; o.w = v[j].w * rstd * g.w + b.w;
        y4[idx] = o;
    }
}

static void cpu_layernorm(const float* x, const float* g, const float* b, float* y, int m, int n) {
    for (int r = 0; r < m; r++) {
        const float* row = x + (size_t)r * n;
        double mean = 0, var = 0;
        for (int i = 0; i < n; i++) mean += row[i];
        mean /= n;
        for (int i = 0; i < n; i++) var += (row[i] - mean) * (row[i] - mean);
        double rstd = 1.0 / sqrt(var / n + EPS);
        for (int i = 0; i < n; i++) y[(size_t)r * n + i] = (float)((row[i] - mean) * rstd * g[i] + b[i]);
    }
}

int main(int argc, char** argv) {
    bool profile = profile_mode(argc, argv);
    size_t count = (size_t)M * N, bytes = count * sizeof(float);

    std::vector<float> hx(count), hres(count), hg(N), hb(N), hy(count), hh(count), href(count), hsum(count);
    srand(11);
    auto rnd = []() { return rand() / (float)RAND_MAX * 2 - 1; };
    for (size_t i = 0; i < count; i++) { hx[i] = 3.0f + 2.0f * rnd(); hres[i] = rnd(); }  // non-zero mean on purpose
    for (int i = 0; i < N; i++) { hg[i] = 1.0f + 0.1f * rnd(); hb[i] = 0.1f * rnd(); }

    float *dx, *dres, *dg, *db, *dy, *dh;
    CHECK(cudaMalloc(&dx, bytes)); CHECK(cudaMalloc(&dres, bytes)); CHECK(cudaMalloc(&dy, bytes));
    CHECK(cudaMalloc(&dh, bytes)); CHECK(cudaMalloc(&dg, N * 4)); CHECK(cudaMalloc(&db, N * 4));
    CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dres, hres.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dg, hg.data(), N * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hb.data(), N * 4, cudaMemcpyHostToDevice));

    std::function<void()> ln[] = {
        [&]() { layernorm_two_pass<<<M, THREADS>>>(dx, dg, db, dy, N); },
        [&]() { layernorm_welford<<<M, THREADS>>>(dx, dg, db, dy, N); },
        [&]() { layernorm_registers<N><<<M, THREADS>>>(dx, dg, db, dy); },
    };
    const size_t n4 = count / 4;
    auto unfused = [&]() {
        residual_add<<<(unsigned)((n4 + 255) / 256), 256>>>((const float4*)dx, (const float4*)dres, (float4*)dh, n4);
        layernorm_registers<N><<<M, THREADS>>>(dh, dg, db, dy);
    };
    auto fused = [&]() { add_layernorm_fused<N><<<M, THREADS>>>(dx, dres, dg, db, dh, dy); };

    if (profile) {
        for (auto& f : ln) f();
        unfused(); fused();
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        return 0;
    }

    const char* versions[] = {"v1 two-pass (3 reads)", "v2 Welford single-pass stats (2 reads)",
                              "v3 register-cached (1 read, float4)"};
    cpu_layernorm(hx.data(), hg.data(), hb.data(), href.data(), M, N);
    printf("LayerNorm %d x %d (FP32)\n", M, N);
    float base_ms = 0;
    for (int k = 0; k < 3; k++) {
        CHECK(cudaMemset(dy, 0, bytes));
        float ms = time_ms(ln[k], 50);
        if (k == 0) base_ms = ms;
        CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
        float err = max_rel_err(hy.data(), href.data(), count);
        bool ok = err < 1e-4f;
        double gbps = 2.0 * bytes / (ms / 1e3) / 1e9;          // minimum traffic: read x, write y
        printf("%-42s | %7.3f ms | %6.1f GB/s | %3.0f%% of peak | %5.2fx vs v1 | err %.1e %s\n",
               versions[k], ms, gbps, gbps / T4_PEAK_GBS * 100, base_ms / ms, err, ok ? "PASS" : "FAIL");
        report("LayerNorm", versions[k], "4096x4096", ms, gbps, "GB/s", gbps / T4_PEAK_GBS * 100, "of peak BW", ok);
    }

    // ---- Fusion: residual add + LayerNorm
    for (size_t i = 0; i < count; i++) hsum[i] = hx[i] + hres[i];
    cpu_layernorm(hsum.data(), hg.data(), hb.data(), href.data(), M, N);
    const char* fv[] = {"unfused: add kernel + LayerNorm kernel", "fused: add + LayerNorm in one kernel"};
    std::function<void()> fl[] = {unfused, fused};
    float unfused_ms = 0;
    for (int k = 0; k < 2; k++) {
        CHECK(cudaMemset(dy, 0, bytes)); CHECK(cudaMemset(dh, 0, bytes));
        float ms = time_ms(fl[k], 50);
        if (k == 0) unfused_ms = ms;
        CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hh.data(), dh, bytes, cudaMemcpyDeviceToHost));
        float err = fmaxf(max_rel_err(hy.data(), href.data(), count), max_rel_err(hh.data(), hsum.data(), count));
        bool ok = err < 1e-4f;
        // Useful traffic for the op: read x, read residual, write h, write y = 4 arrays.
        double gbps = 4.0 * bytes / (ms / 1e3) / 1e9;
        printf("%-42s | %7.3f ms | %6.1f GB/s | %5.2fx vs unfused | err %.1e %s\n",
               fv[k], ms, gbps, unfused_ms / ms, err, ok ? "PASS" : "FAIL");
        report("Residual add + LayerNorm", fv[k], "4096x4096", ms, gbps, "GB/s", gbps / T4_PEAK_GBS * 100,
               "of peak BW", ok);
    }

    cudaFree(dx); cudaFree(dres); cudaFree(dg); cudaFree(db); cudaFree(dy); cudaFree(dh);
}
