// Parallel reduction (sum of 16M floats): three versions, each faster than the last.
#include "../../common/common.h"

// ---------- v1: every thread does an atomicAdd to one global value ----------
__global__ void reduce_atomic(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, in[i]);
}

// ---------- v2: shared-memory tree reduction, one atomic per block ----------
__global__ void reduce_shared(const float* in, float* out, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();                       // everyone finishes this level before the next
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// ---------- v3: grid-stride loop + warp shuffle ----------
__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);   // pull value from lane (me + offset)
    return v;
}

__global__ void reduce_warp(const float* in, float* out, int n) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        sum += in[i];                          // each thread sums many elements in a register

    sum = warp_sum(sum);                       // reduce within each warp of 32

    __shared__ float warp_sums[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp] = sum;      // one value per warp
    __syncthreads();

    if (warp == 0) {                           // first warp reduces the per-warp results
        sum = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) atomicAdd(out, sum);
    }
}

// ---------- benchmark harness ----------
float bench(int version, const float* d_in, float* d_out, int n, float* result, bool profile) {
    int threads = 256, blocks = (n + threads - 1) / threads;
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);

    auto launch = [&]() {
        cudaMemset(d_out, 0, sizeof(float));
        if (version == 1) reduce_atomic<<<blocks, threads>>>(d_in, d_out, n);
        if (version == 2) reduce_shared<<<blocks, threads>>>(d_in, d_out, n);
        if (version == 3) reduce_warp<<<sms * 8, threads>>>(d_in, d_out, n);
    };

    if (profile) {                             // one launch for Nsight Compute
        launch(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        return 0.0f;
    }

    // Note: the timed region includes a 4-byte cudaMemset per launch (negligible).
    float ms = time_ms(launch, 20);

    CHECK(cudaMemcpy(result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    return ms;
}

int main(int argc, char** argv) {
    bool profile = profile_mode(argc, argv);
    int n = 1 << 24;                           // 16M floats
    size_t bytes = n * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;   // all ones -> exact expected sum = n
    float expected = (float)n;

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes)); CHECK(cudaMalloc(&d_out, sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    const char* names[] = {"", "v1 atomic      ", "v2 shared tree ", "v3 warp shuffle"};
    float base_ms = 0;
    for (int v = 1; v <= 3; v++) {
        float sum;
        float ms = bench(v, d_in, d_out, n, &sum, profile);
        if (profile) continue;
        if (v == 1) base_ms = ms;
        double gbps = bytes / (ms / 1e3) / 1e9;
        printf("%s | %8.3f ms | %6.1f GB/s | %3.0f%% of peak | %5.1fx vs v1 | sum=%.0f %s\n",
               names[v], ms, gbps, gbps / 320.0 * 100, base_ms / ms, sum,
               sum == expected ? "PASS" : "FAIL");
        const char* ver[] = {"", "v1 atomicAdd per element", "v2 shared-mem tree", "v3 grid-stride + warp shuffle"};
        report("Reduction (sum)", ver[v], "16M floats", ms, gbps, "GB/s", gbps / T4_PEAK_GBS * 100,
               "of peak BW", sum == expected);
    }

    cudaFree(d_in); cudaFree(d_out); free(h_in);
}