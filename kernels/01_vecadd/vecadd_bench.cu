// Vector add benchmark: c[i] = a[i] + b[i]
// Memory-bound: 1 FLOP per 12 bytes moved, so the only number that matters is GB/s.
#include "../../common/common.h"

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    bool profile = profile_mode(argc, argv);
    int n = 1 << 24;                       // 16M floats
    size_t bytes = n * sizeof(float);

    // Host (CPU) memory
    float *h_a = (float*)malloc(bytes), *h_b = (float*)malloc(bytes), *h_c = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    // Device (GPU) memory
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes)); CHECK(cudaMalloc(&d_b, bytes)); CHECK(cudaMalloc(&d_c, bytes));
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads = 256, blocks = (n + threads - 1) / threads;
    auto launch = [&]() { add<<<blocks, threads>>>(d_a, d_b, d_c, n); };

    if (profile) {                          // one launch for Nsight Compute
        launch(); CHECK(cudaDeviceSynchronize());
        return 0;
    }

    float ms = time_ms(launch, 100);        // warm-up + average of 100 launches

    // Bandwidth: read a, read b, write c = 3 arrays
    double gbps = 3.0 * bytes / (ms / 1e3) / 1e9;

    // Verify
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < n; i++) if (h_c[i] != 3.0f) { ok = false; break; }

    printf("%s | n=%d | %.3f ms | %.1f GB/s | %.0f%% of T4 peak (320 GB/s)\n",
           ok ? "PASS" : "FAIL", n, ms, gbps, gbps / T4_PEAK_GBS * 100);
    report("Vector add", "naive", "16M floats", ms, gbps, "GB/s", gbps / T4_PEAK_GBS * 100, "of peak BW", ok);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
}
