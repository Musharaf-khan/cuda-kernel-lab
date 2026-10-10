#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
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

    // Warm-up run (first launch is always slower)
    add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());

    // Time 100 runs with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int iters = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    // Bandwidth: read a, read b, write c = 3 arrays
    double gbps = 3.0 * bytes / (ms / 1e3) / 1e9;

    // Verify
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < n; i++) if (h_c[i] != 3.0f) { ok = false; break; }

    printf("%s | n=%d | %.3f ms | %.1f GB/s | %.0f%% of T4 peak (320 GB/s)\n",
           ok ? "PASS" : "FAIL", n, ms, gbps, gbps / 320.0 * 100);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
}