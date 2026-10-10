#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <cublas_v2.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

const int T = 32;   // tile size / block is T x T = 1024 threads

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
            acc += As[ty][k] * Bs[k][tx];        // 32 multiply-adds from fast memory
        __syncthreads();                         // done reading before next load
    }
    C[row * n + col] = acc;
}

// ---------- harness ----------
float time_ms(std::function<void()> f, int iters) {
    f(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());   // warm-up
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) f();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    return ms / iters;
}

float max_rel_err(const float* c, const float* ref, int count) {
    float m = 0;
    for (int i = 0; i < count; i++) m = fmaxf(m, fabsf(c[i] - ref[i]) / (1.0f + fabsf(ref[i])));
    return m;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 2048;
    size_t bytes = (size_t)n * n * sizeof(float);
    double flops = 2.0 * n * n * n;
    const double PEAK = 8100.0;   // T4 FP32 peak ~8.1 TFLOPS (boost clock)

    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes), *hRef = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < n * n; i++) {
        hA[i] = rand() / (float)RAND_MAX * 2 - 1;
        hB[i] = rand() / (float)RAND_MAX * 2 - 1;
    }

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes)); CHECK(cudaMalloc(&dB, bytes)); CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // cuBLAS is column-major. Row-major C = A*B  <=>  column-major C^T = B^T * A^T,
    // so we pass B first, then A.
    cublasHandle_t h; cublasCreate(&h);
    float alpha = 1.0f, beta = 0.0f;
    auto run_cublas = [&]() {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dC, n);
    };

    // Reference result from cuBLAS
    run_cublas(); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hRef, dC, bytes, cudaMemcpyDeviceToHost));
    float cublas_ms = time_ms(run_cublas, 10);
    double cublas_gf = flops / (cublas_ms / 1e3) / 1e9;

    typedef void (*Kernel)(int, const float*, const float*, float*);
    Kernel kernels[] = {sgemm_naive_uncoalesced, sgemm_naive_coalesced, sgemm_tiled};
    const char* names[] = {"v1 naive uncoalesced", "v2 naive coalesced  ", "v3 smem tiled       "};
    dim3 block(T, T), grid(n / T, n / T);

    printf("SGEMM %dx%dx%d on T4\n", n, n, n);
    for (int k = 0; k < 3; k++) {
        CHECK(cudaMemset(dC, 0, bytes));
        float ms = time_ms([&]() { kernels[k]<<<grid, block>>>(n, dA, dB, dC); }, 5);
        CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
        float err = max_rel_err(hC, hRef, n * n);
        double gf = flops / (ms / 1e3) / 1e9;
        printf("%s | %8.2f ms | %7.1f GFLOPS | %4.1f%% of peak | %5.1f%% of cuBLAS | err %.1e %s\n",
               names[k], ms, gf, gf / PEAK * 100, gf / cublas_gf * 100, err, err < 1e-3 ? "PASS" : "FAIL");
    }
    printf("cuBLAS               | %8.2f ms | %7.1f GFLOPS | %4.1f%% of peak | 100.0%% of cuBLAS\n",
           cublas_ms, cublas_gf, cublas_gf / PEAK * 100);

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hRef);
}