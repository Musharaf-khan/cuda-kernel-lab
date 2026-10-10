#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

const int TILE = 32;   // each block handles a 32x32 tile
const int ROWS = 8;    // block is 32x8 threads; each thread handles 4 elements

// Baseline: plain coalesced copy (the speed ceiling for transpose)
__global__ void copy(float* out, const float* in, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS)
        out[(y + j) * n + x] = in[(y + j) * n + x];
}

// Naive: reads coalesced (along a row), writes strided (down a column)
__global__ void transpose_naive(float* out, const float* in, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS)
        out[x * n + (y + j)] = in[(y + j) * n + x];
}

// Tiled through shared memory. PAD=0 -> bank conflicts, PAD=1 -> fixed
template <int PAD>
__global__ void transpose_tiled(float* out, const float* in, int n) {
    __shared__ float tile[TILE][TILE + PAD];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS)
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];      // coalesced read
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;      // swap block coordinates
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS)
        out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];     // coalesced write
}

typedef void (*Kernel)(float*, const float*, int);

int main(int argc, char** argv) {
    bool profile = (argc > 1);                // "./transpose profile" -> run each kernel once
    int n = 4096;                             // must be a multiple of 32
    size_t bytes = (size_t)n * n * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for (int i = 0; i < n * n; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes)); CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, ROWS), grid(n / TILE, n / TILE);
    Kernel kernels[] = {copy, transpose_naive, transpose_tiled<0>, transpose_tiled<1>};
    const char* names[] = {"copy (ceiling)  ", "naive           ", "tiled (conflicts)", "tiled + padded  "};

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

    for (int k = 0; k < 4; k++) {
        CHECK(cudaMemset(d_out, 0, bytes));
        kernels[k]<<<grid, block>>>(d_out, d_in, n);               // warm-up
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        if (profile) continue;

        int iters = 50;
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) kernels[k]<<<grid, block>>>(d_out, d_in, n);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); ms /= iters;

        // Verify
        CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int r = 0; r < n && ok; r++)
            for (int c = 0; c < n; c++) {
                float want = (k == 0) ? h_in[r * n + c] : h_in[c * n + r];
                if (h_out[r * n + c] != want) { ok = false; break; }
            }

        double gbps = 2.0 * bytes / (ms / 1e3) / 1e9;   // read + write
        printf("%s | %6.3f ms | %6.1f GB/s | %3.0f%% of peak | %s\n",
               names[k], ms, gbps, gbps / 320.0 * 100, ok ? "PASS" : "FAIL");
    }

    cudaFree(d_in); cudaFree(d_out); free(h_in); free(h_out);
}