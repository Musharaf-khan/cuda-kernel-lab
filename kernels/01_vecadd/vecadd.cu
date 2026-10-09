#include <cstdio>
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    int n = 1 << 20; size_t bytes = n * sizeof(float);
    float *a, *b, *c;
    cudaMallocManaged(&a, bytes); cudaMallocManaged(&b, bytes); cudaMallocManaged(&c, bytes);
    for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }
    add<<<(n + 255) / 256, 256>>>(a, b, c, n);
    cudaDeviceSynchronize();
    bool ok = true;
    for (int i = 0; i < n; i++) if (c[i] != 3.0f) { ok = false; break; }
    printf(ok ? "PASS\n" : "FAIL\n");
}