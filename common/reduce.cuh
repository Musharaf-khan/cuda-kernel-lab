// Warp- and block-level reductions used by softmax, LayerNorm and attention.
//
// Pattern (same as kernels/02_reduction v3):
//   1. each warp reduces its 32 values with register shuffles (no shared memory, no barrier)
//   2. lane 0 of each warp writes one partial to shared memory
//   3. warp 0 reduces the partials, and the result is broadcast to every thread
#pragma once
#include <cmath>

constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);   // xor: every lane gets the total
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, o));
    return v;
}

// Every thread of the block must call these (they contain __syncthreads()).
// blockDim.x must be a multiple of 32 and <= 1024.
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float partial[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    v = warp_sum(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < nwarps) ? partial[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) partial[0] = v;
    }
    __syncthreads();
    v = partial[0];
    __syncthreads();                       // partial[] may be reused by the next call
    return v;
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float partial[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    v = warp_max(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < nwarps) ? partial[lane] : -INFINITY;
        v = warp_max(v);
        if (lane == 0) partial[0] = v;
    }
    __syncthreads();
    v = partial[0];
    __syncthreads();
    return v;
}

// ---- Online-softmax state: (running max m, running normalizer l = sum exp(x - m)) ----
// Merging two partial states is associative, which is what lets us reduce them in a tree.
__device__ __forceinline__ void online_merge(float& m, float& l, float m2, float l2) {
    const float mn = fmaxf(m, m2);
    if (mn == -INFINITY) return;           // both states empty
    l = l * __expf(m - mn) + l2 * __expf(m2 - mn);   // rescale both to the new max
    m = mn;
}

__device__ __forceinline__ void block_online_merge(float& m, float& l) {
    __shared__ float pm[32], pl[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        online_merge(m, l, __shfl_xor_sync(FULL_MASK, m, o), __shfl_xor_sync(FULL_MASK, l, o));
    if (lane == 0) { pm[warp] = m; pl[warp] = l; }
    __syncthreads();
    if (warp == 0) {
        m = (lane < nwarps) ? pm[lane] : -INFINITY;
        l = (lane < nwarps) ? pl[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            online_merge(m, l, __shfl_xor_sync(FULL_MASK, m, o), __shfl_xor_sync(FULL_MASK, l, o));
        if (lane == 0) { pm[0] = m; pl[0] = l; }
    }
    __syncthreads();
    m = pm[0]; l = pl[0];
    __syncthreads();
}
