// Shared helpers for every benchmark in this repo.
//
// - CHECK(...)      : abort with file/line on any CUDA error
// - time_ms(f, n)   : warm up once, then average n launches using CUDA events
// - report(...)     : print one machine-readable "RESULT|..." line that
//                     scripts/make_table.py turns into benchmarks/results.md
// - profile_mode()  : "./bench profile" runs each kernel exactly once so that
//                     Nsight Compute (ncu) profiles stay fast
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <functional>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// NVIDIA Tesla T4 (Turing, sm_75) spec-sheet ceilings.
// Colab's T4 is power-capped at 70 W, so sustained numbers sit below these.
constexpr double T4_PEAK_GBS    = 320.0;    // GDDR6 bandwidth, GB/s
constexpr double T4_PEAK_GFLOPS = 8100.0;   // FP32 FMA throughput, GFLOP/s (1590 MHz boost)

inline float time_ms(const std::function<void()>& f, int iters) {
    f();                                         // warm-up: first launch pays one-time costs
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; i++) f();
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    CHECK(cudaGetLastError());
    float ms; CHECK(cudaEventElapsedTime(&ms, s, e));
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;
}

inline bool profile_mode(int argc, char** argv) {
    for (int i = 1; i < argc; i++) if (strcmp(argv[i], "profile") == 0) return true;
    return false;
}

// RESULT|kernel|version|size|time_ms|value|unit|pct|pct_of|status
inline void report(const char* kernel, const char* version, const char* size,
                   float ms, double value, const char* unit,
                   double pct, const char* pct_of, bool pass) {
    printf("RESULT|%s|%s|%s|%.4f|%.2f|%s|%.1f|%s|%s\n",
           kernel, version, size, ms, value, unit, pct, pct_of, pass ? "PASS" : "FAIL");
}

// Max relative error that treats any NaN/Inf as an automatic failure.
// (fmaxf silently ignores NaN, so a naive max() would report 0 = "PASS".)
inline float max_rel_err(const float* got, const float* ref, size_t count) {
    float m = 0.0f;
    for (size_t i = 0; i < count; i++) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) return INFINITY;
        m = fmaxf(m, fabsf(got[i] - ref[i]) / (1.0f + fabsf(ref[i])));
    }
    return m;
}
