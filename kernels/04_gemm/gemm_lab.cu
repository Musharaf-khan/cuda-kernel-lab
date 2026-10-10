// SGEMM experiments: every attempt to close the gap to cuBLAS, measured in the same run.
//
//   ./gemm_lab attempts [n]     Part B: v5 re-expressed, each single change from v5, then the cumulative
//                               ladder. cuBLAS is timed before AND after; ratios use the average.
//   ./gemm_lab tune [n]         Part C: every autotuner candidate that fits n (TUNE|... lines)
//   ./gemm_lab profile [n]      every attempt once + cuBLAS once, for Nsight Compute
#include "../../common/common.h"
#include "sgemm_lab.cuh"
#include <cublas_v2.h>
#include <string>
#include <vector>

struct Attempt { const char* config; const char* what; };
static const Attempt kAttempts[] = {
    {"lab_v5",           "A0 v5 re-expressed in the lab kernel (baseline)"},
    {"v5_warptile",      "B1 v5 + warp tiling (64x32 warp tile, 2 sub-tiles)"},
    {"v5_dbuf",          "B2 v5 + register-prefetch double buffering"},
    {"v5_128thr",        "B3a v5 with 128 threads (8x16 per thread)"},
    {"v5_512thr",        "B3b v5 with 512 threads (4x8 per thread)"},
    {"v5_lb2",           "B3c v5 + __launch_bounds__(256, 2)"},
    {"v5_padA",          "B4 v5 + padded transposed A tile (stride 132)"},
    {"v5_bk16",          "B5 v5 + BK=16 (half the barriers per FLOP)"},
    {"c2_warp_dbuf",     "C2 warp tiling + double buffering"},
    {"c3_warp_dbuf_pad", "C3 C2 + padded A tile"},
    {"c4_wdp_bk16",      "C4 C3 + BK=16"},
    {"c4b_wd_bk16_nopad","C4b C4 without padding (32 KB smem fits 2 blocks/SM)"},
    {"c5_wdp_bk16_lb2",  "C5 C4 + __launch_bounds__(256, 2)"},
};

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "attempts";
    int n = argc > 2 ? atoi(argv[2]) : 4096;
    size_t count = (size_t)n * n, bytes = count * sizeof(float);
    double flops = 2.0 * n * n * n;

    std::vector<float> hA(count), hB(count), hC(count), hRef(count);
    srand(42);
    for (size_t i = 0; i < count; i++) {
        hA[i] = rand() / (float)RAND_MAX * 2 - 1;
        hB[i] = rand() / (float)RAND_MAX * 2 - 1;
    }
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes)); CHECK(cudaMalloc(&dB, bytes)); CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t h; cublasCreate(&h);
    float alpha = 1.0f, beta = 0.0f;
    auto run_cublas = [&]() {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dC, n);
    };

    std::vector<std::string> names;
    if (mode == "tune") {
        for (int i = 0; i < kNumLabConfigs; i++)
            if (std::string(kLabConfigs[i].name).rfind("t", 0) == 0 || std::string(kLabConfigs[i].name) == "lab_v5")
                names.push_back(kLabConfigs[i].name);
    } else {
        for (auto& a : kAttempts) names.push_back(a.config);
    }

    if (mode == "profile") {
        for (auto& nm : names) { launch_lab(nm.c_str(), n, dA, dB, dC); CHECK(cudaGetLastError()); }
        run_cublas();
        CHECK(cudaDeviceSynchronize());
        return 0;
    }

    run_cublas(); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hRef.data(), dC, bytes, cudaMemcpyDeviceToHost));

    int iters = n >= 4096 ? 5 : n >= 2048 ? 20 : 50;
    float cub_before = time_ms(run_cublas, 4 * iters);
    std::vector<float> ms(names.size(), 0.0f), err(names.size(), INFINITY);
    for (size_t k = 0; k < names.size(); k++) {
        if (!launch_lab(names[k].c_str(), n, dA, dB, dC)) { ms[k] = -1; continue; }   // doesn't fit n
        CHECK(cudaMemset(dC, 0, bytes));
        ms[k] = time_ms([&]() { launch_lab(names[k].c_str(), n, dA, dB, dC); }, iters);
        CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        err[k] = max_rel_err(hC.data(), hRef.data(), count);
    }
    float cub_after = time_ms(run_cublas, 4 * iters);
    float cub_ms = 0.5f * (cub_before + cub_after);
    double cub_gf = flops / (cub_ms / 1e3) / 1e9;

    char size[32]; snprintf(size, sizeof size, "%d^3", n);
    printf("SGEMM %s %d^3   cuBLAS %.3f ms (before %.3f, after %.3f) = %.1f GFLOPS\n",
           mode.c_str(), n, cub_ms, cub_before, cub_after, cub_gf);
    for (size_t k = 0; k < names.size(); k++) {
        if (ms[k] < 0) continue;
        double gf = flops / (ms[k] / 1e3) / 1e9;
        bool ok = err[k] < 1e-3f;
        const char* what = names[k].c_str();
        for (auto& a : kAttempts) if (names[k] == a.config) what = a.what;
        printf("%-52s | %8.3f ms | %7.1f GFLOPS | %5.1f%% of cuBLAS | err %.1e %s\n",
               what, ms[k], gf, gf / cub_gf * 100, err[k], ok ? "PASS" : "FAIL");
        if (mode == "tune")
            printf("TUNE|%s|%d|%.4f|%.2f|%.2f|%s\n", names[k].c_str(), n, ms[k], gf, gf / cub_gf * 100, ok ? "PASS" : "FAIL");
        else
            report("SGEMM attempts", what, size, ms[k], gf, "GFLOPS", gf / cub_gf * 100, "of cuBLAS", ok);
    }
    if (mode == "tune") printf("TUNE|cuBLAS|%d|%.4f|%.2f|100.00|PASS\n", n, cub_ms, cub_gf);
    else report("SGEMM attempts", "cuBLAS (same run)", size, cub_ms, cub_gf, "GFLOPS", 100.0, "of cuBLAS", true);

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
