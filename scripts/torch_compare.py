"""PyTorch baselines on the same shapes as the CUDA benchmarks.

Prints the same RESULT|... lines as the .cu benchmarks so make_table.py can merge them.
Timed with CUDA events after warm-up, like everything else in this repo.

    python scripts/torch_compare.py
"""
import math

import torch
import torch.nn.functional as F

T4_PEAK_GBS = 320.0


def time_ms(fn, iters=50):
    fn()  # warm-up
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def report(kernel, version, size, ms, value, unit, pct, pct_of, status="PASS"):
    print(f"RESULT|{kernel}|{version}|{size}|{ms:.4f}|{value:.2f}|{unit}|{pct:.1f}|{pct_of}|{status}")


def main():
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False  # T4 has no TF32 anyway; keep it explicit
    print(f"PyTorch {torch.__version__} on {torch.cuda.get_device_name(0)}")

    # ---- Softmax 4096 x 4096
    x = torch.randn(4096, 4096, device=dev) * 2
    ms = time_ms(lambda: torch.softmax(x, dim=-1))
    gbps = 2 * x.numel() * 4 / (ms / 1e3) / 1e9
    print(f"softmax                  {ms:8.3f} ms  {gbps:6.1f} GB/s")
    report("Softmax", "PyTorch torch.softmax", "4096x4096", ms, gbps, "GB/s", gbps / T4_PEAK_GBS * 100, "of peak BW")

    # ---- LayerNorm 4096 x 4096
    x = torch.randn(4096, 4096, device=dev) + 3
    res = torch.randn_like(x)
    w = torch.ones(4096, device=dev)
    b = torch.zeros(4096, device=dev)
    ms = time_ms(lambda: F.layer_norm(x, (4096,), w, b, 1e-5))
    gbps = 2 * x.numel() * 4 / (ms / 1e3) / 1e9
    print(f"layer_norm               {ms:8.3f} ms  {gbps:6.1f} GB/s")
    report("LayerNorm", "PyTorch F.layer_norm", "4096x4096", ms, gbps, "GB/s", gbps / T4_PEAK_GBS * 100, "of peak BW")

    def add_ln():
        h = x + res
        return h, F.layer_norm(h, (4096,), w, b, 1e-5)

    ms = time_ms(add_ln)
    gbps = 4 * x.numel() * 4 / (ms / 1e3) / 1e9
    print(f"x + res, layer_norm      {ms:8.3f} ms  {gbps:6.1f} GB/s (PyTorch eager = unfused)")
    report("Residual add + LayerNorm", "PyTorch eager (x + res, F.layer_norm)", "4096x4096", ms, gbps, "GB/s",
           gbps / T4_PEAK_GBS * 100, "of peak BW")

    # ---- SGEMM 4096^3 (torch.mm calls cuBLAS)
    a = torch.rand(4096, 4096, device=dev) * 2 - 1
    bm = torch.rand(4096, 4096, device=dev) * 2 - 1
    ms = time_ms(lambda: torch.mm(a, bm), iters=10)
    gf = 2 * 4096**3 / (ms / 1e3) / 1e9
    print(f"torch.mm 4096^3          {ms:8.3f} ms  {gf:7.1f} GFLOPS")
    report("SGEMM", "PyTorch torch.mm (cuBLAS)", "4096^3", ms, gf, "GFLOPS", 0.0, "-")

    # ---- Attention: scaled_dot_product_attention in FP32
    BH, D = 16, 64
    for n in [1024, 2048, 4096, 8192, 16384]:
        q = torch.rand(2, 8, n, D, device=dev) * 2 - 1
        k = torch.rand_like(q) * 2 - 1
        v = torch.rand_like(q) * 2 - 1
        flops = 4.0 * BH * n * n * D
        try:
            ms = time_ms(lambda: F.scaled_dot_product_attention(q, k, v), iters=3 if n >= 8192 else 10)
            gf = flops / (ms / 1e3) / 1e9
            print(f"SDPA N={n:<6}             {ms:8.3f} ms  {gf:7.1f} GFLOPS")
            report("Attention (BH=16, d=64)", "PyTorch F.scaled_dot_product_attention (FP32)", f"N={n}",
                   ms, gf, "GFLOPS", 0.0, "-")
        except torch.OutOfMemoryError:
            print(f"SDPA N={n:<6}             OOM")
            report("Attention (BH=16, d=64)", "PyTorch F.scaled_dot_product_attention (FP32)", f"N={n}",
                   0, 0, "GFLOPS", 0.0, "-", "OOM")
        del q, k, v
        torch.cuda.empty_cache()

        # naive PyTorch attention (materializes the N x N matrix) for comparison
        if n <= 8192:
            q = torch.rand(2, 8, n, D, device=dev)
            k, v = torch.rand_like(q), torch.rand_like(q)
            try:
                naive = lambda: torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(D), dim=-1) @ v
                ms = time_ms(naive, iters=3 if n >= 8192 else 10)
                gf = flops / (ms / 1e3) / 1e9
                print(f"naive torch N={n:<6}      {ms:8.3f} ms  {gf:7.1f} GFLOPS")
                report("Attention (BH=16, d=64)", "PyTorch naive (matmul, softmax, matmul)", f"N={n}",
                       ms, gf, "GFLOPS", 0.0, "-")
            except torch.OutOfMemoryError:
                report("Attention (BH=16, d=64)", "PyTorch naive (matmul, softmax, matmul)", f"N={n}",
                       0, 0, "GFLOPS", 0.0, "-", "OOM")
            del q, k, v
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
