"""Triton ports of the softmax, LayerNorm and attention kernels, benchmarked on the same shapes.

Each port is checked against PyTorch (FP32), then timed with CUDA events after warm-up. PyTorch is timed
in the same process as a fixed reference point, so the Triton-vs-CUDA comparison in the README goes
through ratios to PyTorch measured in the same session.

    python scripts/triton_ports.py
"""
import math

import torch
import triton
import triton.language as tl

T4_PEAK_GBS = 320.0


def time_ms(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def report(kernel, version, size, ms, value, unit, pct, pct_of, ok):
    print(f"RESULT|{kernel}|{version}|{size}|{ms:.4f}|{value:.2f}|{unit}|{pct:.1f}|{pct_of}|{'PASS' if ok else 'FAIL'}")


# ------------------------------------------------------------------------------- softmax
@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))   # whole row in registers
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(y_ptr + row * n_cols + offs, y, mask=mask)


def triton_softmax(x):
    y = torch.empty_like(x)
    softmax_kernel[(x.shape[0],)](x, y, x.shape[1], BLOCK=triton.next_power_of_2(x.shape[1]), num_warps=8)
    return y


# ------------------------------------------------------------------------------- LayerNorm (+ fused add)
@triton.jit
def layernorm_kernel(x_ptr, r_ptr, w_ptr, b_ptr, h_ptr, y_ptr, n_cols, eps,
                     BLOCK: tl.constexpr, ADD_RESIDUAL: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    if ADD_RESIDUAL:
        x = x + tl.load(r_ptr + row * n_cols + offs, mask=mask, other=0.0)
        tl.store(h_ptr + row * n_cols + offs, x, mask=mask)
    mean = tl.sum(x, axis=0) / n_cols
    d = tl.where(mask, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / n_cols + eps)
    w = tl.load(w_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(y_ptr + row * n_cols + offs, d * rstd * w + b, mask=mask)


def triton_layernorm(x, w, b, residual=None, eps=1e-5):
    y = torch.empty_like(x)
    h = torch.empty_like(x) if residual is not None else y
    layernorm_kernel[(x.shape[0],)](x, residual if residual is not None else x, w, b, h, y, x.shape[1], eps,
                                    BLOCK=triton.next_power_of_2(x.shape[1]),
                                    ADD_RESIDUAL=residual is not None, num_warps=8)
    return (h, y) if residual is not None else y


# ------------------------------------------------------------------------------- attention (FlashAttention-style)
@triton.jit
def attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, n_ctx, scale_log2,
                HD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    head = tl.program_id(1)
    base = head * n_ctx * HD
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HD)
    q = tl.load(q_ptr + base + offs_m[:, None] * HD + offs_d[None, :])
    q = q * scale_log2                                     # fold log2(e)/sqrt(d) into Q, use exp2
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HD], tl.float32)
    for start_n in range(0, n_ctx, BLOCK_N):
        kt = tl.load(k_ptr + base + (start_n + offs_n)[None, :] * HD + offs_d[:, None])   # [HD, BLOCK_N]
        s = tl.dot(q, kt, input_precision="ieee")                                       # FP32, no TF32
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.math.exp2(s - m_new[:, None])
        corr = tl.math.exp2(m_i - m_new)
        l_i = l_i * corr + tl.sum(p, axis=1)
        v = tl.load(v_ptr + base + (start_n + offs_n)[:, None] * HD + offs_d[None, :])
        acc = acc * corr[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + base + offs_m[:, None] * HD + offs_d[None, :], acc)


def triton_attention(q, k, v, block_m=64, block_n=32, num_warps=4):
    b, h, n, d = q.shape
    o = torch.empty_like(q)
    attn_kernel[(n // block_m, b * h)](q, k, v, o, n, (1.0 / math.sqrt(d)) * 1.4426950408889634,
                                       HD=d, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=num_warps)
    return o


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    print(f"Triton {triton.__version__}, PyTorch {torch.__version__} on {torch.cuda.get_device_name(0)}")

    # softmax
    x = torch.randn(4096, 4096, device=dev) * 2
    ok = torch.allclose(triton_softmax(x), torch.softmax(x, -1), rtol=1e-4, atol=1e-7)
    ms_t = time_ms(lambda: triton_softmax(x))
    ms_p = time_ms(lambda: torch.softmax(x, -1))
    gbps = 2 * x.numel() * 4 / (ms_t / 1e3) / 1e9
    print(f"softmax     triton {ms_t:.3f} ms ({gbps:.1f} GB/s) | torch {ms_p:.3f} ms | {ms_p / ms_t:.2f}x vs torch | {ok}")
    report("Softmax", "Triton (row in registers)", "4096x4096", ms_t, gbps, "GB/s", gbps / T4_PEAK_GBS * 100,
           "of peak BW", ok)
    report("Softmax", "PyTorch torch.softmax (same process as Triton)", "4096x4096", ms_p,
           2 * x.numel() * 4 / (ms_p / 1e3) / 1e9, "GB/s", 0.0, "-", True)

    # LayerNorm and fused add + LayerNorm
    x = torch.randn(4096, 4096, device=dev) + 3
    r = torch.randn_like(x)
    w = 1 + 0.1 * torch.randn(4096, device=dev)
    b = 0.1 * torch.randn(4096, device=dev)
    ref = torch.nn.functional.layer_norm(x, (4096,), w, b, 1e-5)
    ok = torch.allclose(triton_layernorm(x, w, b), ref, rtol=1e-4, atol=1e-4)
    ms_t = time_ms(lambda: triton_layernorm(x, w, b))
    ms_p = time_ms(lambda: torch.nn.functional.layer_norm(x, (4096,), w, b, 1e-5))
    gbps = 2 * x.numel() * 4 / (ms_t / 1e3) / 1e9
    print(f"layernorm   triton {ms_t:.3f} ms ({gbps:.1f} GB/s) | torch {ms_p:.3f} ms | {ms_p / ms_t:.2f}x vs torch | {ok}")
    report("LayerNorm", "Triton (row in registers)", "4096x4096", ms_t, gbps, "GB/s", gbps / T4_PEAK_GBS * 100,
           "of peak BW", ok)
    report("LayerNorm", "PyTorch F.layer_norm (same process as Triton)", "4096x4096", ms_p,
           2 * x.numel() * 4 / (ms_p / 1e3) / 1e9, "GB/s", 0.0, "-", True)

    h_t, y_t = triton_layernorm(x, w, b, residual=r)
    ok = torch.allclose(h_t, x + r) and torch.allclose(
        y_t, torch.nn.functional.layer_norm(x + r, (4096,), w, b, 1e-5), rtol=1e-4, atol=1e-4)
    ms_t = time_ms(lambda: triton_layernorm(x, w, b, residual=r))
    ms_p = time_ms(lambda: torch.nn.functional.layer_norm(x + r, (4096,), w, b, 1e-5))
    gbps = 4 * x.numel() * 4 / (ms_t / 1e3) / 1e9
    print(f"add+LN      triton {ms_t:.3f} ms ({gbps:.1f} GB/s) | torch eager {ms_p:.3f} ms | {ms_p / ms_t:.2f}x | {ok}")
    report("Residual add + LayerNorm", "Triton fused add + LayerNorm", "4096x4096", ms_t, gbps, "GB/s",
           gbps / T4_PEAK_GBS * 100, "of peak BW", ok)

    # attention
    for n in [1024, 2048, 4096, 8192, 16384]:
        q = torch.rand(2, 8, n, 64, device=dev) * 2 - 1
        k = torch.rand_like(q) * 2 - 1
        v = torch.rand_like(q) * 2 - 1
        flops = 4.0 * 16 * n * n * 64
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        best = None
        for bm, bn, nw in [(64, 32, 4), (64, 64, 4), (128, 32, 8), (32, 32, 4)]:
            try:
                out = triton_attention(q, k, v, bm, bn, nw)
                ok = torch.allclose(out, ref, rtol=1e-3, atol=1e-4)
                ms = time_ms(lambda: triton_attention(q, k, v, bm, bn, nw), iters=3 if n >= 8192 else 10)
            except Exception as e:  # e.g. out of shared memory for a block shape
                print(f"attn N={n} blocks ({bm},{bn},{nw}) failed: {type(e).__name__}")
                continue
            if ok and (best is None or ms < best[0]):
                best = (ms, bm, bn, nw)
        ms_s = time_ms(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
                       iters=3 if n >= 8192 else 10)
        if best:
            ms, bm, bn, nw = best
            gf = flops / (ms / 1e3) / 1e9
            print(f"attn N={n:<6} triton {ms:9.3f} ms ({gf:7.1f} GFLOPS, blocks {bm}x{bn}, {nw} warps) | "
                  f"SDPA {ms_s:9.3f} ms | {ms_s / ms:.2f}x vs SDPA")
            report("Attention (BH=16, d=64)", f"Triton FlashAttention-style (best of 4 block shapes: {bm}x{bn})",
                   f"N={n}", ms, gf, "GFLOPS", 0.0, "-", True)
        else:
            print(f"attn N={n} triton: no correct configuration")
            report("Attention (BH=16, d=64)", "Triton FlashAttention-style", f"N={n}", 0, 0, "GFLOPS", 0.0, "-", False)
        report("Attention (BH=16, d=64)", "PyTorch SDPA (same process as Triton)", f"N={n}", ms_s,
               flops / (ms_s / 1e3) / 1e9, "GFLOPS", 0.0, "-", True)
        del q, k, v, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
