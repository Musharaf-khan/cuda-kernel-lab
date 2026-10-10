"""Roofline chart from Nsight Compute hardware counters.

Each benchmark is run once per kernel under ncu ("./bench profile"), collecting:
    FLOPs  = 2 * FFMA + FADD + FMUL      (thread-level FP32 instructions actually executed)
    bytes  = DRAM bytes read + written   (real HBM/GDDR traffic, after L1/L2 caching)
    time   = gpu__time_duration
so every point is MEASURED, not estimated:
    x = arithmetic intensity  = FLOPs / DRAM byte
    y = achieved performance  = FLOPs / second

    python scripts/roofline.py benchmarks/ncu  benchmarks/roofline.png  benchmarks/roofline.csv
"""
import csv
import glob
import io
import os
import sys
from collections import OrderedDict

PEAK_GBS = 320.0        # T4 GDDR6, GB/s
PEAK_GFLOPS = 8100.0    # T4 FP32 at 1590 MHz boost, GFLOP/s

METRICS = {
    "dram__bytes_read.sum": "rd",
    "dram__bytes_write.sum": "wr",
    "gpu__time_duration.sum": "ns",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum": "ffma",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum": "fadd",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum": "fmul",
}
UNIT_SCALE = {"byte": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9,
              "nsecond": 1, "usecond": 1e3, "msecond": 1e6, "second": 1e9,
              "inst": 1, "": 1}

# Launch order inside each "./bench profile" run -> label. The last entry labels all remaining launches
# (e.g. every kernel cuBLAS launches internally).
LABELS = {
    "vecadd":    (["Vector add"], None),
    "reduction": (["Reduce v1 atomic", "Reduce v2 smem tree", "Reduce v3 warp shuffle"], None),
    "gemm":      (["GEMM v1 naive", "GEMM v2 coalesced", "GEMM v3 smem tiled",
                   "GEMM v4 register tiled", "GEMM v5 vectorized"], "cuBLAS"),
    "softmax":   (["Softmax v1 thread/row", "Softmax v2 3-pass", "Softmax v3 online", "Softmax v4 registers"], None),
    "layernorm": (["LayerNorm v1 two-pass", "LayerNorm v2 Welford", "LayerNorm v3 registers",
                   "Add+LN unfused", "Add+LN unfused", "Add+LN fused"], None),
    "attention": (["Attention fused"], "Attention unfused"),
}
FAMILY = {"gemm": "Matrix multiply (SGEMM)", "attention": "Attention"}
FAMILY_DEFAULT = "Memory-bound ops"
# Reference categorical palette, slots 1-3 (validated all-pairs on the light surface).
COLORS = {"Matrix multiply (SGEMM)": "#2a78d6", "Memory-bound ops": "#eb6834", "Attention": "#1baf7a"}


def read_ncu_csv(path):
    with open(path, errors="replace") as f:
        lines = f.readlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        return []
    text = "".join(l for l in lines[start:] if l.startswith('"'))
    launches = OrderedDict()
    for row in csv.DictReader(io.StringIO(text)):
        key = METRICS.get(row.get("Metric Name", ""))
        if key is None:
            continue
        try:
            val = float(row["Metric Value"].replace(",", ""))
        except ValueError:
            continue
        val *= UNIT_SCALE.get(row.get("Metric Unit", "").strip().lower(), 1)
        launches.setdefault(row["ID"], {"name": row.get("Kernel Name", "?")})[key] = val
    return list(launches.values())


def collect(ncu_dir):
    points = OrderedDict()
    for bench, (names, rest) in LABELS.items():
        path = os.path.join(ncu_dir, f"{bench}_roofline.csv")
        if not os.path.exists(path):
            continue
        for i, k in enumerate(read_ncu_csv(path)):
            label = names[i] if i < len(names) else rest
            if label is None:
                continue
            p = points.setdefault(label, dict(family=FAMILY.get(bench, FAMILY_DEFAULT), flops=0.0, bytes=0.0, ns=0.0))
            p["flops"] += 2 * k.get("ffma", 0) + k.get("fadd", 0) + k.get("fmul", 0)
            p["bytes"] += k.get("rd", 0) + k.get("wr", 0)
            p["ns"] += k.get("ns", 0)
    out = []
    for label, p in points.items():
        if p["flops"] <= 0 or p["bytes"] <= 0 or p["ns"] <= 0:
            continue
        out.append(dict(label=label, family=p["family"], oi=p["flops"] / p["bytes"],
                        gflops=p["flops"] / p["ns"], gbs=p["bytes"] / p["ns"]))
    return out


# Main chart: the full GEMM path, attention, and the best version of each memory-bound op.
# Every memory-bound version is shown in the zoomed second chart.
MAIN_MEMORY = {"Vector add", "Reduce v3 warp shuffle", "Softmax v4 registers", "LayerNorm v3 registers", "Add+LN fused"}


def plot(points, png, xlim=(0.01, 1000), ylim=(1, 20000), title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    fig.patch.set_facecolor(surface); ax.set_facecolor(surface)

    xs = np.logspace(-2, 3, 400)
    ridge = PEAK_GFLOPS / PEAK_GBS
    ax.plot(xs, np.minimum(xs * PEAK_GBS, PEAK_GFLOPS), color=ink2, lw=2, zorder=1)
    ax.axvline(ridge, color=grid, lw=1, ls="--", zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    x0 = xlim[0] * 1.3
    if x0 * PEAK_GBS > ylim[0]:
        ax.text(x0, x0 * PEAK_GBS * 1.3, f"memory roof  {PEAK_GBS:.0f} GB/s", color=ink2, fontsize=9,
                rotation=33, rotation_mode="anchor")
    if PEAK_GFLOPS < ylim[1]:
        ax.text(xlim[1] / 4, PEAK_GFLOPS * 1.12, f"FP32 roof  {PEAK_GFLOPS / 1000:.1f} TFLOP/s", color=ink2,
                fontsize=9, ha="right")
    if xlim[0] < ridge < xlim[1]:
        ax.text(ridge * 1.08, ylim[0] * 2, f"ridge point\n{ridge:.1f} FLOP/byte", color=ink2, fontsize=8)
    for fam, color in COLORS.items():
        fp = [p for p in points if p["family"] == fam]
        if not fp:
            continue
        ax.scatter([p["oi"] for p in fp], [p["gflops"] for p in fp], s=70, color=color, label=fam,
                   edgecolors=surface, linewidths=2, zorder=3)

    # Direct labels, nudged down until they don't collide (points cluster near the compute roof).
    fig.canvas.draw()
    placed = []                                    # label anchors in points: (x0, y_center)
    for p in sorted(points, key=lambda q: -q["gflops"]):
        px, py = ax.transData.transform((p["oi"], p["gflops"])) * 72.0 / fig.dpi   # pixels -> points
        dy = -3.0
        while any(abs((px + 7) - bx) < 130 and abs((py + dy) - by) < 11 for bx, by in placed):
            dy -= 11.0
        placed.append((px + 7, py + dy))
        ax.annotate(p["label"], (p["oi"], p["gflops"]), xytext=(7, dy), textcoords="offset points",
                    fontsize=7.5, color=ink, zorder=4, va="center",
                    arrowprops=dict(arrowstyle="-", color=grid, lw=0.6) if dy < -3 else None)

    gemm = [p for p in points if p["label"].startswith("GEMM v")]
    for a, b in zip(gemm, gemm[1:]):          # the GEMM optimization path
        ax.annotate("", xy=(b["oi"], b["gflops"]), xytext=(a["oi"], a["gflops"]),
                    arrowprops=dict(arrowstyle="->", color=COLORS["Matrix multiply (SGEMM)"], lw=1.2, alpha=0.6),
                    zorder=2)

    ax.set_xlabel("Arithmetic intensity (FLOP per DRAM byte, measured)", color=ink)
    ax.set_ylabel("Achieved performance (GFLOP/s, measured)", color=ink)
    ax.set_title(title or "CUDA Kernel Lab on a Tesla T4: measured roofline (Nsight Compute counters)",
                 color=ink, fontsize=11, loc="left")
    ax.grid(True, which="major", color=grid, lw=0.6)
    for s in ax.spines.values():
        s.set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(png, facecolor=surface)
    print(f"wrote {png}")


def main():
    ncu_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/ncu"
    png = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/roofline.png"
    csv_out = sys.argv[3] if len(sys.argv) > 3 else "benchmarks/roofline.csv"
    points = collect(ncu_dir)
    if not points:
        print("no ncu roofline data found in", ncu_dir)
        return
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "family", "arithmetic_intensity_flop_per_byte", "achieved_gflops", "dram_gbs",
                    "bound_by_roofline"])
        for p in points:
            bound = "memory" if p["oi"] < PEAK_GFLOPS / PEAK_GBS else "compute"
            w.writerow([p["label"], p["family"], f"{p['oi']:.3f}", f"{p['gflops']:.1f}", f"{p['gbs']:.1f}", bound])
            print(f"{p['label']:<26} OI {p['oi']:9.3f} FLOP/B  {p['gflops']:9.1f} GFLOP/s  {p['gbs']:7.1f} GB/s  {bound}")
    main_pts = [p for p in points if p["family"] != FAMILY_DEFAULT or p["label"] in MAIN_MEMORY]
    plot(main_pts, png)
    mem = [p for p in points if p["family"] == FAMILY_DEFAULT]
    if mem:
        plot(mem, png.replace(".png", "_memory.png"), xlim=(0.05, 5), ylim=(10, 2000),
             title="Memory-bound kernels, every version (zoom of the memory roof)")


if __name__ == "__main__":
    main()
