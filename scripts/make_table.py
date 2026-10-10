"""Turn the RESULT|... lines printed by every benchmark into benchmarks/results.md.

    python scripts/make_table.py benchmarks/raw  > benchmarks/results.md

Rules
- Lines are grouped by (kernel, size), in the order they first appear.
- If the same (kernel, version, size) appears in several runs (e.g. GEMM run 3 times),
  the time/throughput columns come from the run with the median time, and the "% of" column
  (a same-run ratio, e.g. % of cuBLAS) is the median of the per-run ratios.
- "Speedup" is relative to the first version listed in that group (the baseline).
"""
import glob
import os
import statistics
import sys
from collections import OrderedDict


def parse(raw_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.txt"))):
        with open(path, errors="replace") as f:
            for line in f:
                if not line.startswith("RESULT|"):
                    continue
                parts = line.rstrip("\n").split("|")
                if len(parts) != 10:
                    continue
                _, kernel, version, size, ms, value, unit, pct, pct_of, status = parts
                rows.append(dict(kernel=kernel, version=version, size=size, ms=float(ms), value=float(value),
                                 unit=unit, pct=float(pct), pct_of=pct_of, status=status, src=os.path.basename(path)))
    return rows


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/raw"
    rows = parse(raw_dir)
    if not rows:
        print("_No RESULT lines found. Run scripts/run_all.sh on a GPU first._")
        return

    # (kernel, size) -> version -> [rows]
    groups = OrderedDict()
    for r in rows:
        groups.setdefault((r["kernel"], r["size"]), OrderedDict()).setdefault(r["version"], []).append(r)

    print("| Kernel | Size | Version | Time (ms) | Throughput | Speedup | % of | Check |")
    print("|---|---|---|---:|---:|---:|---:|---|")
    for (kernel, size), versions in groups.items():
        # Baseline = first version that isn't a "(ceiling)" reference. If the baseline didn't run
        # (e.g. out of memory), there is no speedup to report for that group.
        base_name = next((v for v in versions if "ceiling" not in v), None)
        base_runs = [r for r in versions.get(base_name, []) if r["ms"] > 0]
        base_ms = statistics.median_low([r["ms"] for r in base_runs]) if base_runs else None
        for version, runs in versions.items():
            ok_runs = [r for r in runs if r["ms"] > 0]
            if ok_runs:
                med_ms = statistics.median_low([r["ms"] for r in ok_runs])
                r = next(x for x in ok_runs if x["ms"] == med_ms)
            else:
                r = runs[0]
            n = len(runs)
            if r["ms"] > 0:
                speed = f"{base_ms / r['ms']:.2f}x" if base_ms else "-"
                tp = f"{r['value']:,.1f} {r['unit']}"
                ms = f"{r['ms']:.3f}"
                # the % column is a same-run ratio, so take the median of the per-run ratios
                pct_v = statistics.median([x["pct"] for x in ok_runs]) if ok_runs else r["pct"]
                pct = f"{pct_v:.1f}% {r['pct_of']}" if r["pct_of"] != "-" and pct_v > 0 else "-"
            else:
                speed, tp, ms, pct = "-", "-", "-", "-"
            check = r["status"] + (f" (median of {n})" if n > 1 else "")
            if version.startswith("PyTorch"):
                check = "baseline (timing only)"
            print(f"| {kernel} | {size} | {version} | {ms} | {tp} | {speed} | {pct} | {check} |")


if __name__ == "__main__":
    main()
