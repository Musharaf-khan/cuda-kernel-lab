"""SGEMM autotuner: pick the fastest tile configuration for each problem size.

Runs `build/gemm_lab tune <n>` (every candidate config in kernels/04_gemm/sgemm_lab.cuh that fits n,
each checked against cuBLAS, cuBLAS timed before and after in the same run) REPEATS times per size,
takes the median GFLOPS per config, and writes configs/sgemm_autotune.json:

    {"gpu": ..., "choices": {"256": "t64x64_8x8", "1024": ..., ...}, "results": {...}}

`build/gemm` reads "choices" and runs the chosen config as "v6 autotuned": for a given n it uses the
choice for the smallest tuned size >= n (or the largest tuned size).

    python scripts/autotune.py [--sizes 256 1024 2048 4096] [--repeats 3]
"""
import argparse
import json
import os
import statistics
import subprocess
from collections import defaultdict


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 1024, 2048, 4096])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--binary", default="build/gemm_lab")
    ap.add_argument("--out", default="configs/sgemm_autotune.json")
    ap.add_argument("--log", default=None, help="also append raw tuner output here")
    args = ap.parse_args()

    results = {}
    choices = {}
    log = open(args.log, "a") if args.log else None
    for n in args.sizes:
        gflops = defaultdict(list)
        pct = defaultdict(list)
        failed = set()
        for rep in range(args.repeats):
            out = subprocess.run([args.binary, "tune", str(n)], capture_output=True, text=True, check=True).stdout
            if log:
                log.write(f"# n={n} repeat={rep + 1}\n{out}\n")
            for line in out.splitlines():
                if not line.startswith("TUNE|"):
                    continue
                _, name, _n, ms, gf, p, status = line.split("|")
                if status != "PASS":
                    failed.add(name)
                gflops[name].append(float(gf))
                pct[name].append(float(p))
        table = {}
        for name in gflops:
            table[name] = {
                "median_gflops": round(statistics.median(gflops[name]), 1),
                "median_pct_of_cublas": round(statistics.median(pct[name]), 1) if name != "cuBLAS" else 100.0,
                "runs": len(gflops[name]),
                "status": "FAIL" if name in failed else "PASS",
            }
        candidates = {k: v for k, v in table.items() if k != "cuBLAS" and v["status"] == "PASS"}
        best = max(candidates, key=lambda k: candidates[k]["median_gflops"])
        choices[str(n)] = best
        results[str(n)] = table
        ranked = sorted(candidates.items(), key=lambda kv: -kv[1]["median_gflops"])
        print(f"\nn={n}: best {best} ({table[best]['median_gflops']} GFLOPS, "
              f"{table[best]['median_pct_of_cublas']}% of cuBLAS); lab_v5 (= v5) "
              f"{table.get('lab_v5', {}).get('median_gflops', '-')} GFLOPS")
        for name, r in ranked:
            print(f"   {name:22s} {r['median_gflops']:8.1f} GFLOPS  {r['median_pct_of_cublas']:6.1f}% of cuBLAS")
    if log:
        log.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"gpu": gpu_name(), "method": f"median of {args.repeats} runs, cuBLAS timed before and after "
                   "each run, every config checked against cuBLAS (max rel err < 1e-3)",
                   "choices": choices, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}: {choices}")


if __name__ == "__main__":
    main()
