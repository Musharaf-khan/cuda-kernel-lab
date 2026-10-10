"""Instruction mix per kernel from `cuobjdump -sass` output.

    cuobjdump -sass build/gemm_lab > gemm_lab.sass
    python scripts/sass_stats.py gemm_lab.sass [name-filter]

For each kernel it counts the instructions that matter for a GEMM inner loop:
FFMA (the useful math), shared loads by width (LDS / LDS.64 / LDS.128), shared stores,
global loads, barriers, and local-memory traffic (LDL/STL = register spills).
It also reports the same counts for the hottest loop only: the instructions between the
backward branch with the most FFMAs and its target, i.e. what runs ~n/BK times.
"""
import re
import sys
from collections import Counter

INSTR = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)([^;]*);")
FUNC = re.compile(r"Function : (\S+)")


def demangle(name):
    m = re.match(r"_Z\d+(\w+?)(I|v|P|i)", name)
    base = re.match(r"_Z(\d+)", name)
    if base:
        n = int(base.group(1))
        start = len(base.group(0))
        short = name[start:start + n]
        tmpl = [("T" if v == "1" else "F") if k == "b" else v for k, v in re.findall(r"L([ib])(-?\d+)E", name)]
        return short + (f"<{','.join(tmpl)}>" if tmpl else "")
    return name


def classify(op):
    if op.startswith("FFMA"):
        return "FFMA"
    if op.startswith("HMMA"):
        return "HMMA"
    if op.startswith("LDS"):
        if ".128" in op:
            return "LDS.128"
        if ".64" in op:
            return "LDS.64"
        return "LDS.32"
    if op.startswith("STS"):
        return "STS"
    if op.startswith("LDG"):
        return "LDG"
    if op.startswith("STG"):
        return "STG"
    if op.startswith("BAR"):
        return "BAR"
    if op.startswith("LDL") or op.startswith("STL"):
        return "LOCAL(spill)"
    if op.startswith("LDSM"):
        return "LDSM"
    return None


def parse(path):
    funcs, cur = {}, None
    with open(path, errors="replace") as f:
        for line in f:
            m = FUNC.search(line)
            if m:
                cur = demangle(m.group(1))
                funcs[cur] = []
                continue
            m = INSTR.search(line)
            if m and cur is not None:
                funcs[cur].append((int(m.group(1), 16), m.group(3), m.group(4)))
    return funcs


def hot_loop(instrs):
    """Instructions inside the backward branch whose body has the most FFMA/HMMA."""
    best = None
    for addr, op, args in instrs:
        if op.startswith("BRA"):
            t = re.search(r"0x([0-9a-f]+)", args)
            if not t:
                continue
            target = int(t.group(1), 16)
            if target < addr:
                body = [i for i in instrs if target <= i[0] <= addr]
                math = sum(1 for _, o, _ in body if o.startswith(("FFMA", "HMMA")))
                if best is None or math > best[0]:
                    best = (math, body)
    return best[1] if best else []


def summarize(instrs):
    c = Counter(classify(op) for _, op, _ in instrs)
    c.pop(None, None)
    c["total"] = len(instrs)
    return c


def main():
    path = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    cols = ["FFMA", "HMMA", "LDS.32", "LDS.64", "LDS.128", "LDSM", "STS", "LDG", "BAR", "LOCAL(spill)", "total"]
    print("| kernel | scope | " + " | ".join(cols) + " | FFMA per shared load |")
    print("|---|---|" + "---:|" * (len(cols) + 1))
    for name, instrs in parse(path).items():
        if filt not in name:
            continue
        for scope, body in (("whole", instrs), ("hot loop", hot_loop(instrs))):
            c = summarize(body)
            lds = c["LDS.32"] + c["LDS.64"] + c["LDS.128"] + c["LDSM"]
            ratio = f"{(c['FFMA'] + c['HMMA']) / lds:.1f}" if lds else "-"
            print(f"| {name} | {scope} | " + " | ".join(str(c.get(k, 0)) for k in cols) + f" | {ratio} |")


if __name__ == "__main__":
    main()
