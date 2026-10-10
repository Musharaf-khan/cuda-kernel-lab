# Interview notes: answers backed by this repo

Short answers to the questions GPU / CUDA / ML-systems interviews keep asking, each tied to a kernel and a number here, so the answer is "here's what I measured", not "I read that".

## Fundamentals

**What is a warp? Why block sizes in multiples of 32?**
32 threads issued together in lockstep (SIMT). A block of 250 threads still occupies 8 warps; the last one runs 26 useful lanes. Every kernel here uses 256 (8 full warps) or 1024.

**What does warp divergence cost?**
When lanes of one warp take different branches, the warp executes both paths with inactive lanes masked off, so the cost is the sum of the paths. Bounds guards (`if (i < n)`) only diverge in the last warp, which is fine. Restructure so the branch condition is uniform per warp.

**Memory hierarchy, slowest to fastest?**
DRAM/HBM (T4: 320 GB/s, hundreds of cycles) → L2 (shared by all SMs) → L1 / shared memory (per SM, programmer-managed, ~32 banks) → registers (per thread). Every big win in this repo is moving data one level up this list.

**What is coalescing?**
A warp's 32 accesses to 32 consecutive 4-byte words merge into a few 32-byte sector transactions; scattered accesses become up to 32. Measured: swapping two index lines in naive SGEMM = **8.2x**; naive transpose (scattered writes) = 85 GB/s vs 235 GB/s for a coalesced copy; thread-per-row softmax read **8.6x** the minimum bytes from DRAM.

**What is a bank conflict and how do you fix it?**
Shared memory has 32 banks of 4-byte words; bank = (address/4) mod 32. Two lanes hitting different addresses in the same bank serialize. Reading a column of a `float[32][32]` tile is a 32-way conflict; pad to `[32][33]` and column elements land in banks `(r + c) mod 32`. Measured: **16,252,928 → 0 conflicts** in `ncu`, and I can derive the number (524,288 warp loads × 31 replays). Follow-up I can talk about: the same conflict count cost 1.8x in one session and only 1.06x in another (different toolchain). A counter tells you a problem exists; timing tells you what it costs.

**What is occupancy, and is 100% the goal?**
Active warps per SM / maximum. Limited by registers per thread, shared memory per block and threads per block. Not the goal: SGEMM v5 uses 125 registers per thread (low occupancy) and is the fastest hand-written kernel here, because each thread has 64 independent FMAs of instruction-level parallelism to hide latency. The fused attention kernel runs at 254 registers for the same reason.

## Performance reasoning

**Walk me through optimizing a matmul.** (have the table ready: [`learnings.md`](learnings.md#04--sgemm-every-fix-exposes-the-next-bottleneck))
Naive (61.7 GFLOPS) → coalesce (8.2x) → shared-memory tiles (only 1.5x: the L2 was already catching v2's reuse, DRAM reads barely moved 1,654 → 1,584 MB, and shared memory became the limit at 1 FMA per 2 loads) → register tiling 8×8 per thread (4 FMA per load, 3.7x, 85% of cuBLAS) → `float4` loads and a transposed A tile (+9%, **91.6% of cuBLAS**, 3,101.6 GFLOPS). Next: double buffering, warp tiling, Tensor Cores.

**Is this kernel memory-bound or compute-bound?**
Compute arithmetic intensity = FLOPs / DRAM bytes and compare with the ridge point = peak FLOP/s ÷ peak bandwidth (T4: 8100 / 320 ≈ 25 FLOP/byte). Below → memory-bound: coalesce, tile, fuse, cut passes. Above → compute-bound: register tiling, ILP, unrolling, Tensor Cores. The roofline in the README plots every kernel with measured counters.

**Why does fusion win?**
Elementwise and normalization ops do ~1 FLOP per byte, far left of the ridge point, so their time is bytes moved. Fusing add + LayerNorm removes one full write + read of the intermediate (5 array-passes → 4): measured 375 → 304 MB of DRAM traffic and 1.22x, almost exactly the 5 : 4 prediction. FlashAttention removes the N×N score matrix entirely: 1,536 MB → 59 MB (26x) at N = 2048, and N = 16384 runs where the unfused version needs 17.2 GB.

**Explain online softmax / why FlashAttention is legal.**
Keep a running max `m` and sum `l`. When a new chunk raises the max to `m'`, everything accumulated so far is too large by `exp(m' − m)`, so multiply `l` (and the partial output `o`) by `exp(m − m')`. The (m, l) merge is associative, so it can be done chunk by chunk, in any order, or in a reduction tree, and the result is exactly softmax. That lets attention process K/V in tiles without ever holding a full row of scores.

Honest follow-up: for plain softmax on 4096-wide rows, online softmax gave **no** speedup on the T4. The counters showed the "3-pass" version only read 1.4x the minimum bytes because the row was still cached. Keeping the row in registers (1 read) is what won: 1.25x faster than `torch.softmax`.

**Why subtract the max in softmax?**
`exp` overflows FP32 above ~88.7. Softmax is shift-invariant, so subtracting the row max is free. The softmax benchmark demonstrates the NaNs on logits in [90, 110].

**Why Welford instead of Σx and Σx²?**
`E[x²] − E[x]²` cancels catastrophically when the mean is large compared with the variance. Welford keeps (count, mean, M2) and merges with Chan's formula; one stable pass.

**How did you make sure your benchmark numbers are real?**
Warm-up launch, CUDA events, kernel-only timing, many iterations, outputs checked against a double-precision CPU or cuBLAS reference with stated tolerances (and NaN counted as failure). The T4 throttles, so I compare within one run (cuBLAS timed before and after) and report medians of 3; cuBLAS ranged 3,165–3,575 GFLOPS across three runs while my v5/cuBLAS ratio stayed within 90–93%. Also stated where I lose: my fused attention is ~10% slower than PyTorch's SDPA and slower than unfused below N = 2048.

## System questions

**How do you optimize host-to-device transfers?**
Pinned (page-locked) host memory for full PCIe bandwidth, `cudaMemcpyAsync` on streams to overlap copies with kernels, batch small copies, and above all keep data resident on the GPU between kernels. (The benchmarks here copy once and never time the copy.)

**How do you make a kernel portable across GPU generations?**
No hard-coded SM counts (the reduction queries `cudaDevAttrMultiProcessorCount`), compile for the target `-arch` (`make ARCH=sm_80`), keep tile sizes as template parameters, re-tune them per architecture, and check occupancy with the profiler instead of assuming.

**What would you do next on a newer GPU?**
Tensor Cores (WMMA → MMA → WGMMA on Hopper), `cp.async`/TMA for asynchronous copies and multi-stage pipelines, FP16/BF16/FP8 data types, warp specialization; or reach for CUTLASS/CuTe or Triton instead of raw CUDA.

## Vocabulary (say it in one line)

| Term | Meaning |
|---|---|
| HBM / DRAM | GPU main memory (T4 uses GDDR6) |
| SM | streaming multiprocessor; a block lives on one SM |
| Coalescing | a warp's accesses merged into few transactions |
| Bank conflict | lanes hitting the same shared-memory bank, serialized |
| Occupancy | active warps / max warps per SM |
| Arithmetic intensity | FLOPs per byte of DRAM traffic |
| Roofline | performance ceiling = min(peak FLOP/s, intensity × bandwidth) |
| Warp shuffle | `__shfl_*_sync`, register exchange inside a warp |
| `__syncthreads()` | block-wide barrier; must be reached by every thread |
| PTX / SASS | virtual ISA / real machine code (`cuobjdump -sass`) |
| Tensor Cores / MMA | matrix-multiply-accumulate units (FP16/BF16/TF32/INT8/FP8) |
| Streams / events | ordered work queues / timestamps for sync and timing |
