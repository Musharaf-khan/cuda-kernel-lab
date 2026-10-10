#!/usr/bin/env bash
# One command, every number in the README:
#   build -> run all benchmarks -> GEMM 3x at 4096 -> PyTorch baselines
#   -> Nsight Compute counters for the roofline + bank conflicts -> results.md + roofline.png
#
#   bash scripts/run_all.sh            (from the repo root, on a machine with an NVIDIA GPU)
#
# A failing step is logged and skipped; it never stops the run.
set -u
cd "$(dirname "$0")/.."
RAW=benchmarks/raw
NCU_DIR=benchmarks/ncu
mkdir -p "$RAW" "$NCU_DIR"
rm -f "$RAW"/*.txt

step() { echo; echo "=================== $*"; }

step "GPU"
nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm,memory.total,power.limit,driver_version --format=csv \
  | tee "$RAW/00_gpu.txt"
nvcc --version | tail -2 | tee -a "$RAW/00_gpu.txt"

ARCH=${ARCH:-sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}
step "Build (ARCH=$ARCH)"
make -j4 ARCH="$ARCH" || { echo "BUILD FAILED"; exit 1; }

step "Benchmarks"
./build/vecadd_bench          | tee "$RAW/01_vecadd.txt"
./build/reduction             | tee "$RAW/02_reduction.txt"
./build/transpose             | tee "$RAW/03_transpose.txt"
./build/gemm 256              | tee "$RAW/04_gemm_256_correctness.log"   # .log: correctness only, not in the table
for i in 1 2 3; do
  ./build/gemm 4096           | tee "$RAW/04_gemm_4096_run$i.txt"
done
./build/softmax               | tee "$RAW/05_softmax.txt"
./build/layernorm             | tee "$RAW/06_layernorm.txt"
./build/attention             | tee "$RAW/07_attention.txt"

step "PyTorch baselines"
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  python3 scripts/torch_compare.py | tee "$RAW/08_pytorch.txt"
else
  echo "PyTorch with CUDA not available - skipping"
fi

step "Nsight Compute: roofline counters (one launch per kernel)"
if command -v ncu >/dev/null; then
  METRICS=dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum
  METRICS=$METRICS,sm__sass_thread_inst_executed_op_ffma_pred_on.sum
  METRICS=$METRICS,sm__sass_thread_inst_executed_op_fadd_pred_on.sum
  METRICS=$METRICS,sm__sass_thread_inst_executed_op_fmul_pred_on.sum
  for b in vecadd:vecadd_bench reduction:reduction gemm:gemm softmax:softmax layernorm:layernorm attention:attention; do
    name=${b%%:*}; bin=${b##*:}
    echo "-- $name"
    # --clock-control none: profile at the GPU's normal clocks (ncu otherwise locks to base clock)
    ncu --clock-control none --csv --print-units base --metrics "$METRICS" ./build/$bin profile \
      > "$NCU_DIR/${name}_roofline.csv" 2> "$NCU_DIR/${name}_roofline.err" \
      || echo "   ncu failed for $name (see $NCU_DIR/${name}_roofline.err)"
  done
  echo "-- transpose bank conflicts"
  ncu --clock-control none --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    ./build/transpose profile > "$NCU_DIR/transpose_bank_conflicts.txt" 2>&1 || true
  echo "-- full report for GEMM v4 vs v5 (open the .ncu-rep in the Nsight Compute GUI)"
  ncu --clock-control none --set full -k regex:"sgemm_2d_regtile|sgemm_vectorized" -o "$NCU_DIR/gemm_v4_v5" -f \
    ./build/gemm 2048 profile > /dev/null 2>&1 || true
else
  echo "ncu not found - skipping profiling"
fi

step "Tables and roofline"
python3 scripts/make_table.py "$RAW" > benchmarks/results.md && echo "wrote benchmarks/results.md"
python3 scripts/roofline.py "$NCU_DIR" benchmarks/roofline.png benchmarks/roofline.csv || true

step "Summary"
echo "PASS: $(grep -h '^RESULT' "$RAW"/*.txt | grep -c '|PASS$')   FAIL: $(grep -h '^RESULT' "$RAW"/*.txt | grep -c '|FAIL$')"
grep -h '^RESULT' "$RAW"/*.txt | grep '|FAIL$' || true
grep -h 'FAIL' "$RAW"/*.log || true
cat benchmarks/results.md
