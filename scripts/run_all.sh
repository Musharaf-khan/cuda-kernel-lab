#!/usr/bin/env bash
# One command, every number in the README:
#   build -> benchmarks -> SGEMM autotuner -> GEMM (3 runs) -> SGEMM attempt ladder (3 runs)
#   -> softmax, LayerNorm, attention (v1 + v2), WMMA HGEMM -> PyTorch + Triton baselines
#   -> Nsight Compute (roofline counters, bank conflicts, SGEMM attempt stall/occupancy analysis)
#   -> SASS instruction mix -> results.md + roofline charts
#
#   bash scripts/run_all.sh            (from the repo root, on a machine with an NVIDIA GPU)
#   SKIP_NCU=1 bash scripts/run_all.sh (benchmarks only, much faster)
#
# A failing step is logged and skipped; it never stops the run.
set -u
cd "$(dirname "$0")/.."
RAW=benchmarks/raw
NCU_DIR=benchmarks/ncu
SASS_DIR=benchmarks/sass
mkdir -p "$RAW" "$NCU_DIR" "$SASS_DIR" configs
rm -f "$RAW"/*.txt "$RAW"/*.log

step() { echo; echo "=================== $*"; }

step "GPU"
nvidia-smi --query-gpu=name,compute_cap,clocks.max.sm,memory.total,power.limit,driver_version --format=csv \
  | tee "$RAW/00_gpu.txt"
nvcc --version | tail -2 | tee -a "$RAW/00_gpu.txt"

ARCH=${ARCH:-sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}
step "Build (ARCH=$ARCH)"
make -j4 ARCH="$ARCH" || { echo "BUILD FAILED"; exit 1; }

step "Memory-bound basics"
./build/vecadd_bench          | tee "$RAW/01_vecadd.txt"
./build/reduction             | tee "$RAW/02_reduction.txt"
./build/transpose             | tee "$RAW/03_transpose.txt"

step "SGEMM autotuner (writes configs/sgemm_autotune.json, read by build/gemm as v6)"
python3 scripts/autotune.py --sizes 256 1024 2048 4096 --repeats 3 --log "$RAW/04_autotune.log" \
  | tee "$RAW/04_autotune_summary.log"

step "SGEMM v1-v6 vs cuBLAS"
for i in 1 2 3; do
  ./build/gemm 256            | tee "$RAW/04_gemm_256_run$i.txt"       # small size: the autotuner's n=256 fix
done
for i in 1 2 3; do
  ./build/gemm 4096           | tee "$RAW/04_gemm_4096_run$i.txt"
done

step "SGEMM attempt ladder (each change measured against cuBLAS in the same run)"
for i in 1 2 3; do
  ./build/gemm_lab attempts 4096 | tee "$RAW/04_gemm_attempts_run$i.txt"
done

step "Softmax, LayerNorm, attention, WMMA HGEMM"
./build/softmax               | tee "$RAW/05_softmax.txt"
./build/layernorm             | tee "$RAW/06_layernorm.txt"
./build/attention             | tee "$RAW/07_attention.txt"
./build/hgemm                 | tee "$RAW/08_hgemm.txt"

step "PyTorch and Triton baselines"
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  python3 scripts/torch_compare.py | tee "$RAW/09_pytorch.txt"
  if python3 -c "import triton" 2>/dev/null; then
    python3 scripts/triton_ports.py 2>&1 | tee "$RAW/10_triton.txt"
  else
    echo "Triton not available - skipping"
  fi
else
  echo "PyTorch with CUDA not available - skipping"
fi

if [ "${SKIP_NCU:-0}" = "1" ]; then
  echo "SKIP_NCU=1 - skipping Nsight Compute"
elif command -v ncu >/dev/null; then
  step "Nsight Compute: roofline counters (one launch per kernel)"
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

  echo "-- SGEMM attempts: occupancy, warp stalls, scheduler and throughput sections (n=4096)"
  ncu --clock-control none --section SpeedOfLight --section Occupancy --section LaunchStats \
      --section WarpStateStats --section SchedulerStats \
      ./build/gemm_lab profile 4096 > "$NCU_DIR/sgemm_attempts_sections.txt" 2>&1 \
    || echo "   ncu sections failed (see $NCU_DIR/sgemm_attempts_sections.txt)"
  echo "-- SGEMM attempts: shared-memory bank conflicts, wavefronts, FMA-pipe utilization"
  ncu --clock-control none --csv --print-units base \
      --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum \
      ./build/gemm_lab profile 4096 > "$NCU_DIR/sgemm_attempts_metrics.csv" 2> "$NCU_DIR/sgemm_attempts_metrics.err" \
    || echo "   ncu metrics failed (see $NCU_DIR/sgemm_attempts_metrics.err)"
  echo "-- WMMA HGEMM and cuBLAS FP16: sections"
  ncu --clock-control none --section SpeedOfLight --section Occupancy --section WarpStateStats \
      ./build/hgemm 2048 profile > "$NCU_DIR/hgemm_sections.txt" 2>&1 || true
  echo "-- attention v1 vs v2: occupancy and stalls"
  ncu --clock-control none --section Occupancy --section LaunchStats --section WarpStateStats \
      -k regex:"flash_attention" ./build/attention profile > "$NCU_DIR/attention_sections.txt" 2>&1 || true
else
  echo "ncu not found - skipping profiling"
fi

step "SASS instruction mix (cuobjdump)"
if command -v cuobjdump >/dev/null; then
  for b in gemm gemm_lab hgemm attention; do
    cuobjdump -sass build/$b > "$SASS_DIR/$b.sass" 2>/dev/null \
      && python3 scripts/sass_stats.py "$SASS_DIR/$b.sass" > "$SASS_DIR/${b}_instruction_mix.md" \
      && rm -f "$SASS_DIR/$b.sass"
  done
  echo "wrote $SASS_DIR/*_instruction_mix.md"
fi

step "Tables and roofline"
python3 scripts/make_table.py "$RAW" > benchmarks/results.md && echo "wrote benchmarks/results.md"
python3 scripts/roofline.py "$NCU_DIR" benchmarks/roofline.png benchmarks/roofline.csv || true

step "Summary"
echo "PASS: $(grep -h '^RESULT' "$RAW"/*.txt | grep -c '|PASS$')   FAIL: $(grep -h '^RESULT' "$RAW"/*.txt | grep -c '|FAIL$')"
grep -h '^RESULT' "$RAW"/*.txt | grep '|FAIL$' || true
grep -h 'FAIL' "$RAW"/*.log || true
cat benchmarks/results.md
