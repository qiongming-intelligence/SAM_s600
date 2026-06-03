#!/usr/bin/env bash
set -euo pipefail

# Benchmark one SAM3 full-detector HBM on the local S600 across single-core and
# dual-core BPU scheduling, with profiling, and print a compact latency summary.
#
# This runs on the aarch64 S600 (the local box), not on the x86_64 compile host.
# It reuses the fixed FP32 6-input ABI vectors under models/hbm/perf_inputs_full_f32
# by default; pass a different input dir if the HBM expects another ABI.
#
# usage: bench_full_detector.sh HBM_FILE MODEL_NAME [INPUT_DIR] [FRAME_COUNT]

usage() {
  cat >&2 <<'EOF'
usage: bench_full_detector.sh HBM_FILE MODEL_NAME [INPUT_DIR] [FRAME_COUNT]

  HBM_FILE      path to the .hbm to benchmark
  MODEL_NAME    model name inside the HBM (see `hrt_model_exec model_info`)
  INPUT_DIR     dir with the 6 fixed input .bin files in ABI order
                (default models/hbm/perf_inputs_full_f32)
  FRAME_COUNT   frames per perf run (default 20)

Runs model_info, then perf on --core_id 1 (single core) and --core_id 1,2
(dual core, single thread), each with --profile_path, and prints latency +
BPU/CPU split for both.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
if [[ $# -lt 2 || $# -gt 4 ]]; then usage; exit 2; fi

hbm="$1"
model_name="$2"
input_dir="${3:-models/hbm/perf_inputs_full_f32}"
frames="${4:-20}"
hrt="/usr/hobot/bin/hrt_model_exec"

if [[ ! -f "$hbm" ]]; then echo "missing HBM: $hbm" >&2; exit 1; fi
if [[ ! -d "$input_dir" ]]; then echo "missing input dir: $input_dir" >&2; exit 1; fi

# Build the comma-joined input list in fixed ABI order.
declare -a order=(
  pixel_values input_ids attention_mask input_boxes input_boxes_labels geometry_roi_features
)
inputs=""
for key in "${order[@]}"; do
  f=$(ls "$input_dir"/${key}*.bin 2>/dev/null | head -n1 || true)
  if [[ -z "$f" ]]; then echo "missing input file for '$key' in $input_dir" >&2; exit 1; fi
  inputs+="${inputs:+,}$f"
done

variant="$(basename "$hbm" .hbm)"
prof_root="build/sam3_modelscope_export/perf_matrix/$variant"
mkdir -p "$prof_root"

echo "================= model_info: $variant ================="
"$hrt" model_info --model_file "$hbm" 2>&1 | grep -E '^\[model name\]|^name:|valid shape|tensor type' || true

run_one() {
  local label="$1" core="$2" threads="$3"
  local dir="$prof_root/$label"
  rm -rf "$dir"; mkdir -p "$dir"
  echo "----------------- perf $label (core_id=$core thread=$threads) -----------------"
  "$hrt" perf --model_file "$hbm" --model_name "$model_name" \
    --core_id "$core" --frame_count "$frames" --perf_time 0 --thread_num "$threads" \
    --input_file "$inputs" --profile_path "$dir" 2>&1 \
    | grep -E 'Average|Frame      rate|Frame totally|Program run time' || true
  sed -n '/processor_latency/,/task_latency/p' "$dir/profiler.csv" 2>/dev/null \
    | grep -E 'BPU_inference|CPU_inference' || true
}

run_one single_core 1 1
run_one dual_core "1,2" 1
run_one dual_core_2thread "1,2" 2

echo "================= done: $variant ================="
