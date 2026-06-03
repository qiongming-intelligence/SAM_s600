#!/usr/bin/env bash
set -euo pipefail

# Compile the SAM3 full detector to HBM across a small matrix of compiler knobs
# (precision x core_num x optimize_level x compile_mode) so we can benchmark each
# variant on the S600 and pick the Pareto-optimal one.
#
# Unlike compile_resident_precision.sh (fp16/bf16 only) and compile_hbm.sh
# (contract-driven FP), this driver exposes core_num directly. The default
# core_num in the toolchain schema is 1, which means every existing baseline HBM
# only ever uses a single BPU core; setting core_num=2 lets a single inference be
# split across both S600 BPU cores.
#
# hb_compile is an x86_64 host tool: run this on ws-wan inside the sam3-hbm conda
# env, not on the aarch64 S600.

usage() {
  cat >&2 <<'EOF'
usage: compile_full_detector_matrix.sh PRECISION CORE_NUM OPT_LEVEL COMPILE_MODE [OUTPUT_DIR]

  PRECISION      fp32 | fp16 | bf16   floating precision of the graph weights
  CORE_NUM       1 | 2                BPU cores a single inference is split across
  OPT_LEVEL      O0 | O1 | O2 | O3    hbdk optimize_level
  COMPILE_MODE   latency | bandwidth  hbdk compile_mode
  OUTPUT_DIR     default models/hbm

Produces models/hbm/sam3_detector_full_<prec>_core<N>_<opt>_<mode>.hbm and the
matching hb_compile YAML under <OUTPUT_DIR>/hb_compile_configs/.

Environment overrides:
  SAM3_FULL_SOURCE_ONNX   FP32 source ONNX (default build/sam3_modelscope_export/onnx/sam3_detector_hbdk_full.onnx)
  SAM3_FULL_ONNX_DIR      Directory for generated fp16/bf16 ONNX (default alongside source)
  SAM3_FULL_OUTPUT_PREFIX Optional output HBM prefix (default derived from knobs)
  HB_COMPILE_JOBS         compiler jobs parameter (default 32)
  HB_COMPILE_BALANCE_FACTOR  optional balance_factor

Requires hb_compile from the D-Robotics/OpenExplorer x86_64 toolchain.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 4 || $# -gt 5 ]]; then
  usage
  exit 2
fi

precision="$1"
core_num="$2"
opt_level="$3"
compile_mode="$4"
output_dir="${5:-models/hbm}"

case "$precision" in
  fp32|fp16|bf16) ;;
  *) echo "invalid PRECISION: $precision (expected fp32|fp16|bf16)" >&2; exit 2 ;;
esac
case "$core_num" in
  1|2) ;;
  *) echo "invalid CORE_NUM: $core_num (expected 1|2)" >&2; exit 2 ;;
esac
case "$opt_level" in
  O0|O1|O2|O3) ;;
  *) echo "invalid OPT_LEVEL: $opt_level (expected O0|O1|O2|O3)" >&2; exit 2 ;;
esac
case "$compile_mode" in
  latency|bandwidth) ;;
  *) echo "invalid COMPILE_MODE: $compile_mode (expected latency|bandwidth)" >&2; exit 2 ;;
esac

if ! command -v hb_compile >/dev/null 2>&1; then
  echo "hb_compile not found in PATH; run inside the x86_64 OpenExplorer/HBDK toolchain environment" >&2
  exit 1
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "hb_compile is an x86_64 host tool; current architecture is $(uname -m)" >&2
  exit 1
fi

source_onnx="${SAM3_FULL_SOURCE_ONNX:-build/sam3_modelscope_export/onnx/sam3_detector_hbdk_full.onnx}"
onnx_dir="${SAM3_FULL_ONNX_DIR:-$(dirname "$source_onnx")}"
mkdir -p "$onnx_dir" "$output_dir"

if [[ ! -f "$source_onnx" ]]; then
  echo "missing source ONNX: $source_onnx" >&2
  exit 1
fi

# Pick (and for fp16/bf16, ensure) the graph ONNX for this precision.
if [[ "$precision" == "fp32" ]]; then
  graph_onnx="$source_onnx"
else
  graph_onnx="$onnx_dir/sam3_detector_hbdk_full_${precision}.onnx"
  if [[ ! -f "$graph_onnx" ]]; then
    echo "generating $precision ONNX: $graph_onnx" >&2
    python3 -m sam_s600_tools.convert.precision "$source_onnx" "$graph_onnx" --precision "$precision"
  fi
fi

prefix="${SAM3_FULL_OUTPUT_PREFIX:-sam3_detector_full_${precision}_core${core_num}_${opt_level}_${compile_mode}}"
config_dir="$output_dir/hb_compile_configs"
mkdir -p "$config_dir"
config_path="$config_dir/${prefix}.yaml"

# hb_compile resolves a relative onnx_model against the config file's directory,
# not the cwd, so write absolute paths to stay robust regardless of where the
# YAML lives.
graph_onnx_abs="$(readlink -f "$graph_onnx")"
output_dir_abs="$(readlink -f "$output_dir")"

cat >"$config_path" <<EOF
model_parameters:
  onnx_model: "$graph_onnx_abs"
  march: "nash-p"
  working_dir: "$output_dir_abs"
  output_model_file_prefix: "$prefix"

input_parameters:
  input_name: "pixel_values;input_ids;attention_mask;input_boxes;input_boxes_labels;geometry_roi_features"
  input_type_rt: "featuremap;featuremap;featuremap;featuremap;featuremap;featuremap"
  input_type_train: "featuremap;featuremap;featuremap;featuremap;featuremap;featuremap"
  input_layout_train: "NCHW;NCHW;NCHW;NCHW;NCHW;NCHW"
  input_shape: "1x3x1008x1008;1x16;1x16;1x2x4;1x2;2x256x7x7"
  norm_type: "no_preprocess"

calibration_parameters:
  calibration_type: "skip"

compiler_parameters:
  compile_mode: "$compile_mode"
  optimize_level: "$opt_level"
  core_num: $core_num
  jobs: ${HB_COMPILE_JOBS:-32}
EOF

if [[ -n "${HB_COMPILE_BALANCE_FACTOR:-}" ]]; then
  printf '  balance_factor: %s\n' "$HB_COMPILE_BALANCE_FACTOR" >>"$config_path"
fi

echo "converting $graph_onnx -> $output_dir/${prefix}.hbm (core_num=$core_num $opt_level $compile_mode)"
hb_compile --config "$config_path"

echo "wrote $output_dir/${prefix}.hbm"
