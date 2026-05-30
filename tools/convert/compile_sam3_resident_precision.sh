#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: compile_sam3_resident_precision.sh [fp16|bf16] [OUTPUT_DIR]

Build a resident SAM3 detector HBM from a mixed-precision ONNX graph. This is
intended for the fused/full detector path used by sam3_image_resident.yaml.

Requires hb_compile from the D-Robotics/OpenExplorer x86_64 toolchain. Run on an
x86_64 host or inside the AI Toolchain Docker image with this repository mounted.

Environment overrides:
  SAM3_RESIDENT_SOURCE_ONNX   Source FP32 ONNX path.
  SAM3_RESIDENT_ONNX_DIR      Directory for generated precision ONNX files.
  HB_COMPILE_OPTIMIZE_LEVEL   HBDK optimize level, default O2.
  HB_COMPILE_CORE_NUM         Optional compiler core_num parameter.
  HB_COMPILE_COMPILE_MODE     Optional compiler compile_mode parameter.
  HB_COMPILE_BALANCE_FACTOR   Optional compiler balance_factor parameter.
EOF
}

precision="${1:-fp16}"
output_dir="${2:-models/hbm}"
case "$precision" in
  fp16|bf16) ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

if ! command -v hb_compile >/dev/null 2>&1; then
  echo "hb_compile not found in PATH; run inside the x86_64 OpenExplorer/HBDK toolchain environment" >&2
  exit 1
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "hb_compile is an x86_64 host tool; current architecture is $(uname -m)" >&2
  exit 1
fi

source_onnx="${SAM3_RESIDENT_SOURCE_ONNX:-build/sam3_modelscope_export/onnx/sam3_detector_hbdk_full.onnx}"
onnx_dir="${SAM3_RESIDENT_ONNX_DIR:-build/sam3_modelscope_export/onnx}"
precision_onnx="$onnx_dir/sam3_detector_hbdk_full_${precision}.onnx"
mkdir -p "$onnx_dir" "$output_dir"

if [[ ! -f "$source_onnx" ]]; then
  echo "missing source ONNX: $source_onnx" >&2
  exit 1
fi

python3 tools/convert/convert_onnx_precision.py "$source_onnx" "$precision_onnx" --precision "$precision"

config_dir="$output_dir/hb_compile_configs"
mkdir -p "$config_dir"
config_path="$config_dir/sam3_detector_hbdk_full_${precision}.yaml"
prefix="sam3_detector_hbdk_full_${precision}"

cat >"$config_path" <<EOF
model_parameters:
  onnx_model: "$precision_onnx"
  march: "nash-p"
  working_dir: "$output_dir"
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
  optimize_level: "${HB_COMPILE_OPTIMIZE_LEVEL:-O2}"
EOF

if [[ -n "${HB_COMPILE_CORE_NUM:-}" ]]; then
  printf '  core_num: %q\n' "$HB_COMPILE_CORE_NUM" >>"$config_path"
fi
if [[ -n "${HB_COMPILE_COMPILE_MODE:-}" ]]; then
  printf '  compile_mode: %q\n' "$HB_COMPILE_COMPILE_MODE" >>"$config_path"
fi
if [[ -n "${HB_COMPILE_BALANCE_FACTOR:-}" ]]; then
  printf '  balance_factor: %q\n' "$HB_COMPILE_BALANCE_FACTOR" >>"$config_path"
fi

echo "converting $precision_onnx -> $output_dir/${prefix}.hbm"
hb_compile --config "$config_path"

echo "wrote $output_dir/${prefix}.hbm"
