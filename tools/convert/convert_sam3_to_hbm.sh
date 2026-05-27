#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: convert_sam3_to_hbm.sh CONTRACT_DIR [OUTPUT_DIR]

CONTRACT_DIR is produced by tools/export/sam3_export_contract.py, for example:
  tools/export/sam3_export_contract.py --out-dir build/sam3_export --print-plan
  tools/convert/convert_sam3_to_hbm.sh build/sam3_export/contracts models/hbm

The script expects hb_compile from D-Robotics AI Toolchain 3.7.0+ and ONNX files at
the onnx_path recorded in each contract JSON. hb_compile is an x86_64 host tool;
run this script inside the AI Toolchain Docker image or on an x86_64 host with the
OpenExplorer wheels installed.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

contract_dir="$1"
output_dir="${2:-models/hbm}"

if [[ ! -d "$contract_dir" ]]; then
  echo "contract directory does not exist: $contract_dir" >&2
  exit 1
fi

if ! command -v hb_compile >/dev/null 2>&1; then
  echo "hb_compile not found in PATH; install D-Robotics AI Toolchain 3.7.0+ on an x86_64 host or run inside the AI Toolchain Docker image" >&2
  exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "hb_compile is distributed as an x86_64 host tool; current architecture is $(uname -m)" >&2
  exit 1
fi

mkdir -p "$output_dir"
config_dir="$output_dir/hb_compile_configs"
mkdir -p "$config_dir"
shopt -s nullglob
contracts=("$contract_dir"/*.json)
if [[ ${#contracts[@]} -eq 0 ]]; then
  echo "no contract JSON files found in: $contract_dir" >&2
  exit 1
fi

for contract in "${contracts[@]}"; do
  mapfile -t fields < <(python3 - "$contract" "$output_dir" <<'PY'
import json
import sys
from pathlib import Path
contract = json.load(open(sys.argv[1], encoding="utf-8"))
out_dir = Path(sys.argv[2])
print(contract["name"])
print(contract["onnx_path"])
print(contract["hbm_name"])
print(Path(contract["hbm_name"]).stem)
print(";".join(tensor["name"] for tensor in contract["inputs"]))
print(";".join("x".join(str(dim) for dim in tensor["concrete_shape"]) for tensor in contract["inputs"]))
PY
  )
  name="${fields[0]}"
  onnx_path="${fields[1]}"
  hbm_name="${fields[2]}"
  prefix="${fields[3]}"
  input_names="${fields[4]}"
  input_shapes="${fields[5]}"
  config_path="$config_dir/${name}.yaml"
  if [[ ! -f "$onnx_path" ]]; then
    echo "missing ONNX for $(basename "$contract"): $onnx_path" >&2
    exit 1
  fi
  cat >"$config_path" <<EOF
model_parameters:
  onnx_model: "$onnx_path"
  march: "nash-p"
  working_dir: "$output_dir"
  output_model_file_prefix: "$prefix"

input_parameters:
  input_name: "$input_names"
  input_type_rt: "featuremap"
  input_type_train: "featuremap"
  input_layout_train: "NCHW"
  input_shape: "$input_shapes"
  norm_type: "no_preprocess"

calibration_parameters:
  calibration_type: "skip"

compiler_parameters:
  optimize_level: "O2"
EOF
  echo "converting $onnx_path -> $output_dir/$hbm_name"
  hb_compile --config "$config_path"
done
