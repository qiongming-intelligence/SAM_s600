#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: convert_sam3_to_hbm.sh CONTRACT_DIR [OUTPUT_DIR]

CONTRACT_DIR is produced by tools/export/sam3_export_contract.py, for example:
  tools/export/sam3_export_contract.py --out-dir build/sam3_export --print-plan
  tools/convert/convert_sam3_to_hbm.sh build/sam3_export/contracts models/hbm

The script expects hb_mapper from the D-Robotics S600 toolchain and ONNX files at
the onnx_path recorded in each contract JSON.
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

if ! command -v hb_mapper >/dev/null 2>&1; then
  echo "hb_mapper not found in PATH; install and source the D-Robotics S600 toolchain first" >&2
  exit 1
fi

mkdir -p "$output_dir"
shopt -s nullglob
contracts=("$contract_dir"/*.json)
if [[ ${#contracts[@]} -eq 0 ]]; then
  echo "no contract JSON files found in: $contract_dir" >&2
  exit 1
fi

for contract in "${contracts[@]}"; do
  onnx_path="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["onnx_path"])' "$contract")"
  hbm_name="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["hbm_name"])' "$contract")"
  if [[ ! -f "$onnx_path" ]]; then
    echo "missing ONNX for $(basename "$contract"): $onnx_path" >&2
    exit 1
  fi
  echo "converting $onnx_path -> $output_dir/$hbm_name"
  hb_mapper makertbin --model-type onnx --model "$onnx_path" --output-dir "$output_dir"
done
