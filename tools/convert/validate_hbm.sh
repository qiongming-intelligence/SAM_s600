#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 model.hbm" >&2
  exit 2
fi
hrt_model_exec model_info --model_file "$1"
