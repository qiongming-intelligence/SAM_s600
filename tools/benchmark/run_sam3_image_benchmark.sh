#!/usr/bin/env bash
set -euo pipefail

./build/sam3_benchmark --mode image "$@"
