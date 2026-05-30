#!/usr/bin/env bash
set -euo pipefail

echo "Checking S600 SAM3 runtime prerequisites..."
command -v hrt_ucp_monitor >/dev/null && echo "hrt_ucp_monitor: found" || echo "hrt_ucp_monitor: missing"
command -v hrt_model_exec >/dev/null && echo "hrt_model_exec: found" || echo "hrt_model_exec: missing"
command -v cmake >/dev/null && cmake --version | head -1
command -v g++ >/dev/null && g++ --version | head -1
