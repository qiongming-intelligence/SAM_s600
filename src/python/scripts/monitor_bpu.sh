#!/usr/bin/env bash
set -euo pipefail

hrt_ucp_monitor -b -d "${1:-1000}" -e bpu
