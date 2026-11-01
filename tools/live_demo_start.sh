#!/usr/bin/env bash
set -euo pipefail
# Start only this website's bounded API and tunnel. Never stop other GPU jobs.
LIVE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIVE_ROOT="${OPENNWM_LIVE_ROOT:-/file_system/nas/algorithm/dujun.nie/opennwm_live_demo_20261029}"
LIVE_GPU="${OPENNWM_LIVE_GPU:-7}"
cd "$LIVE_REPO"
source .env
mkdir -p "$LIVE_ROOT"
nvidia-smi --id="$LIVE_GPU" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
df -h "$LIVE_ROOT"
for service in api tunnel; do
  if [[ -f "$LIVE_ROOT/$service.pid" ]] && kill -0 "$(cat "$LIVE_ROOT/$service.pid")" 2>/dev/null; then
    printf '%s is already running (PID %s).\n' "$service" "$(cat "$LIVE_ROOT/$service.pid")"
    exit 1
  fi
done
"$NWM_PYTHON" tools/live_demo_start.py --root "$LIVE_ROOT" --gpu "$LIVE_GPU"
