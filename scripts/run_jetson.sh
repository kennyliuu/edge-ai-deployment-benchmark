#!/bin/bash
# One-command smoke suite on Jetson Nano (keeps runtime short).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=python3.6
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python3
fi

echo "=== 1) TFLite latency + resource sample ==="
"$PY" scripts/benchmark.py --output results/jetson_nano_2gb.json --runs 100

echo "=== 2) Resource ladder (100 → 10k) ==="
"$PY" scripts/benchmark.py --runs 50 \
  --ladder 100,1000,10000 \
  --output results/resource_ladder.json

echo "=== 3) IPC benchmark (socket / shm / pipe) ==="
"$PY" scripts/ipc_benchmark.py --iters 1000 --warmup 50 \
  --output results/ipc_benchmark.json

echo "=== 4) Sensor simulator → inference service ==="
"$PY" scripts/inference_service.py --mode demo --frames 100 \
  --output results/inference_service.json

echo "=== 5) Short soak with RSS / stability verdict ==="
"$PY" scripts/soak_test.py --max-runs 5000 --window 1000 \
  --output results/soak_test_smoke.json

echo "Done. See results/*.json and docs/PROFILING.md"
