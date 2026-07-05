#!/bin/bash
# Run on Jetson Nano: benchmark only (uses pre-exported TFLite models).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Benchmark TFLite on Jetson Nano ==="
python3.6 scripts/benchmark.py --output results/jetson_nano_2gb.json
echo "Done."
