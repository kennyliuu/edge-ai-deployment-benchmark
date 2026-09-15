#!/bin/bash
# Demo: soak test that intentionally leaks memory each inference.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python3.6 scripts/soak_test.py \
  --max-runs "${MAX_RUNS:-20000}" \
  --window "${WINDOW:-2000}" \
  --inject-leak-bytes "${LEAK_BYTES:-4096}" \
  --output results/soak_leak_demo.json \
  "$@"
