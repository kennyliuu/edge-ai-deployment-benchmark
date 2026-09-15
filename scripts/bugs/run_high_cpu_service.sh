#!/bin/bash
# Demo: inference service with intentional busy-wait (high CPU).
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python3.6 scripts/inference_service.py --mode serve --busy-spin-us 2000 "$@"
