# Profiling & intentional-bug workflow (Jetson Nano / ARM64)

This project includes **deliberate failure modes** so you can practice the same
loop used in embedded Linux bring-up: reproduce → observe → profile → fix → re-benchmark.

## Tools on Jetson

```bash
# Install if missing (Ubuntu 18.04 / L4T)
sudo apt-get update
sudo apt-get install -y linux-tools-common linux-tools-generic strace valgrind
# On some Jetson images, perf is provided by:
#   sudo apt-get install -y linux-tools-4.9.337-tegra
# or built from kernel sources. `strace` alone is enough for a first pass.
```

## 1) High CPU path (`bug/high-cpu` style)

Start the inference service with an artificial busy-wait:

```bash
python3.6 scripts/inference_service.py --mode serve \
  --busy-spin-us 2000 \
  --socket /tmp/eai_hot.sock
```

In another shell, drive load + sample:

```bash
# Client
python3.6 scripts/inference_service.py --mode client \
  --socket /tmp/eai_hot.sock --frames 500

# Observe
top -H -p $(pgrep -f inference_service)

# Syscall profile (what is the process waiting on / looping in?)
strace -c -p $(pgrep -f 'inference_service.py --mode serve')

# CPU hotspot (if perf is available)
perf top -p $(pgrep -f 'inference_service.py --mode serve')
# or
perf record -g -p $(pgrep -f 'inference_service.py --mode serve') -- sleep 10
perf report
```

**What you should see:** high user-time CPU, little I/O wait, hot frames around the
busy-spin / `interpreter.invoke` path. Remove `--busy-spin-us` and compare IPC +
inference CPU via `scripts/ipc_benchmark.py`.

Interview line:

> I used `top` / `strace` / `perf` on an Edge AI inference service to separate IPC
> overhead from TFLite invoke cost, and to catch an intentional busy-wait CPU
> hotspot before re-running the latency benchmark.

## 2) Memory leak path (`bug/memory-leak` style)

Inject a per-inference leak (keeps a `bytearray` forever):

```bash
# Short soak that will grow RSS
python3.6 scripts/soak_test.py --max-runs 20000 --window 2000 \
  --inject-leak-bytes 4096 \
  --output results/soak_leak_demo.json
```

Watch RSS climb in the printed windows / JSON `stability.answer == "yes"`.

### Valgrind (C helpers) / Python tracemalloc

Pure-Python leaks are easiest to confirm with process RSS + `tracemalloc`:

```bash
python3.6 - <<'PY'
import tracemalloc, time, os, sys
sys.path.insert(0, "scripts")
# Prefer attaching tracemalloc around the service loop when debugging.
tracemalloc.start()
print("tracemalloc on; combine with soak --inject-leak-bytes")
PY
```

For a Valgrind-friendly C leak demo (optional compile on Jetson):

```bash
# docs/examples/leak_demo.c is a tiny standalone leak for valgrind practice
gcc -O0 -g -o /tmp/leak_demo docs/examples/leak_demo.c
valgrind --leak-check=full /tmp/leak_demo
```

Fix path: drop `--inject-leak-bytes`, re-run soak, confirm
`stability.answer == "no"` and `rss_growth_mb` near zero.

## 3) Suggested branch names (optional)

```text
bug/memory-leak   # default inject-leak in a wrapper script
bug/high-cpu      # default busy-spin in a wrapper script
```

Wrappers live under `scripts/bugs/` so `main` stays clean while still showing the
debug story in the repo.
