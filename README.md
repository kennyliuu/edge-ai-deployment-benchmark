# Edge AI System Performance & Stability Benchmark

End-to-end **system performance** benchmark on **Jetson Nano 2GB**: TensorFlow Lite
inference, IPC (Unix socket / shared memory / pipe), CPU·RSS·thermal monitoring,
long-running soak tests with a memory-growth verdict, and a reproducible
`perf` / `strace` / Valgrind debug loop.

Interview narrative this repo supports:

> Application → IPC → AI Runtime → System Resource → Performance → Stability

Upgrade plan: [`docs/SYSTEM_PERF_STABILITY_PLAN.md`](docs/SYSTEM_PERF_STABILITY_PLAN.md)  
Profiling / intentional bugs: [`docs/PROFILING.md`](docs/PROFILING.md)

## Test Platform (Jetson Nano 2GB)

| Component | Specification |
|---|---|
| **SoC** | NVIDIA Tegra X1 (T210) |
| **CPU** | Quad-core ARM Cortex-A57 @ 1.43 GHz |
| **GPU** | 128-core NVIDIA Maxwell |
| **Memory** | 2 GB LPDDR4 (64-bit, 25.6 GB/s) |
| **OS** | Ubuntu 18.04 (L4T) · JetPack 4.6.x · L4T R32.7.6 |
| **Arch** | aarch64 |
| **Runtime** | Python 3.6.9 · TensorFlow 2.7 (`tf.lite.Interpreter`) |

> Jetson Nano 2GB has no NPU; inference runs on **CPU** via TFLite — realistic
> embedded-Linux constraints (memory, glibc, runtime compatibility).

## Results (Jetson Nano 2GB)

### Model deployment (baseline)

| Model | Size | Input | Accuracy | Latency (avg) | Latency (p95) |
|---|---|---|---|---|---|
| `mnist_fp32.tflite` | 883 KB | float32 `[1,28,28]` | 98.84% | **0.83 ms** | 0.87 ms |
| `mnist_int8.tflite` | 228 KB | uint8 `[1,28,28]` | 98.87% | **0.69 ms** | 0.70 ms |

**INT8 vs FP32:** 74% smaller model, ~17% lower latency, no accuracy loss on MNIST.

Full report: [`results/jetson_nano_2gb.json`](results/jetson_nano_2gb.json)

### Stability (existing soak)

| Run | Inferences | Errors | Latency drift | Notes |
|---|---|---|---|---|
| 5 min | 438,270 | 0 | 0.0% | [`soak_test.json`](results/soak_test.json) |
| 30 min | 2,616,600 | 0 | −1.01% | [`soak_test_30min.json`](results/soak_test_30min.json) |

New soak builds also track **process RSS / peak RSS / CPU%** and emit an explicit
verdict: *Does memory usage continuously increase?*

### IPC micro-benchmark (784 B payload, Jetson Nano)

| Transport | Avg latency | Throughput |
|---|---|---|
| Unix socket | 0.076 ms | 13.1k msg/s |
| Shared memory | 0.050 ms | 20.0k msg/s |
| Pipe | **0.047 ms** | **21.2k msg/s** |

Shared memory is ~34% faster than Unix socket at MNIST frame size.
Full report: [`results/ipc_benchmark.json`](results/ipc_benchmark.json)

### Resource ladder (INT8, process RSS)

| Inferences | Avg latency | RSS | CPU% |
|---|---|---|---|
| 100 | 0.688 ms | 300.8 MB | ~100% |
| 1,000 | 0.687 ms | 300.8 MB | ~100% |
| 10,000 | 0.686 ms | 301.3 MB | ~100% |

RSS stays flat across the ladder — no growth signal in short runs.
Full report: [`results/resource_ladder.json`](results/resource_ladder.json)

### Sensor → inference service (Unix socket + TFLite INT8)

End-to-end RTT ≈ **1.28 ms** avg (p95 1.38 ms) @ ~779 fps for 100 frames.
[`results/inference_service.json`](results/inference_service.json)

```bash
# IPC: Unix socket vs shared memory vs pipe
python3.6 scripts/ipc_benchmark.py --output results/ipc_benchmark.json

# Resource ladder: 100 → 1k → 10k → 100k inferences
python3.6 scripts/benchmark.py --ladder 100,1000,10000,100000 \
  --output results/resource_ladder.json

# 30–60 min soak with RSS growth verdict
python3.6 scripts/soak_test.py --duration 1800 --window 5000 \
  --output results/soak_test_30min_rss.json

# Sensor simulator → TFLite inference service (Unix socket)
python3.6 scripts/inference_service.py --mode demo --frames 200
```

## Architecture

```text
Jetson Nano / ARM64 Linux
        │
        ├── Sensor Simulator (client)
        │
        ├── IPC
        │    ├── Unix Domain Socket
        │    ├── Shared Memory (/dev/shm + mmap)
        │    └── Pipe
        │
        ├── AI Inference Service
        │       └── TFLite INT8
        │
        └── System Monitor  (scripts/sysmon.py)
             ├── CPU %
             ├── RSS / Peak RSS
             ├── Temperature
             ├── Latency (avg / p95)
             └── Throughput
```

## What this project demonstrates

- TFLite deploy on embedded Linux (Jetson Nano) with real glibc/runtime trade-offs
- Latency benchmarking (avg / min / max / p95)
- **IPC benchmarking** for low-latency frame handoff
- **CPU / RSS / thermal** sampling from `/proc` (no extra deps)
- **Soak stability verdict** (memory growth + latency drift)
- **Reproducible debug story** with intentional leak / high-CPU demos + `perf`/`strace`/`valgrind`

### Deployment note

On Jetson Nano (glibc 2.27), Coral `tflite_runtime` wheels require glibc ≥ 2.29 and
fail to load. This project uses **NVIDIA TensorFlow 2.7** built-in
`tf.lite.Interpreter` instead — a common real-world trade-off on embedded platforms.

## Quick start (Jetson Nano)

```bash
git clone https://github.com/kennyliuu/edge-ai-deployment-benchmark.git
cd edge-ai-deployment-benchmark
bash scripts/run_jetson.sh
```

Requires: `python3.6`, NVIDIA TensorFlow 2.7 wheel, `numpy`.

### Individual commands

```bash
# Latency + RSS/CPU fields
python3.6 scripts/benchmark.py --output results/jetson_nano_2gb.json

# Soak (5 min default; use --duration 1800/3600 for long runs)
python3.6 scripts/soak_test.py --duration 300 --output results/soak_test.json

# Intentional bug demos (see docs/PROFILING.md)
bash scripts/bugs/run_memory_leak_soak.sh
bash scripts/bugs/run_high_cpu_service.sh
```

### Train & export (optional, on a PC)

```bash
python3 scripts/train_and_export.py
```

## Project structure

```text
├── models/                 # TFLite FP32 / INT8
├── results/                # JSON reports from Jetson
├── docs/
│   ├── SYSTEM_PERF_STABILITY_PLAN.md
│   ├── PROFILING.md
│   └── examples/leak_demo.c
└── scripts/
    ├── sysmon.py               # CPU / RSS / temp helpers
    ├── benchmark.py            # Latency + resource ladder
    ├── soak_test.py            # Soak + stability verdict
    ├── ipc_benchmark.py        # Socket / SHM / pipe
    ├── inference_service.py    # Sensor client ↔ TFLite service
    ├── run_jetson.sh           # Smoke suite
    └── bugs/                   # Intentional leak / high-CPU wrappers
```

## Author

Kenny Liu — [GitHub](https://github.com/kennyliuu)
