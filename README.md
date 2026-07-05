# Edge AI Deployment Benchmark

End-to-end edge inference benchmark on **Jetson Nano 2GB**: TensorFlow Lite model deployment, latency measurement, and deployment notes for resource-constrained Linux devices.

Built for demonstrating **model → runtime → on-device inference** skills relevant to edge AI / SoC deployment roles.

## Test Platform (Jetson Nano 2GB)

Benchmarks in this repo were measured on an **NVIDIA Jetson Nano Developer Kit (2GB)**.

| Component | Specification |
|---|---|
| **SoC** | NVIDIA Tegra X1 (T210) |
| **CPU** | Quad-core ARM Cortex-A57 @ 1.43 GHz |
| **GPU** | 128-core NVIDIA Maxwell |
| **Memory** | 2 GB LPDDR4 (64-bit, 25.6 GB/s) |
| **Storage** | microSD card |
| **Connectivity** | Gigabit Ethernet |
| **OS** | Ubuntu 18.04 (L4T) |
| **JetPack / L4T** | JetPack 4.6.x · L4T R32.7.6 |
| **Architecture** | aarch64 |
| **Python / Runtime** | Python 3.6.9 · TensorFlow 2.7 (`tf.lite.Interpreter`) |

> **Note:** Jetson Nano 2GB has no NPU; inference in this project runs on **CPU** via TensorFlow Lite. This still reflects realistic edge-Linux deployment constraints (memory, glibc, runtime compatibility).

## Results (Jetson Nano 2GB)

| Model | Size | Input | Accuracy | Latency (avg) | Latency (p95) |
|---|---|---|---|---|---|
| `mnist_fp32.tflite` | 883 KB | float32 `[1,28,28]` | 98.84% | **0.83 ms** | 0.87 ms |
| `mnist_int8.tflite` | 228 KB | uint8 `[1,28,28]` | 98.87% | **0.69 ms** | 0.70 ms |

**INT8 vs FP32:** 74% smaller model, ~17% lower latency, no accuracy loss on MNIST test set.

Full report: [`results/jetson_nano_2gb.json`](results/jetson_nano_2gb.json)

## What this project demonstrates

- TensorFlow Lite inference on embedded Linux (Jetson Nano)
- Latency benchmarking (avg / min / max / p95 over 100 runs)
- Practical deployment constraints on edge devices (glibc, runtime choice, memory)

### Deployment note

On Jetson Nano (glibc 2.27), Coral `tflite_runtime` wheels require glibc ≥ 2.29 and fail to load. This project uses **NVIDIA TensorFlow 2.7** built-in `tf.lite.Interpreter` instead — a common real-world trade-off on embedded platforms.

A separate research CNN–LSTM captioning model could not be converted to TFLite due to **dynamic ops in the attention layer**; this MNIST pipeline validates the full conversion → deploy → benchmark loop on-device.

## Quick start

### On Jetson Nano

```bash
git clone https://github.com/kennyliuu/edge-ai-deployment-benchmark.git
cd edge-ai-deployment-benchmark
bash scripts/run_jetson.sh
```

Requires: `python3.6`, NVIDIA TensorFlow 2.7 wheel, `numpy`.

### Benchmark only

```bash
python3.6 scripts/benchmark.py --output results/jetson_nano_2gb.json
```

### Train & export (optional, run on PC with TensorFlow)

```bash
python3 scripts/train_and_export.py
```

Exports `models/mnist_fp32.tflite` and `models/mnist_int8.tflite`, then evaluates test accuracy.

## Project structure

```
├── models/                  # TFLite models
├── results/                 # Benchmark JSON reports
├── scripts/
│   ├── benchmark.py         # Latency & size benchmark
│   ├── train_and_export.py  # Train MNIST CNN → TFLite (optional)
│   └── run_jetson.sh        # One-command benchmark on Jetson
└── labels_mnist.txt
```

## Author

Kenny Liu — [GitHub](https://github.com/kennyliuu)
