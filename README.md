# Edge AI Deployment Benchmark

End-to-end edge inference benchmark on **Jetson Nano 2GB**: TensorFlow Lite model deployment, latency measurement, and deployment notes for resource-constrained Linux devices.

Built for demonstrating **model → runtime → on-device inference** skills relevant to edge AI / SoC deployment roles.

## Results (Jetson Nano 2GB)

| Model | Size | Input | Latency (avg) | Latency (p95) |
|---|---|---|---|---|
| `mnist_fp32.tflite` | 2.6 MB | `[1, 28, 28]` float32 | **0.66 ms** | 0.70 ms |

**Device:** Jetson Nano, L4T R32.7.6, Ubuntu 18.04, Python 3.6, TensorFlow 2.7 (`tf.lite.Interpreter`)

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

Exports `models/mnist_fp32.tflite` and `models/mnist_int8.tflite` (if supported).

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
