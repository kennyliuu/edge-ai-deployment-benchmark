#!/usr/bin/env python3
"""Benchmark TFLite models: latency and model size."""

import argparse
import json
import os
import platform
import time

import numpy as np
import tensorflow as tf


def get_platform_info():
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
    }
    try:
        with open("/etc/nv_tegra_release") as f:
            info["jetson_l4t"] = f.read().strip().split("\n")[0]
    except OSError:
        pass
    return info


def benchmark_model(model_path, runs=100, warmup=10):
    size_kb = os.path.getsize(model_path) / 1024
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    shape = tuple(inp["shape"])
    dtype = inp["dtype"]

    if dtype == np.uint8:
        scale, zero_point = inp["quantization"]
        x = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    else:
        x = np.random.rand(*shape).astype(np.float32)

    interpreter.set_tensor(inp["index"], x)
    for _ in range(warmup):
        interpreter.invoke()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "model": os.path.basename(model_path),
        "model_size_kb": round(size_kb, 1),
        "input_shape": [int(d) for d in shape],
        "input_dtype": str(dtype).replace("class 'numpy.", "").replace("'", ""),
        "latency_avg_ms": round(float(np.mean(times)), 3),
        "latency_min_ms": round(float(np.min(times)), 3),
        "latency_max_ms": round(float(np.max(times)), 3),
        "latency_p95_ms": round(float(np.percentile(times, 95)), 3),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "models"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "benchmark.json"),
    )
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    models = []
    for name in ("mnist_fp32.tflite", "mnist_int8.tflite"):
        path = os.path.join(args.models_dir, name)
        if os.path.exists(path):
            print(f"Benchmarking {name}...")
            models.append(benchmark_model(path, runs=args.runs))

    report = {
        "device": get_platform_info(),
        "models": models,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
