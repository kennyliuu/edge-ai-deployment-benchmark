#!/usr/bin/env python3
"""Benchmark TFLite models: latency, size, and CPU/memory resource use."""

from __future__ import print_function

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sysmon  # noqa: E402


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


def _make_input(inp):
    shape = tuple(inp["shape"])
    dtype = inp["dtype"]
    if dtype == np.uint8:
        return np.random.randint(0, 256, size=shape, dtype=np.uint8)
    return np.random.rand(*shape).astype(np.float32)


def benchmark_model(model_path, runs=100, warmup=10):
    size_kb = os.path.getsize(model_path) / 1024.0
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    shape = tuple(inp["shape"])
    dtype = inp["dtype"]
    x = _make_input(inp)

    interpreter.set_tensor(inp["index"], x)
    for _ in range(warmup):
        interpreter.invoke()

    cpu = sysmon.CpuSampler()
    cpu.sample_pct()
    rss_before = sysmon.process_rss_mb()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000.0)

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
        "cpu_pct": cpu.sample_pct(),
        "rss_mb": sysmon.process_rss_mb(),
        "rss_mb_before": rss_before,
        "peak_rss_mb": sysmon.process_peak_rss_mb(),
        "cpu_temp_c": sysmon.cpu_temp_c(),
        "mem_available_mb": sysmon.mem_available_mb(),
    }


def resource_ladder(model_path, ladder, warmup=10):
    """Run the same model at increasing inference counts and track RSS growth."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    x = _make_input(inp)
    interpreter.set_tensor(inp["index"], x)
    for _ in range(warmup):
        interpreter.invoke()

    points = []
    for runs in ladder:
        cpu = sysmon.CpuSampler()
        cpu.sample_pct()
        rss0 = sysmon.process_rss_mb()
        times = []
        t_wall0 = time.perf_counter()
        for _ in range(runs):
            t0 = time.perf_counter()
            interpreter.invoke()
            times.append((time.perf_counter() - t0) * 1000.0)
        wall = time.perf_counter() - t_wall0
        points.append({
            "runs": runs,
            "latency_avg_ms": round(float(np.mean(times)), 3),
            "latency_p95_ms": round(float(np.percentile(times, 95)), 3),
            "cpu_pct": cpu.sample_pct(),
            "rss_mb": sysmon.process_rss_mb(),
            "rss_mb_start": rss0,
            "peak_rss_mb": sysmon.process_peak_rss_mb(),
            "cpu_temp_c": sysmon.cpu_temp_c(),
            "throughput_ips": round(runs / wall, 1) if wall > 0 else 0,
        })
        print("  ladder %s runs: avg=%.3f ms rss=%.2f MB" % (
            runs, points[-1]["latency_avg_ms"], points[-1]["rss_mb"] or -1))
    return points


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
    parser.add_argument(
        "--ladder",
        default="",
        help="Comma-separated inference counts for resource ladder "
             "(e.g. 100,1000,10000,100000). Empty disables.",
    )
    parser.add_argument("--model", default="", help="If set with --ladder, only this model")
    args = parser.parse_args()

    models = []
    for name in ("mnist_fp32.tflite", "mnist_int8.tflite"):
        path = os.path.join(args.models_dir, name)
        if os.path.exists(path):
            print("Benchmarking %s..." % name)
            models.append(benchmark_model(path, runs=args.runs))

    report = {
        "device": get_platform_info(),
        "models": models,
    }

    if args.ladder.strip():
        ladder = [int(x) for x in args.ladder.split(",") if x.strip()]
        model_name = args.model or "mnist_int8.tflite"
        path = os.path.join(args.models_dir, model_name)
        if not os.path.exists(path):
            raise SystemExit("Model not found for ladder: %s" % path)
        print("Resource ladder on %s: %s" % (model_name, ladder))
        report["resource_ladder"] = {
            "model": model_name,
            "points": resource_ladder(path, ladder),
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("Saved to %s" % args.output)


if __name__ == "__main__":
    main()
