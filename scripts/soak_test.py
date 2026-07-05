#!/usr/bin/env python3
"""Long-running TFLite inference soak test with periodic latency and resource snapshots."""

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


def mem_available_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


def cpu_temp_c():
    try:
        with open("/sys/devices/virtual/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000, 1)
    except OSError:
        return None


def window_stats(latencies_ms):
    arr = np.array(latencies_ms)
    return {
        "latency_avg_ms": round(float(np.mean(arr)), 3),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 3),
        "latency_max_ms": round(float(np.max(arr)), 3),
    }


def soak_model(model_path, duration_sec=None, max_runs=None, window=1000, warmup=10):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    shape = tuple(inp["shape"])
    dtype = inp["dtype"]
    if dtype == np.uint8:
        x = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    else:
        x = np.random.rand(*shape).astype(np.float32)

    interpreter.set_tensor(inp["index"], x)
    for _ in range(warmup):
        interpreter.invoke()

    windows = []
    window_latencies = []
    errors = 0
    total = 0
    t0 = time.perf_counter()
    deadline = t0 + duration_sec if duration_sec else None

    while True:
        if max_runs is not None and total >= max_runs:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break

        try:
            t_start = time.perf_counter()
            interpreter.invoke()
            window_latencies.append((time.perf_counter() - t_start) * 1000)
        except Exception:
            errors += 1

        total += 1
        if len(window_latencies) >= window:
            elapsed = time.perf_counter() - t0
            entry = {
                "window": len(windows),
                "inferences": total,
                "elapsed_sec": round(elapsed, 1),
                "mem_available_mb": mem_available_mb(),
                "cpu_temp_c": cpu_temp_c(),
            }
            entry.update(window_stats(window_latencies))
            windows.append(entry)
            window_latencies = []

    if window_latencies:
        elapsed = time.perf_counter() - t0
        entry = {
            "window": len(windows),
            "inferences": total,
            "elapsed_sec": round(elapsed, 1),
            "mem_available_mb": mem_available_mb(),
            "cpu_temp_c": cpu_temp_c(),
        }
        entry.update(window_stats(window_latencies))
        windows.append(entry)

    elapsed = time.perf_counter() - t0
    summary = {
        "total_inferences": total,
        "elapsed_sec": round(elapsed, 1),
        "errors": errors,
        "throughput_ips": round(total / elapsed, 1) if elapsed > 0 else 0,
    }
    if len(windows) >= 2:
        first_avg = windows[0]["latency_avg_ms"]
        last_avg = windows[-1]["latency_avg_ms"]
        summary["latency_avg_ms_first_window"] = first_avg
        summary["latency_avg_ms_last_window"] = last_avg
        if first_avg > 0:
            summary["latency_drift_pct"] = round((last_avg - first_avg) / first_avg * 100, 2)

    return {
        "model": os.path.basename(model_path),
        "model_size_kb": round(os.path.getsize(model_path) / 1024, 1),
        "config": {
            "duration_sec": duration_sec,
            "max_runs": max_runs,
            "window": window,
            "warmup": warmup,
        },
        "windows": windows,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="TFLite soak test on edge device")
    parser.add_argument(
        "--models-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "models"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "soak_test.json"),
    )
    parser.add_argument("--model", default="mnist_int8.tflite")
    parser.add_argument("--duration", type=int, default=300, help="Run time in seconds (default: 300)")
    parser.add_argument("--max-runs", type=int, default=None, help="Stop after N inferences")
    parser.add_argument("--window", type=int, default=1000, help="Snapshot every N inferences")
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if args.max_runs is not None:
        duration_sec = None
        max_runs = args.max_runs
    else:
        duration_sec = args.duration
        max_runs = None

    model_path = os.path.join(args.models_dir, args.model)
    if not os.path.exists(model_path):
        raise SystemExit(f"Model not found: {model_path}")

    print(f"Soak test: {args.model} (duration={duration_sec}s, max_runs={max_runs}, window={args.window})")
    soak = soak_model(
        model_path,
        duration_sec=duration_sec,
        max_runs=max_runs,
        window=args.window,
        warmup=args.warmup,
    )

    report = {"device": get_platform_info(), "soak": soak}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["soak"]["summary"], indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
