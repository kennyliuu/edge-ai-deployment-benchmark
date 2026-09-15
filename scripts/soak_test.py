#!/usr/bin/env python3
"""Long-running TFLite soak test with RSS growth / stability verdict."""

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


def window_stats(latencies_ms):
    arr = np.array(latencies_ms)
    return {
        "latency_avg_ms": round(float(np.mean(arr)), 3),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 3),
        "latency_max_ms": round(float(np.max(arr)), 3),
    }


def stability_verdict(windows, elapsed_sec, rss_growth_mb_threshold=2.0, drift_pct_threshold=10.0):
    """
    Answer: Does memory usage continuously increase?

    Thresholds (documented for interview / README):
      - RSS growth over the run < rss_growth_mb_threshold → not a leak signal
      - |latency drift| < drift_pct_threshold → latency stable
      - errors == 0 handled by caller
    """
    if len(windows) < 2:
        return {
            "memory_continuously_increasing": None,
            "stable": None,
            "reason": "need >= 2 windows",
        }

    rss_vals = [w.get("rss_mb") for w in windows if w.get("rss_mb") is not None]
    first = windows[0]
    last = windows[-1]
    rss_first = first.get("rss_mb")
    rss_last = last.get("rss_mb")
    rss_growth = None
    if rss_first is not None and rss_last is not None:
        rss_growth = round(rss_last - rss_first, 2)

    # Simple trend: last half mean RSS vs first half mean RSS
    increasing = None
    if len(rss_vals) >= 4:
        mid = len(rss_vals) // 2
        first_mean = float(np.mean(rss_vals[:mid]))
        last_mean = float(np.mean(rss_vals[mid:]))
        # continuously increasing if last half clearly above first and growth exceeds threshold
        increasing = (last_mean - first_mean) > rss_growth_mb_threshold

    first_avg = first.get("latency_avg_ms")
    last_avg = last.get("latency_avg_ms")
    drift = None
    if first_avg and last_avg is not None and first_avg > 0:
        drift = round((last_avg - first_avg) / first_avg * 100.0, 2)

    latency_ok = drift is None or abs(drift) < drift_pct_threshold
    memory_ok = rss_growth is None or rss_growth < rss_growth_mb_threshold
    if increasing is True:
        memory_ok = False

    # Only extrapolate per-hour growth for runs long enough to be meaningful.
    hours = elapsed_sec / 3600.0 if elapsed_sec else 0
    growth_per_hour = None
    if rss_growth is not None and elapsed_sec >= 60 and hours > 0:
        growth_per_hour = round(rss_growth / hours, 2)

    answer = "unknown"
    if increasing is True:
        answer = "yes"
    elif increasing is False or (rss_growth is not None and rss_growth < rss_growth_mb_threshold):
        answer = "no"

    return {
        "question": "Does memory usage continuously increase?",
        "answer": answer,
        "memory_continuously_increasing": increasing if increasing is not None else (answer == "yes"),
        "stable": bool(memory_ok and latency_ok),
        "rss_mb_first": rss_first,
        "rss_mb_last": rss_last,
        "rss_growth_mb": rss_growth,
        "rss_growth_per_hour_mb": growth_per_hour,
        "latency_drift_pct": drift,
        "temp_c_first": first.get("cpu_temp_c"),
        "temp_c_last": last.get("cpu_temp_c"),
        "temp_delta_c": (
            round(last["cpu_temp_c"] - first["cpu_temp_c"], 1)
            if first.get("cpu_temp_c") is not None and last.get("cpu_temp_c") is not None
            else None
        ),
        "thresholds": {
            "rss_growth_mb": rss_growth_mb_threshold,
            "latency_drift_pct": drift_pct_threshold,
        },
        "reason": (
            "RSS growth %.2f MB over %.1fs; latency drift %s%%"
            % (rss_growth if rss_growth is not None else -1,
               elapsed_sec,
               drift if drift is not None else "n/a")
        ),
    }


def soak_model(model_path, duration_sec=None, max_runs=None, window=1000, warmup=10,
               inject_leak_bytes=0):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    dtype = inp["dtype"]
    shape = tuple(inp["shape"])
    if dtype == np.uint8:
        x = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    else:
        x = np.random.rand(*shape).astype(np.float32)

    interpreter.set_tensor(inp["index"], x)
    for _ in range(warmup):
        interpreter.invoke()

    leak_bag = []  # intentional leak sink (for bug demos)
    windows = []
    window_latencies = []
    errors = 0
    total = 0
    cpu = sysmon.CpuSampler()
    cpu.sample_pct()
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
            window_latencies.append((time.perf_counter() - t_start) * 1000.0)
            if inject_leak_bytes > 0:
                leak_bag.append(bytearray(inject_leak_bytes))
        except Exception:
            errors += 1

        total += 1
        if len(window_latencies) >= window:
            elapsed = time.perf_counter() - t0
            entry = {
                "window": len(windows),
                "inferences": total,
                "elapsed_sec": round(elapsed, 1),
                "rss_mb": sysmon.process_rss_mb(),
                "peak_rss_mb": sysmon.process_peak_rss_mb(),
                "mem_available_mb": sysmon.mem_available_mb(),
                "cpu_temp_c": sysmon.cpu_temp_c(),
                "cpu_pct": cpu.sample_pct(),
            }
            entry.update(window_stats(window_latencies))
            windows.append(entry)
            window_latencies = []
            print("window %d: inf=%d rss=%s MB avg=%.3f ms" % (
                entry["window"], total, entry["rss_mb"], entry["latency_avg_ms"]))

    if window_latencies:
        elapsed = time.perf_counter() - t0
        entry = {
            "window": len(windows),
            "inferences": total,
            "elapsed_sec": round(elapsed, 1),
            "rss_mb": sysmon.process_rss_mb(),
            "peak_rss_mb": sysmon.process_peak_rss_mb(),
            "mem_available_mb": sysmon.mem_available_mb(),
            "cpu_temp_c": sysmon.cpu_temp_c(),
            "cpu_pct": cpu.sample_pct(),
        }
        entry.update(window_stats(window_latencies))
        windows.append(entry)

    elapsed = time.perf_counter() - t0
    summary = {
        "total_inferences": total,
        "elapsed_sec": round(elapsed, 1),
        "errors": errors,
        "throughput_ips": round(total / elapsed, 1) if elapsed > 0 else 0,
        "peak_rss_mb": sysmon.process_peak_rss_mb(),
    }
    if len(windows) >= 2:
        first_avg = windows[0]["latency_avg_ms"]
        last_avg = windows[-1]["latency_avg_ms"]
        summary["latency_avg_ms_first_window"] = first_avg
        summary["latency_avg_ms_last_window"] = last_avg
        if first_avg > 0:
            summary["latency_drift_pct"] = round((last_avg - first_avg) / first_avg * 100, 2)

    verdict = stability_verdict(windows, elapsed)
    summary["stability"] = verdict

    return {
        "model": os.path.basename(model_path),
        "model_size_kb": round(os.path.getsize(model_path) / 1024.0, 1),
        "config": {
            "duration_sec": duration_sec,
            "max_runs": max_runs,
            "window": window,
            "warmup": warmup,
            "inject_leak_bytes": inject_leak_bytes,
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
    parser.add_argument("--duration", type=int, default=300,
                        help="Run time in seconds (default: 300; try 1800/3600)")
    parser.add_argument("--max-runs", type=int, default=None, help="Stop after N inferences")
    parser.add_argument("--window", type=int, default=1000, help="Snapshot every N inferences")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--inject-leak-bytes", type=int, default=0,
                        help="DEBUG: leak N bytes per inference (for bug demos)")
    args = parser.parse_args()

    if args.max_runs is not None:
        duration_sec = None
        max_runs = args.max_runs
    else:
        duration_sec = args.duration
        max_runs = None

    model_path = os.path.join(args.models_dir, args.model)
    if not os.path.exists(model_path):
        raise SystemExit("Model not found: %s" % model_path)

    print("Soak test: %s (duration=%ss, max_runs=%s, window=%s)" % (
        args.model, duration_sec, max_runs, args.window))
    soak = soak_model(
        model_path,
        duration_sec=duration_sec,
        max_runs=max_runs,
        window=args.window,
        warmup=args.warmup,
        inject_leak_bytes=args.inject_leak_bytes,
    )

    report = {"device": get_platform_info(), "soak": soak}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["soak"]["summary"], indent=2))
    stab = report["soak"]["summary"].get("stability", {})
    print("Stability verdict: %s — %s" % (
        stab.get("answer"), stab.get("reason")))
    print("Saved to %s" % args.output)


if __name__ == "__main__":
    main()
