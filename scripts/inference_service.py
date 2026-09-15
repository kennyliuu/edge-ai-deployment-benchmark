#!/usr/bin/env python3
"""
Unix-socket TFLite inference service + sensor-simulator client.

Architecture demo for the interview narrative:

  Sensor Simulator  --Unix socket-->  AI Inference Service (TFLite)
                                              │
                                        System Monitor (RSS/CPU)

Python 3.6 compatible. Protocol: length-prefixed frames + JSON reply.
"""

from __future__ import print_function

import argparse
import json
import os
import socket
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sysmon  # noqa: E402

HDR = struct.Struct("!I")


def _recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise IOError("connection closed")
        buf.extend(chunk)
    return bytes(buf)


def load_interpreter(model_path):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    return interpreter, inp, out


def serve(sock_path, model_path, inject_leak_bytes=0, busy_spin_us=0):
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    interpreter, inp, out = load_interpreter(model_path)
    shape = tuple(inp["shape"])
    dtype = inp["dtype"]
    expected = int(np.prod(shape))
    elem = 1 if dtype == np.uint8 else 4
    expected_bytes = expected * elem

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    print("inference_service listening on %s (model=%s, expect %d bytes)" % (
        sock_path, os.path.basename(model_path), expected_bytes))

    leak_bag = []
    total = 0
    cpu = sysmon.CpuSampler()
    cpu.sample_pct()

    try:
        while True:
            conn, _ = srv.accept()
            try:
                while True:
                    hdr = _recv_exact(conn, HDR.size)
                    (n,) = HDR.unpack(hdr)
                    if n == 0:
                        break
                    payload = _recv_exact(conn, n)
                    t0 = time.perf_counter()
                    if len(payload) != expected_bytes:
                        reply = {"ok": False, "error": "bad payload size",
                                 "got": len(payload), "want": expected_bytes}
                    else:
                        if dtype == np.uint8:
                            arr = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
                        else:
                            arr = np.frombuffer(payload, dtype=np.float32).reshape(shape)
                        interpreter.set_tensor(inp["index"], arr)
                        interpreter.invoke()
                        if busy_spin_us > 0:
                            # Intentional CPU burn for bug/high-cpu demos.
                            deadline = time.perf_counter() + busy_spin_us / 1e6
                            while time.perf_counter() < deadline:
                                pass
                        if inject_leak_bytes > 0:
                            leak_bag.append(bytearray(inject_leak_bytes))
                        y = interpreter.get_tensor(out["index"])
                        pred = int(np.argmax(y))
                        lat = (time.perf_counter() - t0) * 1000.0
                        total += 1
                        reply = {
                            "ok": True,
                            "pred": pred,
                            "latency_ms": round(lat, 3),
                            "inferences": total,
                            "rss_mb": sysmon.process_rss_mb(),
                            "cpu_pct": cpu.sample_pct(),
                        }
                    body = json.dumps(reply).encode("utf-8")
                    conn.sendall(HDR.pack(len(body)) + body)
            except (IOError, OSError):
                pass
            finally:
                conn.close()
    finally:
        srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)


def client_benchmark(sock_path, model_path, frames, warmup, payload_override=None):
    import tensorflow as tf  # only to read input dtype/shape if needed
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    shape = tuple(inp["shape"])
    dtype = inp["dtype"]

    cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Wait briefly for server
    for _ in range(50):
        try:
            cli.connect(sock_path)
            break
        except OSError:
            time.sleep(0.05)
    else:
        raise SystemExit("cannot connect to %s" % sock_path)

    times = []
    try:
        for i in range(warmup + frames):
            if payload_override is not None:
                payload = payload_override
            elif dtype == np.uint8:
                payload = np.random.randint(0, 256, size=shape, dtype=np.uint8).tobytes()
            else:
                payload = np.random.rand(*shape).astype(np.float32).tobytes()
            t0 = time.perf_counter()
            cli.sendall(HDR.pack(len(payload)) + payload)
            rh = _recv_exact(cli, HDR.size)
            (n,) = HDR.unpack(rh)
            body = _recv_exact(cli, n)
            dt = time.perf_counter() - t0
            if i >= warmup:
                times.append(dt)
                if (i - warmup) % max(1, frames // 5) == 0:
                    msg = json.loads(body.decode("utf-8"))
                    print("frame %d: rtt=%.3f ms service=%s" % (
                        i - warmup, dt * 1000.0, msg))
        cli.sendall(HDR.pack(0))
    finally:
        cli.close()

    arr = np.array(times) * 1000.0
    return {
        "frames": frames,
        "latency_avg_ms": round(float(np.mean(arr)), 3),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 3),
        "latency_max_ms": round(float(np.max(arr)), 3),
        "throughput_fps": round(frames / sum(times), 1) if times else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="TFLite inference service / sensor client")
    parser.add_argument("--mode", choices=("serve", "client", "demo"), default="demo")
    parser.add_argument("--socket", default="/tmp/eai_inference.sock")
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(__file__), "..", "models", "mnist_int8.tflite"),
    )
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--inject-leak-bytes", type=int, default=0)
    parser.add_argument("--busy-spin-us", type=int, default=0,
                        help="Extra busy-wait per inference (high-CPU demo)")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "inference_service.json"),
    )
    args = parser.parse_args()

    if args.mode == "serve":
        serve(args.socket, args.model, args.inject_leak_bytes, args.busy_spin_us)
        return

    if args.mode == "client":
        result = client_benchmark(args.socket, args.model, args.frames, args.warmup)
        print(json.dumps(result, indent=2))
        return

    # demo: fork server, run client, collect JSON
    pid = os.fork()
    if pid == 0:
        serve(args.socket, args.model, args.inject_leak_bytes, args.busy_spin_us)
        os._exit(0)

    try:
        time.sleep(0.3)
        result = client_benchmark(args.socket, args.model, args.frames, args.warmup)
        report = {
            "mode": "sensor_simulator_to_inference_service",
            "ipc": "unix_socket",
            "model": os.path.basename(args.model),
            "result": result,
            "resources": sysmon.snapshot(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        print("Saved to %s" % args.output)
    finally:
        try:
            # nudge server to exit by connecting and sending 0 — may already be gone
            pass
        except Exception:
            pass
        os.kill(pid, 15)
        os.waitpid(pid, 0)
        if os.path.exists(args.socket):
            try:
                os.unlink(args.socket)
            except OSError:
                pass


if __name__ == "__main__":
    main()
