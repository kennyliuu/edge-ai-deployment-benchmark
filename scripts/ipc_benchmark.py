#!/usr/bin/env python3
"""
IPC micro-benchmark: Unix domain socket vs POSIX shared memory vs pipe.

Measures round-trip / transfer latency, P95, throughput, and CPU/RSS for a
fixed payload size — the kind of IPC comparison used in low-latency edge
pipelines (sensor → inference service).

Python 3.6 compatible. Shared memory uses /dev/shm + mmap (no multiprocessing
SharedMemory required).
"""

from __future__ import print_function

import argparse
import json
import mmap
import os
import platform
import socket
import struct
import sys
import tempfile
import time

import numpy as np

# Allow `python3 scripts/ipc_benchmark.py` without installing a package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sysmon  # noqa: E402


HDR = struct.Struct("!I")  # payload length


def get_platform_info():
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        with open("/etc/nv_tegra_release") as f:
            info["jetson_l4t"] = f.read().strip().split("\n")[0]
    except OSError:
        pass
    return info


def stats_ms(times_s):
    arr = np.array(times_s) * 1000.0
    return {
        "latency_avg_ms": round(float(np.mean(arr)), 4),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 4),
        "latency_max_ms": round(float(np.max(arr)), 4),
        "latency_min_ms": round(float(np.min(arr)), 4),
    }


def _recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise IOError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def bench_unix_socket(payload, iters, warmup):
    path = os.path.join(tempfile.gettempdir(), "eai_ipc_%d.sock" % os.getpid())
    if os.path.exists(path):
        os.unlink(path)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)

    pid = os.fork()
    if pid == 0:
        # Server: echo payload
        srv.close()
        cli = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        cli.connect(path)
        try:
            while True:
                hdr = _recv_exact(cli, HDR.size)
                (n,) = HDR.unpack(hdr)
                if n == 0:
                    break
                data = _recv_exact(cli, n)
                cli.sendall(HDR.pack(n) + data)
        finally:
            cli.close()
        os._exit(0)

    conn, _ = srv.accept()
    srv.close()
    times = []
    cpu = sysmon.CpuSampler()
    cpu.sample_pct()
    try:
        for i in range(warmup + iters):
            t0 = time.perf_counter()
            conn.sendall(HDR.pack(len(payload)) + payload)
            rh = _recv_exact(conn, HDR.size)
            (n,) = HDR.unpack(rh)
            _recv_exact(conn, n)
            dt = time.perf_counter() - t0
            if i >= warmup:
                times.append(dt)
        # shutdown
        conn.sendall(HDR.pack(0))
    finally:
        conn.close()
        os.waitpid(pid, 0)
        if os.path.exists(path):
            os.unlink(path)

    elapsed = sum(times)
    result = stats_ms(times)
    result.update({
        "transport": "unix_socket",
        "iters": iters,
        "payload_bytes": len(payload),
        "throughput_mps": round(iters / elapsed, 1) if elapsed > 0 else 0,
        "throughput_mibs": round((iters * len(payload) / elapsed) / (1024 * 1024), 2) if elapsed > 0 else 0,
        "cpu_pct": cpu.sample_pct(),
        "rss_mb": sysmon.process_rss_mb(),
    })
    return result


def bench_pipe(payload, iters, warmup):
    r1, w1 = os.pipe()  # client → server
    r2, w2 = os.pipe()  # server → client
    pid = os.fork()
    if pid == 0:
        os.close(w1)
        os.close(r2)
        try:
            while True:
                hdr = os.read(r1, HDR.size)
                if len(hdr) < HDR.size:
                    break
                (n,) = HDR.unpack(hdr)
                if n == 0:
                    break
                data = b""
                while len(data) < n:
                    chunk = os.read(r1, n - len(data))
                    if not chunk:
                        break
                    data += chunk
                os.write(w2, HDR.pack(n) + data)
        finally:
            os.close(r1)
            os.close(w2)
        os._exit(0)

    os.close(r1)
    os.close(w2)
    times = []
    cpu = sysmon.CpuSampler()
    cpu.sample_pct()
    try:
        for i in range(warmup + iters):
            t0 = time.perf_counter()
            os.write(w1, HDR.pack(len(payload)) + payload)
            hdr = os.read(r2, HDR.size)
            (n,) = HDR.unpack(hdr)
            got = b""
            while len(got) < n:
                got += os.read(r2, n - len(got))
            dt = time.perf_counter() - t0
            if i >= warmup:
                times.append(dt)
        os.write(w1, HDR.pack(0))
    finally:
        os.close(w1)
        os.close(r2)
        os.waitpid(pid, 0)

    elapsed = sum(times)
    result = stats_ms(times)
    result.update({
        "transport": "pipe",
        "iters": iters,
        "payload_bytes": len(payload),
        "throughput_mps": round(iters / elapsed, 1) if elapsed > 0 else 0,
        "throughput_mibs": round((iters * len(payload) / elapsed) / (1024 * 1024), 2) if elapsed > 0 else 0,
        "cpu_pct": cpu.sample_pct(),
        "rss_mb": sysmon.process_rss_mb(),
    })
    return result


def bench_shared_memory(payload, iters, warmup):
    """
    Ping-pong over a POSIX shm file with a small control pipe for sequencing.
    Layout: [seq:u32][len:u32][payload...]
    """
    shm_path = "/dev/shm/eai_ipc_%d" % os.getpid()
    slot = 8 + len(payload)
    fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, slot)
    mm = mmap.mmap(fd, slot)
    os.close(fd)

    # One-byte handshake pipes: A notifies B, B notifies A.
    r_ab, w_ab = os.pipe()
    r_ba, w_ba = os.pipe()

    pid = os.fork()
    if pid == 0:
        os.close(w_ab)
        os.close(r_ba)
        try:
            while True:
                sig = os.read(r_ab, 1)
                if not sig or sig == b"x":
                    break
                # read request from shm, write response (echo) in place
                mm.seek(0)
                _seq, n = struct.unpack("!II", mm.read(8))
                data = mm.read(n)
                mm.seek(0)
                mm.write(struct.pack("!II", _seq, n))
                mm.write(data)
                mm.flush()
                os.write(w_ba, b"1")
        finally:
            os.close(r_ab)
            os.close(w_ba)
            mm.close()
        os._exit(0)

    os.close(r_ab)
    os.close(w_ba)
    times = []
    cpu = sysmon.CpuSampler()
    cpu.sample_pct()
    try:
        for i in range(warmup + iters):
            t0 = time.perf_counter()
            mm.seek(0)
            mm.write(struct.pack("!II", i + 1, len(payload)))
            mm.write(payload)
            mm.flush()
            os.write(w_ab, b"1")
            os.read(r_ba, 1)
            mm.seek(0)
            _seq, n = struct.unpack("!II", mm.read(8))
            _ = mm.read(n)
            dt = time.perf_counter() - t0
            if i >= warmup:
                times.append(dt)
        os.write(w_ab, b"x")
    finally:
        os.close(w_ab)
        os.close(r_ba)
        os.waitpid(pid, 0)
        mm.close()
        try:
            os.unlink(shm_path)
        except OSError:
            pass

    elapsed = sum(times)
    result = stats_ms(times)
    result.update({
        "transport": "shared_memory",
        "iters": iters,
        "payload_bytes": len(payload),
        "throughput_mps": round(iters / elapsed, 1) if elapsed > 0 else 0,
        "throughput_mibs": round((iters * len(payload) / elapsed) / (1024 * 1024), 2) if elapsed > 0 else 0,
        "cpu_pct": cpu.sample_pct(),
        "rss_mb": sysmon.process_rss_mb(),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="IPC latency/throughput benchmark")
    parser.add_argument("--payload-bytes", type=int, default=784,
                        help="Payload size (default: 784 = MNIST 28x28)")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--transports",
        default="unix_socket,shared_memory,pipe",
        help="Comma-separated: unix_socket,shared_memory,pipe",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "ipc_benchmark.json"),
    )
    args = parser.parse_args()

    payload = os.urandom(args.payload_bytes)
    transports = [t.strip() for t in args.transports.split(",") if t.strip()]
    runners = {
        "unix_socket": bench_unix_socket,
        "shared_memory": bench_shared_memory,
        "pipe": bench_pipe,
    }

    results = []
    for name in transports:
        if name not in runners:
            raise SystemExit("Unknown transport: %s" % name)
        print("Benchmarking %s..." % name)
        results.append(runners[name](payload, args.iters, args.warmup))

    # Rank by average latency for a quick summary.
    ranked = sorted(results, key=lambda r: r["latency_avg_ms"])
    summary = {
        "fastest": ranked[0]["transport"],
        "latency_avg_ms_by_transport": {r["transport"]: r["latency_avg_ms"] for r in results},
        "throughput_mps_by_transport": {r["transport"]: r["throughput_mps"] for r in results},
    }

    report = {
        "device": get_platform_info(),
        "config": {
            "payload_bytes": args.payload_bytes,
            "iters": args.iters,
            "warmup": args.warmup,
            "transports": transports,
        },
        "results": results,
        "summary": summary,
    }

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("Saved to %s" % out)


if __name__ == "__main__":
    main()
