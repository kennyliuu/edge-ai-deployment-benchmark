#!/usr/bin/env python3
"""Process and system resource sampling for edge Linux (Python 3.6+)."""

from __future__ import print_function

import os
import time


def _clk_tck():
    try:
        return os.sysconf("SC_CLK_TCK")
    except (ValueError, OSError, AttributeError):
        return 100


def cpu_temp_c():
    paths = (
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone0/temp",
    )
    for path in paths:
        try:
            with open(path) as f:
                raw = int(f.read().strip())
            # Some boards report millidegrees, some degrees.
            return round(raw / 1000.0, 1) if raw > 200 else float(raw)
        except (OSError, ValueError):
            continue
    return None


def mem_available_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return None


def process_rss_mb(pid=None):
    """Resident set size of this process (or pid) in MiB."""
    if pid is None:
        pid = os.getpid()
    try:
        with open("/proc/%d/status" % pid) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 2)
    except OSError:
        pass
    return None


def process_peak_rss_mb(pid=None):
    if pid is None:
        pid = os.getpid()
    try:
        with open("/proc/%d/status" % pid) as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024.0, 2)
    except OSError:
        pass
    return None


class CpuSampler(object):
    """Approximate process CPU% between two samples using /proc/<pid>/stat."""

    def __init__(self, pid=None):
        self.pid = os.getpid() if pid is None else pid
        self._prev = None
        self._prev_t = None
        self._hz = float(_clk_tck())

    def _read_jiffies(self):
        with open("/proc/%d/stat" % self.pid) as f:
            fields = f.read().split()
        # utime + stime
        return int(fields[13]) + int(fields[14])

    def sample_pct(self, min_dt=0.2):
        """Return CPU% since last sample. Needs >= min_dt seconds for a stable reading."""
        now = time.time()
        try:
            jiffies = self._read_jiffies()
        except OSError:
            return None
        if self._prev is None:
            self._prev = jiffies
            self._prev_t = now
            return None
        dt = now - self._prev_t
        if dt < min_dt:
            return None
        dj = jiffies - self._prev
        self._prev = jiffies
        self._prev_t = now
        return round((dj / self._hz) / dt * 100.0, 1)


def snapshot(cpu_sampler=None, pid=None):
    """One resource snapshot suitable for JSON reports."""
    data = {
        "rss_mb": process_rss_mb(pid),
        "peak_rss_mb": process_peak_rss_mb(pid),
        "mem_available_mb": mem_available_mb(),
        "cpu_temp_c": cpu_temp_c(),
    }
    if cpu_sampler is not None:
        data["cpu_pct"] = cpu_sampler.sample_pct()
    return data
