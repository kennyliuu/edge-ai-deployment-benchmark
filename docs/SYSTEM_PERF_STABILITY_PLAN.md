# Edge AI System Performance & Stability Benchmark — Upgrade Plan

> Reposition this repo from **AI Deployment Benchmark** toward
> **System Performance + Embedded Linux + Stability**, without starting a new project.

Target interview narrative (Foxconn / automotive edge):

> Application → IPC → AI Runtime → System Resource → Performance → Stability

---

## 0. Current baseline (already on Jetson Nano 2GB)

| Item | Status | Evidence |
|---|---|---|
| TFLite FP32 / INT8 | Done | `models/`, `scripts/benchmark.py` |
| Latency avg / min / max / p95 | Done | `results/jetson_nano_2gb.json` |
| INT8 size −74%, latency ~−17% | Done | README |
| Soak (5 min / 30 min) | Partial | `scripts/soak_test.py`, `results/soak_test*.json` |
| System `MemAvailable` + thermal | Partial | soak windows only |
| Process RSS / peak / CPU % | Missing | — |
| Memory-growth verdict | Missing | only system-wide free mem |
| IPC (socket / shm / pipe) | Missing | single-process `invoke()` |
| Sensor / client ↔ inference service | Missing | — |
| `perf` / `strace` / Valgrind workflow | Missing | — |
| Intentional bug branches | Missing | — |

### What existing soak already shows (good foundation)

**5 min (`soak_test.json`):** 438,270 inferences, 0 errors, latency drift **0.0%**, MemAvailable 808→809 MB, temp 43.5→47.5°C.

**30 min (`soak_test_30min.json`):** 2,616,600 inferences, 0 errors, latency drift **−1.01%**, MemAvailable 809→808 MB, temp 47→50°C.

Interpretation for the upgrade:

- Inference itself looks stable under continuous load.
- Today we only track **system free memory**, not **process RSS / VmHWM**.
- That means we **cannot yet rigorously answer** “Does this service leak memory?” — the next milestone.

### Platform constraints to respect

- Jetson Nano 2GB · aarch64 · Ubuntu 18.04 · L4T R32.7.6
- Python **3.6.9** · TensorFlow **2.7** (`tf.lite.Interpreter`)
- ~2 GB RAM; soak already runs with ~800 MB available — keep tooling light
- Prefer stdlib / C; avoid new heavy packages

---

## 1. New project positioning

**Name / tagline**

`Edge AI System Performance & Stability Benchmark`

**Architecture (target)**

```text
Jetson Nano / ARM64 Linux
        │
        ├── Sensor Simulator (client)
        │
        ├── IPC
        │    ├── Unix Domain Socket
        │    ├── Shared Memory (+ semaphore / eventfd)
        │    └── Pipe (optional baseline)
        │
        ├── AI Inference Service
        │       └── TFLite INT8 (existing model)
        │
        └── System Monitor
             ├── CPU %
             ├── RSS / Peak RSS
             ├── Temperature
             ├── Latency (avg / p95)
             └── Throughput (msg/s, inf/s)
```

**Keep:** MNIST TFLite models, Jetson results, deployment notes (glibc / runtime trade-off).  
**Extend:** turn the in-process microbenchmark into a **multi-process edge service** under measurement.

---

## 2. Milestone plan (priority order)

Do **①–③** first. **④–⑤** are interview force-multipliers. Defer CAN / U-Boot.

### Milestone ① — Shared Memory + Unix Socket IPC benchmark ⭐⭐⭐⭐⭐

**Goal:** Compare IPC mechanisms under a fixed payload (MNIST `28×28` uint8 frame, plus optional larger synthetic frames).

**Suggested layout**

```text
ipc/
  include/ipc_common.h
  src/
    sensor_sim.c          # producer
    inference_service.c   # consumer → optional TFLite via Python bridge OR stub
    bench_socket.c
    bench_shm.c
    bench_pipe.c          # optional
  python/
    tflite_worker.py      # TFLite invoke process (Python 3.6 compatible)
  scripts/
    run_ipc_bench.sh
results/
  ipc_bench.json
```

**Metrics (required)**

| Metric | How |
|---|---|
| Average latency | producer send → consumer ack (or reply) |
| P95 latency | same samples |
| Throughput | messages / sec |
| CPU usage | `/proc/<pid>/stat` delta or `getrusage` |
| Payload size | fixed + sweep (e.g. 784 B, 4 KiB, 64 KiB) |

**Acceptance**

- One table in README: Socket vs SHM (± Pipe) on Jetson.
- Clear claim such as: “SHM cut p95 IPC latency by X% vs Unix socket at 64 KiB.”

**Interview line**

> I benchmarked Unix socket vs shared memory for feeding frames into an on-device TFLite service, and measured latency, throughput, and CPU on Jetson Nano.

---

### Milestone ② — CPU / Memory profiling in the hot path ⭐⭐⭐⭐⭐

**Goal:** Upgrade `benchmark.py` / `soak_test.py` (and the new service) from “model latency only” to **system-aware** metrics.

**Add samplers (Python 3.6-safe, `/proc` based — no new deps)**

| Signal | Source |
|---|---|
| Process RSS | `/proc/self/status` → `VmRSS` |
| Peak RSS | `VmHWM` |
| CPU % | `/proc/self/stat` utime+stime deltas |
| Temperature | existing `thermal_zone0` |
| Inference latency | existing |

**Workload ladder**

```text
100 → 1,000 → 10,000 → 100,000 inferences
```

Emit JSON like:

```json
{
  "runs": 10000,
  "latency_avg_ms": 0.69,
  "latency_p95_ms": 0.71,
  "cpu_percent": 97.2,
  "rss_mb": 142.5,
  "rss_peak_mb": 143.1,
  "temp_c": 48.0
}
```

**Acceptance**

- Resource columns appear in README results.
- Script works on Jetson Python 3.6 without extra packages.

---

### Milestone ③ — Stability / Soak upgrade (answer the leak question) ⭐⭐⭐⭐⭐

**Goal:** Extend existing soak into a **stability verdict**, not just a log.

**Changes to `soak_test.py`**

1. Snapshot **process RSS** every window (not only `MemAvailable`).
2. Support `--duration 1800` / `3600` as first-class presets.
3. Compute summary fields:

```text
rss_mb_first / rss_mb_last / rss_growth_mb
rss_growth_per_hour_mb
latency_drift_pct          (already have)
temp_delta_c
stable: true|false         (thresholds documented)
```

4. Explicit conclusion printed + stored:

> Does memory usage continuously increase? **No** (RSS +0.2 MB over 30 min).

**Acceptance**

- Re-run 30–60 min on Jetson with new fields.
- README has a “Stability verdict” subsection citing the JSON.

---

### Milestone ④ — `perf` / `strace` profiling playbook ⭐⭐⭐⭐

**Goal:** Document **measured** CPU hotspots for IPC modes + inference service.

**On Jetson (commands to capture into `docs/profiling/`)**

```bash
# CPU hotspots (service PID)
sudo perf top -p $PID
sudo perf record -F 99 -p $PID -- sleep 30
sudo perf report

# Syscall / IPC behavior
strace -c -p $PID
strace -tt -e trace=network,write,read,futex -p $PID -o strace_socket.txt
```

**Deliverables**

- `docs/profiling/perf_socket.md` vs `perf_shm.md` — top symbols, brief interpretation.
- Compare: socket path spends more time in `copy_to_user` / syscall; shm path dominated by TFLite / memcpy.

**Acceptance**

- At least one before/after or socket-vs-shm perf comparison checked into `docs/profiling/`.

---

### Milestone ⑤ — Intentional bugs + Valgrind / fix loop ⭐⭐⭐⭐

**Branches**

| Branch | Fault | Detect | Fix story |
|---|---|---|---|
| `bug/memory-leak` | append buffer each request; never free | Valgrind / RSS soak slope | free / reuse buffer → RSS flat |
| `bug/high-cpu` | busy-wait poll instead of blocking | `top` + `perf` + `strace` | blocking wait / eventfd |

**Workflow to document**

```text
inject bug → soak shows RSS growth / CPU pegged
    → valgrind / perf / strace
    → identify function
    → fix on main
    → re-benchmark
```

**Note for Jetson:** Valgrind on aarch64/JetPack can be heavy; if impractical on-device, run the same C IPC stubs under Valgrind on a host aarch64/x86 build, and confirm RSS growth on Jetson soak. Say this explicitly in docs — honesty beats fake screenshots.

---

### Deferred (not for first interview pass)

| Item | Why defer |
|---|---|
| CAN simulator | Nice JD keyword; not needed before IPC/stability story is solid |
| U-Boot / boot time | Belongs more with QEMU BSP track |
| Docker / systemd / GPU CUDA path | Out of current TFLite-CPU scope |
| New CV “vehicle detection” model | Weaker JD fit than system performance extension |

---

## 3. Suggested implementation order (time-boxed)

```text
Week slice A  →  Milestone ② resource samplers in soak/benchmark (fast win)
Week slice B  →  Milestone ① Unix socket vs SHM C bench + Jetson numbers
Week slice C  →  Wire sensor_sim → IPC → tflite_worker service
Week slice D  →  Milestone ③ 30–60 min soak with RSS growth verdict
Week slice E  →  Milestone ④ perf/strace notes
Week slice F  →  Milestone ⑤ bug/* branches + write narrative
```

Minimum interview-ready package = **A + B + D**.

---

## 4. Repo layout after upgrade

```text
edge-ai-deployment-benchmark/
  README.md                 # reposition + dual story: deploy + system perf
  docs/
    SYSTEM_PERF_STABILITY_PLAN.md   # this file
    profiling/              # perf / strace notes
  models/                   # unchanged TFLite
  scripts/
    benchmark.py            # + CPU/RSS
    soak_test.py            # + RSS growth verdict
    run_jetson.sh
    run_system_bench.sh     # new umbrella script
  ipc/                      # new
  results/
    jetson_nano_2gb.json
    soak_test_30min.json
    ipc_bench.json          # new
    system_profile.json     # new
```

---

## 5. README claims to unlock (after data exists)

1. **IPC:** Shared memory vs Unix socket latency / throughput / CPU on Jetson Nano.  
2. **Resources:** RSS, peak RSS, CPU %, temperature alongside inference latency.  
3. **Stability:** 30–60 min soak; RSS growth and latency drift quantified; explicit leak verdict.  
4. **Debug:** Used `perf` / `strace` / Valgrind on intentional fault branches and fixed them.

Portfolio one-liner:

> Built an ARM64 edge AI inference service on Jetson Nano and benchmarked IPC, CPU/memory, and long-running stability — including perf-driven diagnosis of injected faults.

---

## 6. Link to QEMU BSP track (separate repo)

```text
Linux-qemu-busybox (DT → driver → sysfs)
        ↓
Edge AI Deployment (TFLite / Jetson)
        ↓
This extension (IPC / CPU / Memory / soak / perf)
```

Do **not** merge QEMU into this repo; keep two clear artifacts, one career story.

---

## 7. Jetson execution checklist (when SSH is available)

```bash
# 0) Confirm device
uname -a; cat /etc/nv_tegra_release; free -h; tegrastats --interval 1000

# 1) Sync repo
cd ~/edge-ai-deployment-benchmark && git pull

# 2) Baseline (existing)
python3.6 scripts/benchmark.py --output results/jetson_nano_2gb.json
python3.6 scripts/soak_test.py --duration 300 --output results/soak_test.json

# 3) After Milestone ②/③ land
python3.6 scripts/soak_test.py --duration 1800 --output results/soak_test_30min_rss.json

# 4) After Milestone ① lands
bash ipc/scripts/run_ipc_bench.sh --output results/ipc_bench.json

# 5) Profiling artifacts
# (see Milestone ④ commands; save under docs/profiling/)
```

### SSH access needed from Cloud Agent

This planning pass ran in Cursor Cloud **without** Jetson host/credentials in the environment.
To execute on-device next, provide one of:

- `user@host` (LAN / Tailscale / public) + SSH key or allowlisted password
- Or run the checklist locally and push `results/*.json` back to the branch

---

## 8. Definition of done (interview package)

- [ ] README repositioned to System Performance & Stability  
- [ ] Socket vs SHM table with Jetson numbers  
- [ ] Soak JSON includes RSS growth + explicit leak verdict  
- [ ] At least one `docs/profiling/` note using `perf` or `strace`  
- [ ] Optional: `bug/memory-leak` branch + fix write-up  
