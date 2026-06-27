# Datasets

Real-world GPU-cluster traces that ground EvoSentinel DC's telemetry, failure
classification, and remediation in production data rather than pure simulation.

Fetch everything with:

```bash
scripts/download_datasets.sh        # default: pai + acme
scripts/download_datasets.sh all    # + philly (best-effort) + extra
scripts/download_datasets.sh --list # names only
```

All data lands in **`data/`**, which is **gitignored** (too large to commit —
~4 GB). The script is idempotent: existing datasets are skipped, so re-running is
safe and cheap. PAI tarballs are checksum-verified (SHA-256) before extraction.

```
hack/
├── scripts/download_datasets.sh
└── data/                  # gitignored
    ├── pai/               # Alibaba PAI 2020  (full CSVs + .header sidecars)
    ├── acme/              # Acme LLM trace    (git repo: InternLM/AcmeTrace)
    ├── philly/            # Philly 2017       (repo only; data blob blocked)
    └── clusterdata/       # alibaba/clusterdata (newer traces v2023/25/26)
```

---

## What gets downloaded

| Target  | Source | On-disk | Workload | Pulled by default |
|---------|--------|--------:|----------|:-----------------:|
| `pai`   | Aliyun OSS (`v2020GPUTraces`) | ~3.9 GB | Mixed DL train + inference, **fractional GPU sharing** | ✅ |
| `acme`  | GitHub `InternLM/AcmeTrace` | ~116 MB | **LLM development** (pretrain / eval / debug) | ✅ |
| `philly`| GitHub `msr-fiddle/philly-traces` | ~2 MB (repo only) | DNN training, 2017 | ❌ (`all`) |
| `extra` | GitHub `alibaba/clusterdata` | ~3.7 GB | K8s GPU-share, DLRM, GenAI serving | ❌ (`all`) |

> **Philly is best-effort.** Its 1.1 GB `trace-data.tar.gz` sits behind GitHub
> LFS whose budget is exhausted (HTTP 403) — only Microsoft can restore it. The
> script clones the repo (README + analysis notebook) and *attempts* `git lfs
> pull`, then reports the failure without aborting. **Use `pai` as the closest
> structural analog.**

---

## 1. PAI 2020 — `data/pai/` (primary)

Alibaba Platform for AI production cluster, Jul–Aug 2020. ~1.26M jobs, ~1,800
machines (P100 / T4 / V100 / V100M32 + CPU nodes). The closest analog to Philly:
same job → util → machine triad, plus **fractional GPU allocation** (avg 0.68
GPU/job — multiple jobs share one device).

**CSVs are headerless by design.** Each `<table>.csv` pairs with a
`<table>.header` sidecar (written by the downloader) listing its columns. Load with:

```python
import pandas as pd
cols = open("data/pai/pai_sensor_table.header").read().strip().split(",")
df = pd.read_csv("data/pai/pai_sensor_table.csv", names=cols)
```

| Table | Rows | Columns (key fields) |
|-------|-----:|----------------------|
| `pai_job_table` | 1.06M | job_name, inst_id, user, status, start_time, end_time |
| `pai_task_table` | 1.26M | + task_name, inst_num, **plan_cpu/plan_mem/plan_gpu**, gpu_type |
| `pai_instance_table` | 7.52M | + inst_name, worker_name, **machine** (placement) |
| `pai_sensor_table` | 3.03M | **gpu_wrk_util, cpu_usage, avg/max_mem, avg/max_gpu_wrk_mem, read/write** |
| `pai_group_tag_table` | 1.06M | inst_id, gpu_type_spec, group, **workload** |
| `pai_machine_spec` | 1,897 | machine, gpu_type, **cap_cpu/cap_mem/cap_gpu** |
| `pai_machine_metric` | 2.01M | **machine_cpu_*, machine_gpu, machine_load_1, machine_net_receive** |

Join keys: `job_name`, `inst_id`, `worker_name`, `machine`.
**Timestamps are relative seconds** (floats from trace start), *not* wall-clock.

**Most relevant for EvoSentinel:** `pai_sensor_table` (per-worker GPU/mem
utilization → DCGM-style telemetry) and `pai_machine_metric` (per-machine
health signals). Failure signal: `status` fields (`Failed`, `Terminated`).

## 2. Acme — `data/acme/` (LLM workloads)

Shanghai AI Lab "Acme" datacenter, Mar–Aug 2023 (NSDI'24, Hu et al.). Two
clusters with explicit job **failure states** and queueing — useful for
failure-pattern and resilience modeling.

- `data/acme/data/job_trace/trace_seren.csv` — 818K jobs (mixed research/eval)
- `data/acme/data/job_trace/trace_kalos.csv` — 62K jobs (large LLM pretraining, avg 27 GPU/job)
- `data/acme/data/cluster_summary.csv` — normalized cross-trace stats (Philly/PAI/Helios/Acme)
- `data/acme/data/utilization/util_pkl/` — processed GPU util / temp / power pickles

Columns (headers present): `job_id, user, node_num, gpu_num, cpu_num, type,
state, submit_time, start_time, end_time, duration, queue, gpu_time`.
**Timestamps are wall-clock ISO** (`2023-03-01 00:18:22+08:00`).
`state` ∈ {COMPLETED, FAILED, CANCELLED, …} — the failure label.

> The raw ~80 GB utilization/power data is **not** pulled here; it lives on
> HuggingFace `Qinghao/AcmeTrace`. Only the processed pickles in-repo are local.

## 3. Philly — `data/philly/` (reference only)

Microsoft Philly DNN trace, Aug–Dec 2017 (ATC'19). 117K GPU jobs. The original
inspiration, but **the data blob is unobtainable** (LFS 403). Only the README +
`analysis/Philly Trace Analysis.ipynb` are fetched. Schema (for reference):
job log JSON with scheduling attempts + per-minute GPU/CPU/mem util CSVs.

## 4. Extra Alibaba traces — `data/clusterdata/`

Full `alibaba/clusterdata` repo. Beyond PAI 2020, contains newer GPU traces
committed directly in-repo:

- `cluster-trace-gpu-v2023/` — Kubernetes GPU-**sharing** scheduling (pod/node lists, `gpu_milli`, QoS)
- `cluster-trace-gpu-v2025/` — disaggregated **DLRM** serving (role-split, RDMA req/limit)
- `cluster-trace-v2026-GenAI/` — **Stable Diffusion serving** trace (app latency / queue / GPU duty-cycle; LoRA + ControlNet)
- `cluster-trace-gpu-v2026/` — scripts only

> The repo's own `cluster-trace-gpu-v2020/data/` holds only the download script
> + headers + a 100K sample. The **full** v2020 data is under `data/pai/`; the
> downloader symlinks those CSVs into this path so clusterdata's bundled v2020
> analysis notebook + simulator work unchanged.

---

## Choosing a trace

| Goal | Use |
|------|-----|
| GPU telemetry / failure detection (EvoSentinel core) | **PAI** sensor + machine_metric |
| LLM-cluster failure/queue dynamics | **Acme** (Kalos = pretraining scale) |
| Scheduling / GPU bin-packing | **PAI** or **v2023** (K8s) |
| Inference / serving systems | **v2026-GenAI** |
| Philly-style analysis (drop-in) | **PAI** |

Cross-trace normalized stats live in `data/acme/data/cluster_summary.csv`.
