# Datasets

Real-world GPU-cluster traces that ground GPUSitter's telemetry, failure
classification, and remediation in production data rather than pure simulation.

---

## ⚠️ AcmeTrace reality check — READ BEFORE WIRING THE PIPELINE (verified 2026-06-27)

Hands-on inspection of the downloaded data contradicts several common assumptions
about AcmeTrace. **These are the ground truth:**

1. **Job trace and fine-grained telemetry barely overlap in time (~1.5 days).**
   AcmeTrace is two loosely-coupled logs at different granularities:
   - `trace_kalos.csv` = the **scheduler log over the full 6 months** (job rows,
     incl. `fail_time`). FAILED jobs span **May 17 → Aug 16 2023 (UTC)**.
   - `acme-util/data/utilization/kalos/*.csv` = **15-second DCGM telemetry**, only
     released as a **~2-week profiling snapshot: Aug 15 → Aug 31 2023 (+08:00)**.
   Storing 15s × 2,344 GPUs × ~11 metrics for 6 months would be tens of TB, so
   only the snapshot was published. **Result: only ~113 of 13,836 FAILED jobs
   fall inside the telemetry window.** You generally *cannot* time-join an
   arbitrary failure to its telemetry. Timezones also differ (+08 vs +00) and
   must be normalized. This is a property of the public release, not a bad download.

2. **There is NO `NODE_FAIL` state in Kalos.** States are `COMPLETED` (47,311),
   `FAILED` (13,836), `CANCELLED` (1,263), `RUNNING` (3). A filter of
   `{NODE_FAIL, FAILED}` only ever matches `FAILED`. Use `FAILED` + non-null `fail_time`.

3. **Timestamps are ISO strings (UTC), not Unix epoch.** e.g.
   `2023-05-17 11:00:58+00:00`. Parse with `pd.to_datetime(...)`, not as int seconds.

4. **Real Xid codes DO exist** — `acme-util/data/utilization/kalos/XID_ERRORS.csv`
   has per-GPU, per-timestamp Xid codes (43 = GPU stopped processing, 94 = contained
   ECC, 45, 31 = memory page fault). The classifier/agent can read *actual* fault codes
   rather than inferring cause. (It's a DCGM gauge holding the *last* Xid, so a faulted
   GPU repeats its code every sample until cleared — events ≠ raw nonzero-cell count.)

5. **The `acme/data/utilization/util_pkl/*.pkl` files are NOT time series** — they are
   precomputed **CDF distributions** (lists of 1000-point arrays for the paper's plots).
   The `acme-util/.../ipmi/*.csv` files are empty stubs (~133 B). The real, usable,
   time-indexed telemetry lives **only** in `acme-util/data/utilization/kalos/*.csv`
   (`Time` column + 2,344 per-GPU columns named `<node-ip>-<gpu-idx>`, e.g. `172.31.13.235-0`).

**Recommended incident source:** drive incidents off `XID_ERRORS.csv` within the Aug
window. A nonzero Xid is a real, timestamped fault on a specific GPU, and the *same
window* of `GPU_TEMP` / `POWER_USAGE` / `GPU_UTIL` for that GPU is right there to
correlate — fully self-consistent and grounded, sidestepping the overlap problem.

---

Fetch everything with:

```bash
scripts/download_datasets.sh        # default: pai + acme
scripts/download_datasets.sh all    # + philly (best-effort) + extra
scripts/download_datasets.sh --list # names only
```

All data lands in **`data/`**, which is **gitignored** (too large to commit —
~4 GB). The script is idempotent: existing datasets are skipped, so re-running is
safe and cheap. PAI tarballs are checksum-verified (SHA-256) before extraction.

### DigitalOcean Testing & Storage Setup
We run our active python services and test suite directly on our DigitalOcean Droplet CPU VM:
- **Droplet Specs**: 4 GB RAM / 2 Intel vCPUs / 120 GB Storage / SFO3 - Ubuntu 24.04 (LTS) x64 (IP: `134.199.208.214`).
- **Memory & Swap Configuration**: Cloning large datasets like `acme-util` via Git LFS requires substantial memory when indexing and checking out. The clone process was initially terminated by the Linux Out-Of-Memory (OOM) killer because the VM only had 4 GB of RAM (and no swap space enabled by default).
  To resolve this, we created and enabled a **16 GB swap file** on the VM.
- **Disk Optimization via `git lfs fetch`**: The full `acme-util` dataset is ~80 GB. Checking out these files in the working directory while keeping the Git LFS cache would require ~160 GB, exceeding the Droplet's 120 GB disk capacity. To bypass this, the repository was cloned with `GIT_LFS_SKIP_SMUDGE=1` and downloaded using **`git lfs fetch`**. This downloads only the raw cache objects (~80 GB), fitting safely on the VM's disk.
- **Accessing Cache directly**: We created a utility script `scripts/lfs_helper.py` that lets us read files directly from the `.git/lfs/objects/` cache in Python (e.g. into Pandas) without checking them out, preserving the 0-duplicate footprint. (See [docs/TEAM_GUIDE.md](TEAM_GUIDE.md) for usage details).

```
hack/
├── scripts/download_datasets.sh
└── data/                  # gitignored
    ├── pai/               # Alibaba PAI 2020  (full CSVs + .header sidecars)
    ├── acme/              # Acme LLM trace    (git repo: InternLM/AcmeTrace)
    ├── acme-util/         # Acme FULL ~80 GB raw util (opt-in; HuggingFace)
    ├── philly/            # Philly 2017       (repo only; data blob blocked)
    └── clusterdata/       # alibaba/clusterdata (newer traces v2023/25/26)
```

---

## What gets downloaded

| Target      | Source | On-disk | Workload | In default / `all` |
|-------------|--------|--------:|----------|:------------------:|
| `pai`       | Aliyun OSS (`v2020GPUTraces`) | ~3.9 GB | Mixed DL train + inference, **fractional GPU sharing** | default |
| `acme`      | GitHub `InternLM/AcmeTrace` | ~116 MB | **LLM development** — job traces + processed util pkls | default |
| `philly`    | GitHub `msr-fiddle/philly-traces` | ~2 MB (repo only) | DNN training, 2017 | `all` |
| `extra`     | GitHub `alibaba/clusterdata` | ~3.7 GB | K8s GPU-share, DLRM, GenAI serving (tarballs auto-extracted) | `all` |
| `acme-util` | HuggingFace `Qinghao/AcmeTrace` | **~80 GB** | Acme FULL — raw DCGM/Prometheus/IPMI utilization + power | `everything` only |

`all` = pai + acme + philly + extra. `everything` = `all` + acme-util.
`acme-util` requires **git-lfs**; everything else needs only `curl` + `git`.

> **Philly is best-effort.** Its 1.1 GB `trace-data.tar.gz` sits behind GitHub
> LFS whose budget is exhausted (HTTP 403) — only Microsoft can restore it. The
> script clones the repo (README + analysis notebook) and *attempts* `git lfs
> pull`, then reports the failure without aborting. **Use `pai` as the closest
> structural analog.**

### Running on a remote analysis box

The script self-resolves paths and uses only HTTPS sources (no SSH keys), so a
clean run reproduces the full set:

```bash
git clone <this-repo> && cd hack
scripts/download_datasets.sh all          # pai + acme + philly + extra (~7.5 GB)
scripts/download_datasets.sh everything   # + acme-util ~80 GB (needs git-lfs + disk)
DATA_DIR=/mnt/data scripts/download_datasets.sh everything   # custom data root
```

It is idempotent (re-runs skip what's present) and verifies PAI checksums, so it
is safe to re-run after an interrupted transfer. The only data it cannot fetch is
the Philly blob (server-side LFS block).

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

**Most relevant for GPUSitter:** `pai_sensor_table` (per-worker GPU/mem
utilization → DCGM-style telemetry) and `pai_machine_metric` (per-machine
health signals). Failure signal: `status` fields (`Failed`, `Terminated`).

## 2. Acme — `data/acme/` (LLM workloads)

Shanghai AI Lab "Acme" datacenter, Mar–Aug 2023 (NSDI'24, Hu et al.). Two
clusters with explicit job **failure states** and queueing — useful for
failure-pattern and resilience modeling.

- `data/acme/data/job_trace/trace_seren.csv` — 818K jobs (mixed research/eval)
- `data/acme/data/job_trace/trace_kalos.csv` — 62K jobs (large LLM pretraining, avg 27 GPU/job)
- `data/acme/data/cluster_summary.csv` — normalized cross-trace stats (Philly/PAI/Helios/Acme)
- `data/acme/data/utilization/util_pkl/` — **CDF distribution pickles** (lists of
  1000-pt arrays for the paper's plots), *not* time series. Real time-indexed telemetry
  is `data/acme-util/data/utilization/kalos/*.csv` (see reality check at top).

Columns (headers present): `job_id, user, node_num, gpu_num, cpu_num, type,
state, submit_time, start_time, end_time, duration, queue, gpu_time`.
**Timestamps are wall-clock ISO** (`2023-03-01 00:18:22+08:00`).
In **Kalos specifically**, the only observed states are `COMPLETED`, `FAILED`,
`CANCELLED`, `RUNNING` — **there is no `NODE_FAIL`** (despite broader AcmeTrace
docs listing more). The incident signal is `FAILED` + non-null `fail_time`.
Timestamps are ISO UTC strings, not epoch. (See reality check at top.)

> The `acme` target pulls only job traces + processed pickles. The raw ~80 GB
> DCGM/Prometheus/IPMI utilization + power data lives on HuggingFace
> `Qinghao/AcmeTrace`; fetch it with the **`acme-util`** target (needs git-lfs)
> into `data/acme-util/`.

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
- `cluster-trace-v2026-GenAI/` — **Stable Diffusion serving** trace (app latency / queue / GPU duty-cycle; LoRA + ControlNet). Ships as 14 `.tar.gz`; the downloader auto-extracts them to CSV.
- `cluster-trace-gpu-v2026/` — scripts only

> The repo's own `cluster-trace-gpu-v2020/data/` holds only the download script
> + headers + a 100K sample. The **full** v2020 data is under `data/pai/`; the
> downloader symlinks those CSVs into this path so clusterdata's bundled v2020
> analysis notebook + simulator work unchanged.

---

## Choosing a trace

| Goal | Use |
|------|-----|
| GPU telemetry / failure detection (GPUSitter core) | **PAI** sensor + machine_metric |
| LLM-cluster failure/queue dynamics | **Acme** (Kalos = pretraining scale) |
| Scheduling / GPU bin-packing | **PAI** or **v2023** (K8s) |
| Inference / serving systems | **v2026-GenAI** |
| Philly-style analysis (drop-in) | **PAI** |

Cross-trace normalized stats live in `data/acme/data/cluster_summary.csv`.
