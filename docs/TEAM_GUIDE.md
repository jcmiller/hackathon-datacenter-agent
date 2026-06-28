# GPUSitter GPU-Resilience Dataset Team Guide

This guide describes how to connect to the remote analysis VM, locate the datasets, and query/use the data (including the ~80 GB raw telemetry dataset) without running out of disk space.

---

## 1. Connecting to the VM

The datasets are hosted on a shared DigitalOcean VM:
* **IP Address**: `134.199.208.214`
* **Username**: `root`
* **SSH Command**:
  ```bash
  ssh root@134.199.208.214
  ```

---

## 2. Directory Layout & Datasets

All datasets land in `/root/hackathon-datacenter-agent/data/`:

| Dataset Directory | Dataset | Size | Description |
| :--- | :--- | :--- | :--- |
| **`data/pai/`** | Alibaba PAI 2020 | ~3.6 GB | Per-worker utilization sensors, fractional GPU share. CSVs + header sidecars. |
| **`data/acme/`** | Acme NSDI'24 (Job Traces) | ~116 MB | Standard job event traces with explicit failure states. |
| **`data/clusterdata/`** | Alibaba Extra Traces | ~127 MB | Kubernetes GPU-sharing (v2023), DLRM serving (v2025), Stable Diffusion serving (v2026-GenAI). |
| **`data/acme-util/`** | **Acme FULL Telemetry** | **~76 GB** | Raw DCGM, Prometheus, and IPMI utilization + power logs. |

---

## 3. How to Use the 76 GB `acme-util` Dataset (Without Disk Exhaustion)

Because the VM disk size is 120 GB, checking out the entire 76 GB LFS repository in the working directory while maintaining the git-lfs cache would require 150+ GB of disk space.

To avoid this, we cloned the repo skipping the "smudge" phase and downloaded the files directly into the git LFS cache (`.git/lfs/objects/`) using `git lfs fetch`.

### Method A: Read directly from LFS Cache in Python (Recommended)
You can read any LFS file directly from the cache without checking it out (which uses 0 bytes of extra disk space). We have provided a helper script `scripts/lfs_helper.py` in the repository containing the `get_lfs_cache_path` and `resolve_data_path` functions.

**Python Example:**
```python
import pandas as pd
import sys
sys.path.append("/root/hackathon-datacenter-agent/scripts")
from lfs_helper import get_lfs_cache_path

repo_dir = "/root/hackathon-datacenter-agent/data/acme-util"

# Resolve the cache path for a specific file (e.g. Seren cluster's GPU utilization telemetry)
file_rel_path = "data/utilization/seren/GPU_UTIL.csv"
resolved_path = get_lfs_cache_path(repo_dir, file_rel_path)

# Load directly into Pandas
print(f"Loading {file_rel_path} from cache path: {resolved_path}")
df = pd.read_csv(resolved_path)
print(df.head())
```

The helper also has CLI checks for the common Kalos metrics:

```bash
cd /root/hackathon-datacenter-agent
python scripts/lfs_helper.py kalos-status data/acme-util
python scripts/lfs_helper.py resolve data/acme-util data/utilization/kalos/XID_ERRORS.csv
```

This works whether the raw CSV is checked out, only a pointer file exists, or
the working-tree path has been deleted and only the Git LFS cache remains.

### Method B: Selective Checkout (File-by-File)
If you need specific files to appear physically in the folder `data/acme-util/data/...`, you can checkout only those files:

```bash
cd /root/hackathon-datacenter-agent/data/acme-util
git lfs pull --include="data/utilization/kalos/GPU_UTIL.csv"
```

To clean up a checked-out file and free working-tree space again, restore only
the specific heavy path you checked out:
```bash
git checkout -- data/utilization/kalos/GPU_UTIL.csv
```

Do not run broad cleanup commands inside `data/acme-util` unless you have checked
`git status --short`; that repository is often intentionally in a cache-only
state on the droplet.

### Cache-safe early-detection dataset

`scripts/build_early_dataset.py` builds the labeled early-detection dataset
(one row per `(gpu, t_ref)` prediction point; label = an Xid onset for that GPU
within the horizon; features = the lookback window *before* `t_ref`). It resolves
each raw kalos metric through `lfs_helper.resolve_data_path` (materialized file,
LFS pointer, or a deleted working path recovered from `git show`) and streams the
wide CSVs without materializing the ~80 GB frame — so it runs even when
`data/acme-util` is in the cache-only state above.

```bash
# from the repo root on the droplet (PYTHONPATH=src so `gpusitter` imports)
PYTHONPATH=src python scripts/build_early_dataset.py \
    --repo-dir data/acme-util \
    --metrics GPU_TEMP POWER_USAGE GPU_UTIL MEMORY_TEMP \
    --horizons 60 300 600 --lookback 600 \
    --gpu-batch-size 150 \
    --control-gpus 172.31.0.1#0 172.31.0.139#0 ... \
    --out data/early_detection.parquet
```

It writes parquet when `pyarrow` is installed, else a sibling `.csv`. The output
is the compact, reusable table model experiments should read — never re-scan the
raw telemetry per experiment. Check raw availability first with
`python scripts/lfs_helper.py kalos-status data/acme-util`.

**Memory (`--gpu-batch-size`).** `TelemetryStore` indexes every observed cell of
each loaded GPU's *full* timeline as Python objects. The real kalos set is 851
onset GPUs over a 15-day trace; loading them all at once needs >14 GB and the
OOM-killer reaps it on the 15 GB droplet. `--gpu-batch-size N` (default 150)
loads telemetry N GPUs at a time and frees each batch before the next — peak RSS
~1.7 GB at 150. It re-streams the feature CSVs once per batch (I/O traded for
memory); the output rows are identical to a single-load build.

**Cluster controls (`--control-gpus`).** Without controls the only negatives are
same-GPU pre-event points. Pass canonical `node#idx` ids of non-onset GPUs to add
time-matched cluster controls (a negative for each control at every positive
`t_ref`) for a production-style imbalanced dataset. Evaluate with
`scripts/eval_early_dataset.py` (time-ordered, GPU-grouped split; ROC-AUC, AP,
alert-budget, and a permuted no-signal baseline).

---

## 4. Querying PAI 2020 Data

The Alibaba PAI CSVs are headerless. They are paired with `<tablename>.header` files. To load them with the correct columns:

```python
import pandas as pd

header_path = "/root/hackathon-datacenter-agent/data/pai/pai_sensor_table.header"
csv_path = "/root/hackathon-datacenter-agent/data/pai/pai_sensor_table.csv"

cols = open(header_path).read().strip().split(",")
df = pd.read_csv(csv_path, names=cols)
print(df.head())
```
