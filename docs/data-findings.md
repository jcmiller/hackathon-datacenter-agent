# GPUSitter — Data Findings (Acme kalos)

Repo mirror of the Notion data page ("Acme kalos — schema, location & patterns").
Source: droplet exploration, embead bead `vf8`.

## Source & access
- Droplet `134.199.208.214`: `/root/hackathon-datacenter-agent/data/acme-util` (Acme HF dataset `Qinghao/AcmeTrace`).
- **Access (team standard):** read LFS files directly from the `.git/lfs` cache via `scripts/lfs_helper.py` (`get_lfs_cache_path`) — 0 extra disk, no checkout; see `docs/TEAM_GUIDE.md`. (A ~6.5 GB kalos subset was also checked out during exploration; seren ~70 GB stays cache-only.)
- Real **DCGM** wide CSVs: `Time × ~2,344 GPUs`, **78,843 timestamps @ ~15 s**, Aug 2023. GPU id = `<node>-<gpuidx>`; empty cell = idle.
- Metrics: `GPU_TEMP`, `GPU_UTIL`, `POWER_USAGE`, `MEMORY_TEMP`, `MEM_CLOCK`, `SM_ACTIVE`/`SM_OCCUPANCY`, `FB_USED`/`FREE`, `DRAM_ACTIVE`, `PIPE_TENSOR_ACTIVE`, `NODE_*`, `XID_ERRORS` + IPMI power.

## Patterns
- **Aug-29 13:57–14:00 — cluster-wide correlated failure:** 882 GPUs across 141 nodes fire Xid simultaneously (~6 of 8/node), sustained minutes → the demo hero incident; validates `find_correlated_failures`.
- **Xid 43** (channel exception / GPU reset) dominant, then `31` (mem page fault), `94` (contained ECC), `45` (preemptive cleanup).
- **891 / 2,344 GPUs (~38%)** hit an Xid over the window.
- **Job states (`trace_kalos`):** COMPLETED 47,311 / **FAILED 13,836 (~22%)** / CANCELLED 1,263 / RUNNING 3.

## Caveats
- `XID_ERRORS` is a **latched state** (repeats each sample until cleared) → count **0→nonzero transitions** for true events.
- `trace_kalos` ≠ `trace_seren` columns (`state` = col 9; kalos adds `mem_per_pod_GB`, `shared_mem_per_pod`, `fail_time`, `stop_time`).
- GPU-id naming inconsistent across metrics (IP-based vs pod-based) → needs normalization to join.
- Wide format → melt to long `(t, gpu, metric, value)`.

## Open beads
`q2o` ingest · `zxp` mining · `d8z` job↔telemetry RCA join · `p5x` precursors · `6xk` Aug-29 event · `eku` Xid characterization.
