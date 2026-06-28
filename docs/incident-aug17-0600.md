# Incident characterization — Aug-17 06:00 correlated Xid burst (kalos)

**Bead:** `aieng26hack-6xk` · **Source:** read-only droplet pass over
`data/acme-util/data/utilization/kalos/` (134.199.208.214) ·
**Reproduce:** `python3 docs/incident-aug17-0600-analysis.py <repo-root>`
(stdlib only, streaming, read-only — copy to the droplet `/tmp` and run there).

> **Premise correction.** The originally-reported "Aug-29 13:57 882-GPU
> cluster-wide burst" is **not a real event** — it is a latched-state +
> window-edge artifact (XID is a latched DCGM gauge; a window starting mid-fault
> reads pre-latched GPUs as fresh onsets). Under empty-aware edge detection
> Aug-29 has **1** true onset. This bead is retargeted to the real, recurring
> correlated bursts; the **Aug-17 06:00** burst is the representative hero
> incident. Do not use Aug-29/882 anywhere.

All onset counts use the nav-approved empty-aware edge detector
(`stream_xid_onsets`, d8z): an **onset** is a transition *into* a fault — the
sample is nonzero and the GPU's prior **observed** state was non-fault (healthy
`0.0` or idle empty-cell). Latched faults (`nonzero→nonzero`, the dominant kalos
pattern) and GPUs whose first observation is already nonzero (pre-window history
unknown) are **excluded**. Full file: 852 empty-aware onsets over Aug-15..31.

---

## 1. Onset timeline (AC1)

Per-15s cluster-wide onset counts, 2023-08-17 05:55..06:05 (+08:00):

| sample (+08:00) | onsets |
|---|---|
| 05:55:00 .. 06:00:00 | **0** (every 15s sample) |
| **06:00:15** | **57** |
| **06:00:30** | **59** |
| 06:00:45 .. 06:05:00 | **0** (every 15s sample) |

The burst is **two adjacent 15s samples**, with clean baseline-0 neighbours on
both sides — **116 GPU-onsets within ~30 s**, then nothing. It is a genuine
near-instantaneous correlated event, not a window artifact.

## 2. Scope, union-deduped (AC2)

Union of the two burst samples:

- **116 distinct GPUs** across **74 distinct nodes**.
- GPUs-per-node distribution — `{1 GPU: 42 nodes, 2: 23, 3: 8, 4: 1}`;
  **max 4 of 8** GPUs on any single node.

The topology is **scattered**: most affected nodes lose only 1–2 of their 8
GPUs, and no node loses more than half. This argues **against** a per-node
power/PSU or whole-node cause, and **for** a cluster-level common trigger
(fabric / driver / scheduler-synchronized workload action) hitting GPUs broadly.

## 3. Code mix (AC3)

| Xid code | GPUs | meaning |
|---|---|---|
| **43** | **109** (94%) | GPU reset / channel exception |
| 31 | 7 (6%) | MMU / memory page fault |

Each affected GPU reports exactly one code; **43 is dominant**, matching the
fleet-wide pattern.

## 4. Recovery / latch behaviour (AC4)

For the 116 affected GPUs, latched-run length after onset (consecutive nonzero
15s samples until clear-to-0, idle, or window end):

- run length **min 59,521 · median 69,601 · max 69,602** samples (×15 s).
- terminal state: **115 GPUs stay faulted to the end of the telemetry window**
  (Aug-31); **1** goes idle (empty). **None were observed clearing to 0.**

So within the ~2-week telemetry window these GPUs **never show a recovery** —
the Xid code persists on the gauge from onset to the end of the trace.

> **Caveat (latched gauge).** XID_ERRORS holds the *last* Xid and repeats it
> every sample until explicitly cleared. "Faulted to window end" therefore means
> *the code was never cleared in telemetry*, not necessarily that the GPU was
> physically dead for ~12 days. But it does mean **no reset/clear event is
> observable** for these GPUs in the snapshot.

## 5. Temporal job correlation (AC5)

Method is **temporal-only**: trace_kalos records `fail_time` (UTC) and job-level
counts but **no per-GPU map** (the d8z gap), so we can only ask whether FAILED
jobs cluster near the burst. Burst centre 06:00:15+08:00 = **2023-08-16
22:00:15 UTC**; both sides parsed tz-aware.

- FAILED jobs (ISO `fail_time`) total: 13,836; **113** fall inside the telemetry
  window (matches DATA.md's overlap figure — pipeline sanity check).
- FAILED jobs within **±5 min**: **0**; within **±30 min**: **0**; within
  **±6 h**: **0**. Nearest FAILED job is **10.9 h** away.

**Finding:** this cluster-wide Xid burst is **decoupled from trace job-failures**
— no FAILED job coincides with it. Consistent with the d8z telemetry↔trace
decoupling and with these Xid 43 resets being handled without a logged job
failure.

## 6. Precursors (AC6)

GPU_TEMP / POWER_USAGE on affected GPUs over the ~5 min before 06:00:15,
delta(last−first):

| metric | GPUs (≥2 samples) | min | median | max | rising |
|---|---|---|---|---|---|
| GPU_TEMP (°C) | 86 | −18.0 | **0.0** | +16.0 | 24/86 |
| POWER_USAGE (W) | 89 | −342.8 | **0.0** | +309.3 | 40/89 |

Median delta ≈ 0 and rising≈falling for both signals: **no consistent thermal or
power lead signal** precedes the burst — the onset is effectively
**instantaneous**. This corroborates the lys/classifier finding that pre-Xid
warning signal in kalos telemetry is weak; a temp/power-only early-detector would
not anticipate this event.

## 7. Context — recurring bursts (AC7)

Empty-aware onset counts flag **8** correlated bursts (any 15s sample ≥ 40
onsets) over Aug-15..18 alone:

| start (+08:00) | samples | onsets | nodes |
|---|---|---|---|
| 2023-08-15 15:30:30 | 1 | 50 | 35 |
| 2023-08-16 00:32:15 | 2 | 115 | 78 |
| 2023-08-16 04:00:15 | 2 | 107 | 71 |
| 2023-08-16 18:48:15 | 2 | 100 | 69 |
| 2023-08-17 01:08:15 | 2 | 110 | 70 |
| 2023-08-17 03:12:15 | 2 | 98 | 68 |
| **2023-08-17 06:00:15** | **2** | **116** | **74** |
| 2023-08-18 05:44:15 | 2 | 111 | 75 |

These are **near-daily**, ~1–2 samples each, ~100–116 onsets over ~70 nodes,
same 43-dominant signature and scattered topology. **Aug-17 06:00 is the
largest and fully representative** — a recurring cluster-wide pattern, not a
one-off.

## 8. Integrity (AC8)

All numbers come from **read-only** streaming passes over the materialized kalos
CSVs (no checkout/clean/write to `data/acme-util`). Onset methodology is the
verified empty-aware `stream_xid_onsets` edge detector. The analysis script
(`docs/incident-aug17-0600-analysis.py`) is committed so every figure above is
falsifiable by re-running it. No edits to `backend/` or `src/`.

---

### Demo framing

A real, recurring, cluster-wide event: **116 GPUs across 74 nodes fault within
30 seconds**, Xid 43 dominant, scattered ≤4-of-8 per node (not a node-power
fault), **no thermal/power precursor**, **no clearing within the window**, and
**no coinciding job failure** in the trace. It is one of ~daily bursts — the kind
of correlated fault an on-call RCA agent must recognize and triage from
telemetry alone.
