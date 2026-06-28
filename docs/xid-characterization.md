# Kalos Xid fault characterization + precursor analysis

*Bead: aieng26hack-eku. Produced by `scripts/characterize_xid.py` over the real
kalos DCGM telemetry on the droplet (gitignored, ~3.5 GB streamed, never
densified). Reproduce with:*

```bash
python3 scripts/characterize_xid.py \
    --data-dir data/acme-util/data/utilization/kalos \
    --out docs/xid_characterization.json
```

This defines the **detectable signal + label scheme** that early-incident
detection (aieng26hack-5fq) and the GPUSitter env's fault dynamics
(aieng26hack-w28) build on. Numbers below are the committed
`docs/xid_characterization.json`.

> **Onset detector.** Event extraction delegates to the canonical
> `gpusitter.rca.job_join.stream_xid_onset_records` — the same edge detector the
> runtime RCA/detection path uses — so this characterization's labels are by
> construction identical to what detection sees.

---

## TL;DR

1. **852 true Xid onset events on 851 GPUs over 15.7 days** (Aug 15–31 2023,
   +08:00), a fleet of 2,344 GPUs. Aggregate MTBF ≈ **1,036 GPU-hours per
   event**. A further **135 GPUs are already faulted at the first sample (t0)**
   and are **excluded as left-censored** (their onset predates the window).
2. **One fault type dominates: Xid 43 (GPU stopped processing / fell off the
   bus) = 794 of 852 (93%).** The rest: Xid 31 memory/MMU fault (55), Xid 94
   contained ECC (2), Xid 45 soft cleanup (1). No uncontained-ECC (95) or
   double-bit (48) in this window.
3. **An *onset* is a transition INTO a fault, not a nonzero cell.**
   `XID_ERRORS.csv` is a DCGM gauge that holds the last code and re-emits it every
   15 s until cleared; a healthy/cleared GPU is an **empty** cell (the trace has
   no explicit `0`s), so empty reads as idle/cleared. An onset is a nonzero where
   the GPU's prior observed state was non-fault. Detection is **empty-aware**: a
   fault that clears (to empty) and re-raises the **same** code is a *new* onset.
   Latched faults (nonzero→nonzero, including a code change with no intervening
   clear) are **not** separate onsets. Counting nonzero cells or nonzero-GPU
   snapshots massively overcounts.
4. **Faults are front-loaded, not spread evenly:** 96% (822/852) land in Aug
   15–18; the tail (Aug 19–31) is sporadic (~30 events). The largest *genuinely
   correlated* burst is **Aug-16 00:32:15–30: 115 GPUs across 78 nodes in a
   single 15 s sample** (43×106 + 31×9) — the best candidate "hero" incident.
5. **There is no usable multi-minute precursor in temp/power/clock.** The only
   telemetry correlate of a fault is a *coincident collapse* of power (median
   −36…−41 W) and a slight temp drop (~−1 °C); `MEM_CLOCK` is pinned (zero
   variance) and carries no signal. Only **416/851 event GPUs** emit any
   telemetry in the 2 h before their Xid event; the rest are already dark. Early
   detection cannot lean on smooth thermal/power ramps — it must use the Xid
   signal itself and spatial/temporal correlation.

---

## 1. Event distribution

| Xid | Meaning | Onsets | Severity group |
|----:|---------|-------:|----------------|
| 43 | GPU stopped processing (fell off bus / hang) | 794 | `fatal_hang` |
| 31 | GPU memory page fault (MMU / illegal address) | 55 | `memory_fault` |
| 94 | Contained ECC error | 2 | `contained_ecc` |
| 45 | Preemptive cleanup (robust channel; often app-induced) | 1 | `soft_cleanup` |

- Total: **852 onset events** on **851 distinct GPUs** ⇒ ~1 event/GPU. Once a GPU
  faults, its gauge holds the code and (in this window) it rarely clears and
  re-faults, so recurrence is near-zero and event-count ≈ distinct-faulted-GPUs.
- **Left-censoring:** 135 GPUs are nonzero in the very first sample (t0). Their
  true onset is outside the window, so they are **not** counted — counting them
  would manufacture a rising edge at the window boundary. Reported as
  `observed.n_left_censored`.

## 2. Frequency / MTBF

- Observed GPU-time: 2,344 GPUs × 15.69 d = **882,506 GPU-hours**.
- **MTBF ≈ 1,035.8 GPU-hours per event** (0.000966 events / GPU-hour).
- **Caveat:** events are front-loaded (Aug 15–18) and gauges hold, so this is
  *not* a steady-state rate. Read it as "≈1 Xid per 1,000 GPU-hours over a
  2-week profiling snapshot," dominated by an early rough patch, not a per-GPU
  reliability constant.

Daily onset counts:

```
Aug15:63  Aug16:323  Aug17:324  Aug18:112    <- 96% (822/852) of all events
Aug19-31: ~30 total  (11, then sporadic single/low digits per day)
```

## 3. Correlated bursts

The largest single-bin (15 s) burst:

- **Aug-16 00:32:15 → 00:32:30**, **115 GPUs across 78 nodes**, codes 43×106 +
  31×9 — 13.5% of all events in one sample. A cluster-wide correlated event
  (shared infra / job-cascade signature), the strongest demo "hero" incident
  candidate. *(Feeds aieng26hack-6xk.)*
- The remaining **737 events are sporadic** (outside that single largest bin).

### On reading a snapshot of the held gauge

A naive "how many GPUs are nonzero right now?" count late in the window returns
~850+, because the gauge **holds** every code raised since Aug 15: that count is
the *cumulative* faulted set, not a simultaneous event. The number is
flat-to-decreasing through the back half of the window (GPUs clearing), with only
a handful of *new* onsets per day after Aug 18. Reading such a snapshot as a
single cluster-wide event is the same nonzero-cell trap as §TL;DR#3 — the only
true correlated cluster-wide burst is **Aug-16 00:32** (115 GPUs / 78 nodes in
one 15 s tick, true onsets). aieng26hack-6xk should anchor there.

## 4. Precursors (the central question for detection)

For each onset, `scripts/characterize_xid.py` compares the GPU's own pre-fault
window (lead horizons 60 / 300 / 600 s) against its baseline (median over the
prior 2 h, excluding the final 10 min). GPU_TEMP / POWER_USAGE / MEM_CLOCK share
the IP-named namespace, so they join to XID per-GPU (SM_ACTIVE is pod-named with
no deterministic IP↔pod map and is excluded). Windows are keyed by *event*
(`gpu, t_event`), so repeated faults on one GPU are scored independently.

| Metric | n (events w/ baseline) | median Δ vs baseline @300 s | frac detectable @2σ (60/300/600 s) |
|--------|----:|----:|----|
| GPU_TEMP | 416 | −1.2 °C | 0.075 / 0.043 / 0.034 |
| POWER_USAGE | 411 | −36.5 W | 0.071 / 0.017 / 0.022 |
| MEM_CLOCK | 414 | 0.0 | 0.0 / 0.0 / 0.0 |

**Interpretation:**

- **MEM_CLOCK is pinned** (datacenter GPUs run memory at a fixed clock) → zero
  variance, zero signal. Drop it as a precursor feature.
- **POWER_USAGE and GPU_TEMP fall around the fault** (median −36 W @300 s,
  −41 W @600 s; −1.2 °C), consistent with the GPU winding down / dropping off
  the bus. But the *2σ-detectable* fraction is low (2–8%) because per-GPU
  power/temp baselines swing widely with the training workload, so a fixed 2σ
  rule rarely fires cleanly. The deviation is **coincident, not a multi-minute
  lead**, and the detectable fraction *falls* as the horizon lengthens — there is
  no smooth ramp building minutes ahead.
- **Only 416 / 851 event GPUs have any baseline telemetry in the 2 h
  pre-window** — the other ~435 are dark or not covered by the metric file. For
  the covered half, the fault is effectively instantaneous from the telemetry's
  point of view.

**Consequence for detection (5fq):** do not bank on thermal/power early-warning.
The actionable signals are (a) the Xid code itself the moment it rises, and (b)
**spatial/temporal correlation** — many GPUs/nodes faulting in one sample (the
Aug-16 burst) is a far stronger and earlier cluster-health signal than any
single GPU's temperature trend.

## 5. Label scheme (consumed by w28 + 5fq)

- **Fault onset** = a transition INTO a fault: a nonzero `XID_ERRORS` sample
  where the GPU's prior observed state was non-fault (healthy `0` or idle/empty
  cell). Detection is empty-aware (clear→same-code re-raise is a new onset).
  **Latched faults** (nonzero→nonzero, including a code change with no intervening
  clear) and **t0-left-censored** states are NOT onsets. Never count nonzero
  cells or nonzero-GPU snapshots. Canonical detector:
  `gpusitter.rca.job_join.stream_xid_onset_records`.
- **Severity groups:** `fatal_hang` {43, 79}, `fatal_ecc` {48, 95},
  `contained_ecc` {94}, `memory_fault` {31}, `soft_cleanup` {45}.
- **Positive-label horizon:** because no reliable lead exists, "pre-fault"
  positive windows for a precursor classifier should be treated as *weak* — the
  honest target is *detection at onset + correlation*, not minutes-ahead
  prediction. If a horizon label is still wanted, ≤60 s is the only band with any
  (marginal, ~8%) separability.
- **env fault dynamics (w28):** model the dominant mode as an abrupt Xid-43
  "fall off bus" with a coincident power/temp collapse (no gradual precursor),
  plus rare correlated bursts (e.g. ~100+ GPUs in one tick). MEM_CLOCK is
  constant.

---

*Raw output: `docs/xid_characterization.json` (includes the full 852-event list
with `{gpu, t, code}` for reuse by aieng26hack-p5x precursor deep-dive and
aieng26hack-6xk burst deep-dive).*
