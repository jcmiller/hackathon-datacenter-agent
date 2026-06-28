# Dashboard data substrate (bead t7p)

A small, committed, **derived-real** artifact the dashboard APIs serve without
scanning the ~80 GB raw kalos telemetry per click. It is *generated*, not
hand-authored: `scripts/build_dashboard_substrate.py` →
`gpusitter.app.dashboard_substrate.build_substrate`.

## Contract

| file | shape |
|---|---|
| `meta.json` | `{window, cascadeTs, totalGpus, faulted, nodesAffected, source}` — `faulted` is the hero burst **onset** count (edge-detected), never the cumulative latched count |
| `fleet.json` | `{ts, nodes, faulted, cells:[{node,idx,temp,util,xid,status}]}` — real per-GPU snapshot at the hero center |
| `incidents.json` | `[{id, ts, gpu, xid, xidLabel, severity, nodeCofaults, correlatedCount, correlated, hero, state}]` |
| `telemetry/INC-*.json` | `{gpu, centerTs, series:{temp,power,util}}` — real ±3 min windows |
| `manifest.json` | provenance: source, generation command, input paths/OIDs, event window, `telemetryKind: "real"`, builder git rev, onset/burst/hero summary |

This mirrors the legacy `dashboard/fixtures/*` shapes (a documented successor
contract) so the FastAPI layer (bead h7w) can serve it directly.

## Honesty — why this is not the Aug-29/882 story

Incidents are **edge-detected, empty-aware Xid onsets** (a non-fault → fault
transition, via `rca.job_join.stream_xid_onset_records`), the same detector the
runtime RCA / detection paths use. They are **not** latched cumulative Xid-gauge
snapshots. At any late timestamp almost every cumulatively-faulted GPU reads
nonzero, which is exactly how the debunked "Aug-29 13:57 / 882-GPU simultaneous
cascade" arose (a latched-state + window-edge artifact). Under edge detection
Aug-29 has **1** true onset; the real recurring hero burst is **Aug-17 06:00**
(~116 onsets / 74 nodes, Xid 43 dominant — see `docs/incident-aug17-0600.md`).

The fleet snapshot reinforces this: a cell is rendered `fault` **only** if it is
an edge-detected member of the hero burst cohort; every other GPU's `status`
derives from utilization, never from its latched Xid value. So the heatmap cannot
reconstruct the cumulative cascade either. Raw kalos data is never modified.

The `tests/test_dashboard_substrate.py` latched-state regression locks this in.

## Regenerating (droplet)

The raw trace is materialized only on the droplet. Off-droplet the canonical
resolver raises a clear "raw data not materialized" error by design.

```
PYTHONPATH=src python scripts/build_dashboard_substrate.py
```

Deterministic: identical inputs produce a byte-identical tree (sorted-key JSON).
`manifest.json` records the inputs and the builder git rev for every regeneration.
