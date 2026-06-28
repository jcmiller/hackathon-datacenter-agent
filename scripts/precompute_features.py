"""Offline, run-once on the box where the raw telemetry lives (the DigitalOcean droplet):
AcmeTrace job trace + per-metric DCGM telemetry CSVs -> small data/jobs.csv.
This is the ONLY step that reads the large (~80 GB) telemetry; it is never imported at runtime.

Data reality (see docs/DATA.md):
- Timestamps are ISO UTC strings, parsed with pd.to_datetime(..., utc=True). Telemetry is
  recorded at +08:00; utc=True normalizes both sides so the window join is timezone-correct.
- Telemetry lives in acme-util/data/utilization/kalos/<METRIC>.csv (POWER_USAGE / GPU_TEMP /
  GPU_UTIL): a `Time` column + ~2,344 per-GPU columns. util_pkl/*.pkl and ipmi/*.csv are unusable.
- Label is downstream: dataset.build_xy treats state in {NODE_FAIL, FAILED} as positive; Kalos
  only has FAILED. We pass `state` and `fail_time` through unchanged.
- Only ~113 of 13,836 FAILED jobs overlap the ~2-week telemetry snapshot; jobs outside it get
  zero-filled aggregates. That is expected, not a bug.
- Aggregates are taken over ALL GPU columns in the window (not just the job's GPUs). Per-job GPU
  attribution needs node/GPU id normalization (ids are inconsistent across metrics) and is deferred.
"""

import pandas as pd

_META = [
    "job_id",
    "type",
    "node_num",
    "gpu_num",
    "cpu_num",
    "duration",
    "queue",
    "mem_per_pod_GB",
    "state",
    "start_time",
    "end_time",
    "fail_time",
]
_AGG = ["mean", "max", "std"]
_ZERO = dict.fromkeys(_AGG, 0.0)


def _window_aggs(signal_df, start, end):
    """Aggregate every per-GPU sample whose Time is within [start, end]."""
    if signal_df is None:
        return dict(_ZERO)
    win = signal_df[(signal_df["Time"] >= start) & (signal_df["Time"] <= end)]
    vals = win.drop(columns=["Time"]).to_numpy(dtype="float64").ravel()
    vals = vals[~pd.isna(vals)]
    if vals.size == 0:
        return dict(_ZERO)
    return {"mean": float(vals.mean()), "max": float(vals.max()), "std": float(vals.std())}


def _load_signal(path):
    if path is None:
        return None
    df = pd.read_csv(path)
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    return df


def precompute_jobs(trace_csv, power_csv, temp_csv, util_csv, out_csv):
    trace = pd.read_csv(trace_csv)
    trace["start_time"] = pd.to_datetime(trace["start_time"], utc=True)
    trace["end_time"] = pd.to_datetime(trace["end_time"], utc=True)
    signals = {
        "power": _load_signal(power_csv),
        "temp": _load_signal(temp_csv),
        "util": _load_signal(util_csv),
    }
    rows = []
    for _, j in trace.iterrows():
        row = {k: j.get(k) for k in _META}
        for name, sig in signals.items():
            aggs = _window_aggs(sig, j["start_time"], j["end_time"])
            for a in _AGG:
                row[f"{name}_{a}"] = aggs[a]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
