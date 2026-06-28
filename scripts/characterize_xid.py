#!/usr/bin/env python3
"""Characterize Kalos Xid faults and their telemetry precursors.

Produces the *detectable signal + label scheme* that early-incident detection
(aieng26hack-5fq) and the GPUSitter env's fault dynamics (aieng26hack-w28)
consume:

  1. Xid event distribution  - per-code counts of true fault *events*.
  2. Frequency / MTBF         - events per GPU-hour, mean inter-event time.
  3. Precursors               - for each event, how GPU_TEMP / POWER_USAGE /
                                MEM_CLOCK deviate in the lead window before the
                                fault, vs that GPU's own baseline, at several
                                horizons.

Why "events" and not nonzero cells
----------------------------------
``XID_ERRORS.csv`` is a DCGM *gauge*: it holds the GPU's last Xid code and
re-emits it every 15s sample until the GPU is reset/cleared. A single fault
therefore paints tens of thousands of nonzero cells. An EVENT is a *rising
edge*: a transition from 0/clear to a nonzero code, or a change to a different
nonzero code. (On the real trace, code 43 shows 59M nonzero cells but only 830
events -- a ~71,000x difference.)

Scale discipline
----------------
The kalos wide CSVs are ~0.8-1.2 GB each (Time + ~2344 GPU columns). We never
densify them. The Xid pass streams long records (q2o ``iter_long_records``).
The precursor pass streams each metric file exactly once, retaining only a
bounded pre-fault window per event GPU -- loading every event GPU's full series
into a ``TelemetryStore`` would exhaust the 4 GB analysis box. (The store
remains the right tool for *bounded* incident replay, which is w28's use.)

Usage (on the droplet, with the gpusitter package importable):
    python3 scripts/characterize_xid.py \
        --data-dir data/acme-util/data/utilization/kalos \
        --out xid_characterization.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

# Make the ``gpusitter`` package importable when run from the repo root (the
# package lives under ``src/`` per the hatchling src-layout).
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from gpusitter.telemetry.ingest import iter_long_records  # noqa: E402

# Kalos DCGM sample period (verified: 15s grid).
SAMPLE_SECONDS = 15

# Metrics whose IP-named namespace aligns with XID_ERRORS (so a per-GPU join is
# valid). SM_ACTIVE is pod-named with no deterministic IP<->pod map -> excluded.
PRECURSOR_METRICS = ["GPU_TEMP", "POWER_USAGE", "MEM_CLOCK"]

# Human-readable Xid meanings (NVIDIA DCGM Xid reference) for the report.
XID_MEANING = {
    43: "GPU stopped processing (fell off bus / hang)",
    31: "GPU memory page fault (MMU / illegal address)",
    45: "Preemptive cleanup (robust channel; often app-induced)",
    94: "Contained ECC error",
    95: "Uncontained ECC error",
    48: "Double-bit ECC error",
    79: "GPU has fallen off the bus",
}

# Severity grouping consumed by w28/5fq label scheme.
XID_SEVERITY = {
    43: "fatal_hang",
    79: "fatal_hang",
    48: "fatal_ecc",
    95: "fatal_ecc",
    94: "contained_ecc",
    31: "memory_fault",
    45: "soft_cleanup",
}


from typing import NamedTuple  # noqa: E402


class XidEvent(NamedTuple):
    gpu: str  # canonical GPU id
    t: str    # ISO timestamp of the rising edge
    code: int


# --------------------------------------------------------------------------- #
# 1. Event extraction (rising-edge detection over the gauge)
# --------------------------------------------------------------------------- #
def extract_events(xid_csv: str) -> list[XidEvent]:
    """Stream XID_ERRORS and return true fault events (rising edges).

    A record's value is the gauge's current code. We track the last code per
    GPU; an event fires when the code becomes a *different nonzero* value than
    the GPU's previous code.

    Left-censoring: a GPU whose *first observed* (first non-empty) sample is
    already nonzero is **not** counted as an event. That code was raised before
    the observation window opened, so its true onset is unknown — emitting it
    would manufacture a rising edge at the window boundary. The first sample only
    establishes the GPU's baseline code; events fire on transitions seen *after*
    it. (A GPU that first appears clear, then later faults, is a genuine onset.)
    """
    last: dict[str, int] = {}
    events: list[XidEvent] = []
    for rec in iter_long_records(xid_csv, "XID_ERRORS"):
        gpu = rec.gpu.canonical
        code = int(rec.value)
        if gpu not in last:
            # First observation: seed baseline state, never emit (left-censored
            # if already nonzero; nothing to transition from if clear).
            last[gpu] = code
            continue
        prev = last[gpu]
        if code != 0 and code != prev:
            events.append(XidEvent(gpu=gpu, t=rec.t, code=code))
        last[gpu] = code
    return events


# --------------------------------------------------------------------------- #
# 2. Frequency / MTBF
# --------------------------------------------------------------------------- #
def observed_span_seconds(xid_csv: str) -> tuple[str, str, float]:
    """First/last timestamp and elapsed seconds, from the XID time column."""
    first = last = None
    import csv

    with open(xid_csv, newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            if first is None:
                first = row[0]
            last = row[0]
    if first is None:
        return ("", "", 0.0)
    return (first, last, _iso_delta_seconds(first, last))


def _iso_delta_seconds(t0: str, t1: str) -> float:
    from datetime import datetime

    fmt = "%Y-%m-%d %H:%M:%S%z"
    return (datetime.strptime(t1, fmt) - datetime.strptime(t0, fmt)).total_seconds()


def frequency_stats(
    events: list[XidEvent], n_gpus_total: int, span_seconds: float
) -> dict:
    """Distribution + MTBF. MTBF is reported as GPU-hours per event over the
    observed GPU-time (n_gpus_total * span)."""
    by_code: dict[int, int] = defaultdict(int)
    for e in events:
        by_code[e.code] += 1
    gpu_hours = (n_gpus_total * span_seconds) / 3600.0 if span_seconds else 0.0
    total = len(events)
    return {
        "total_events": total,
        "distinct_event_gpus": len({e.gpu for e in events}),
        "by_code": {
            int(code): {
                "events": cnt,
                "meaning": XID_MEANING.get(code, "unknown"),
                "severity": XID_SEVERITY.get(code, "unknown"),
            }
            for code, cnt in sorted(by_code.items(), key=lambda kv: -kv[1])
        },
        "observed_gpu_hours": round(gpu_hours, 1),
        "mtbf_gpu_hours_per_event": round(gpu_hours / total, 1) if total else None,
        "events_per_gpu_hour": round(total / gpu_hours, 6) if gpu_hours else None,
    }


# --------------------------------------------------------------------------- #
# 2b. Temporal distribution + correlated-burst detection
# --------------------------------------------------------------------------- #
def _node_of(gpu_canonical: str) -> str:
    return gpu_canonical.split("#", 1)[0]


def temporal_stats(events: list[XidEvent], *, burst_bin_seconds: int = 300) -> dict:
    """Are faults sporadic or dominated by correlated cluster-wide bursts?

    Buckets events by calendar day and into fixed bins; reports the single
    largest bin (a candidate cluster-wide event) with its distinct-node and
    code breakdown. A burst hitting many nodes at once is an exogenous /
    infrastructure event, NOT independent per-GPU faults -- it must be separated
    before computing a meaningful per-GPU MTBF or precursor lead.
    """
    from datetime import datetime

    fmt = "%Y-%m-%d %H:%M:%S%z"
    by_day: dict[str, int] = defaultdict(int)
    bins: dict[int, list[XidEvent]] = defaultdict(list)
    if not events:
        return {"by_day": {}, "largest_burst": None}
    t0 = min(datetime.strptime(e.t, fmt) for e in events)
    for e in events:
        dt = datetime.strptime(e.t, fmt)
        by_day[e.t[:10]] += 1
        b = int((dt - t0).total_seconds() // burst_bin_seconds)
        bins[b].append(e)

    # Largest bin = candidate correlated burst.
    top_b = max(bins, key=lambda b: len(bins[b]))
    burst = bins[top_b]
    burst_codes: dict[int, int] = defaultdict(int)
    for e in burst:
        burst_codes[e.code] += 1
    burst_start = min(e.t for e in burst)
    burst_end = max(e.t for e in burst)

    n = len(events)
    n_burst = len(burst)
    return {
        "burst_bin_seconds": burst_bin_seconds,
        "by_day": dict(sorted(by_day.items())),
        "largest_burst": {
            "window": [burst_start, burst_end],
            "events": n_burst,
            "distinct_gpus": len({e.gpu for e in burst}),
            "distinct_nodes": len({_node_of(e.gpu) for e in burst}),
            "by_code": dict(sorted(burst_codes.items(), key=lambda kv: -kv[1])),
            "frac_of_all_events": round(n_burst / n, 3),
        },
        "sporadic_events": n - n_burst,
    }


# --------------------------------------------------------------------------- #
# 3. Precursors
# --------------------------------------------------------------------------- #
def collect_prefault_windows(
    metric_csv: str,
    metric: str,
    event_times: dict[str, tuple[str, str]],
    *,
    baseline_seconds: int,
) -> dict[str, list[tuple[str, float]]]:
    """Single streaming pass: for each *event*, retain samples in
    ``[t_event - baseline_seconds, t_event)``.

    ``event_times`` maps a unique event key -> ``(gpu, t_event)``. Keying by
    event (not by GPU) keeps repeated faults on the *same* GPU as distinct
    windows; a single streamed record can feed several of that GPU's windows when
    their baseline spans overlap. Returns ``{event_key: [(t, value), ...]}``
    sorted by time. Empty cells are skipped (idle/unobserved) -- never
    zero-filled.
    """
    from datetime import datetime, timedelta

    fmt = "%Y-%m-%d %H:%M:%S%z"
    # Precompute the [lo, hi) bound per event and index event keys by GPU so each
    # streamed record is matched against all of that GPU's pending windows.
    lo_bound: dict[str, str] = {}
    hi_bound: dict[str, str] = {}
    events_by_gpu: dict[str, list[str]] = defaultdict(list)
    for key, (gpu, te) in event_times.items():
        te_dt = datetime.strptime(te, fmt)
        lo_bound[key] = (te_dt - timedelta(seconds=baseline_seconds)).strftime(fmt)
        hi_bound[key] = te  # exclusive upper bound (strictly before fault)
        events_by_gpu[gpu].append(key)

    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    keep = {gpu for gpu, _ in event_times.values()}
    for rec in iter_long_records(metric_csv, metric, gpus=keep):
        for key in events_by_gpu.get(rec.gpu.canonical, ()):
            if lo_bound[key] <= rec.t < hi_bound[key]:
                out[key].append((rec.t, rec.value))
    for key in out:
        out[key].sort(key=lambda kv: kv[0])
    return out


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _slope_per_min(samples: list[tuple[str, float]]) -> float | None:
    """Least-squares slope of value vs minutes, over the given samples."""
    if len(samples) < 2:
        return None
    from datetime import datetime

    fmt = "%Y-%m-%d %H:%M:%S%z"
    t0 = datetime.strptime(samples[0][0], fmt)
    xs = [(datetime.strptime(t, fmt) - t0).total_seconds() / 60.0 for t, _ in samples]
    ys = [v for _, v in samples]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den


def precursor_stats(
    windows: dict[str, list[tuple[str, float]]],
    event_times: dict[str, tuple[str, str]],
    *,
    horizons_seconds: list[int],
    lead_for_baseline: int,
) -> dict:
    """For one metric, aggregate pre-fault deviation across events.

    ``event_times`` maps event key -> ``(gpu, t_event)``; ``windows`` is keyed by
    the same event key (see :func:`collect_prefault_windows`), so repeated faults
    on one GPU are aggregated as distinct events. ``windows`` is already bounded
    to each event's baseline span; here the baseline is everything earlier than
    the final ``lead_for_baseline`` seconds before the fault.

    For each event: baseline = median over [t_event-baseline .. t_event-lead).
    For each horizon H: lead window = [t_event-H .. t_event). Report mean of the
    lead window, delta vs baseline, slope/min, and whether the deviation exceeds
    2x the baseline std ("detectable").
    """
    from datetime import datetime, timedelta

    fmt = "%Y-%m-%d %H:%M:%S%z"
    per_horizon: dict[int, dict] = {}
    n_with_data = 0
    deltas_by_h: dict[int, list[float]] = {h: [] for h in horizons_seconds}
    slopes_by_h: dict[int, list[float]] = {h: [] for h in horizons_seconds}
    detect_by_h: dict[int, int] = dict.fromkeys(horizons_seconds, 0)
    count_by_h: dict[int, int] = dict.fromkeys(horizons_seconds, 0)

    for key, (_gpu, te) in event_times.items():
        samples = windows.get(key, [])
        if not samples:
            continue
        n_with_data += 1
        te_dt = datetime.strptime(te, fmt)
        base_hi = (te_dt - timedelta(seconds=lead_for_baseline)).strftime(fmt)
        base_vals = [v for t, v in samples if t < base_hi]
        base_med = _median(base_vals)
        base_std = _stdev(base_vals)
        for h in horizons_seconds:
            lead_lo = (te_dt - timedelta(seconds=h)).strftime(fmt)
            lead = [(t, v) for t, v in samples if t >= lead_lo]
            if not lead or base_med is None:
                continue
            count_by_h[h] += 1
            lead_mean = sum(v for _, v in lead) / len(lead)
            delta = lead_mean - base_med
            deltas_by_h[h].append(delta)
            slope = _slope_per_min(lead)
            if slope is not None:
                slopes_by_h[h].append(slope)
            # Detectable = lead-window mean outside baseline_median +/- 2*std.
            # A rock-steady baseline (std==0) that then deviates is detectable.
            if (base_std > 0 and abs(delta) >= 2 * base_std) or (
                base_std == 0 and abs(delta) > 1e-9
            ):
                detect_by_h[h] += 1

    for h in horizons_seconds:
        ds = deltas_by_h[h]
        ss = slopes_by_h[h]
        per_horizon[h] = {
            "horizon_seconds": h,
            "events_with_lead_samples": count_by_h[h],
            "mean_delta_vs_baseline": round(sum(ds) / len(ds), 3) if ds else None,
            "median_delta_vs_baseline": round(_median(ds), 3) if ds else None,
            "mean_slope_per_min": round(sum(ss) / len(ss), 4) if ss else None,
            "frac_detectable_2sigma": round(detect_by_h[h] / count_by_h[h], 3)
            if count_by_h[h]
            else None,
        }
    return {
        "events_with_any_baseline_data": n_with_data,
        "by_horizon": {str(h): per_horizon[h] for h in horizons_seconds},
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _present(data_dir: str, name: str) -> str | None:
    path = os.path.join(data_dir, f"{name}.csv")
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return path
    return None


def characterize(
    data_dir: str,
    *,
    horizons_seconds: list[int],
    baseline_seconds: int,
    lead_for_baseline: int,
) -> dict:
    xid_csv = _present(data_dir, "XID_ERRORS")
    if not xid_csv:
        raise SystemExit(f"XID_ERRORS.csv not found/usable under {data_dir}")

    events = extract_events(xid_csv)
    first, last, span = observed_span_seconds(xid_csv)

    # Total GPUs present in the XID header (the fleet under observation).
    import csv

    with open(xid_csv, newline="") as fh:
        header = next(csv.reader(fh))
    n_gpus_total = len(header) - 1

    freq = frequency_stats(events, n_gpus_total, span)
    temporal = temporal_stats(events)
    # Key by unique event, not by GPU: a (gpu, t) pair is unique (a GPU has at
    # most one rising edge per 15s sample), so repeated faults on the same GPU
    # stay distinct instead of collapsing to the last one.
    event_times = {f"{e.gpu}@{e.t}": (e.gpu, e.t) for e in events}

    precursors = {}
    for metric in PRECURSOR_METRICS:
        mpath = _present(data_dir, metric)
        if not mpath:
            precursors[metric] = {"error": "metric CSV absent"}
            continue
        windows = collect_prefault_windows(
            mpath, metric, event_times, baseline_seconds=baseline_seconds
        )
        precursors[metric] = precursor_stats(
            windows,
            event_times,
            horizons_seconds=horizons_seconds,
            lead_for_baseline=lead_for_baseline,
        )

    return {
        "data_dir": data_dir,
        "observed": {
            "first_ts": first,
            "last_ts": last,
            "span_seconds": span,
            "span_days": round(span / 86400.0, 2),
            "sample_seconds": SAMPLE_SECONDS,
            "n_gpus_total": n_gpus_total,
        },
        "frequency": freq,
        "temporal": temporal,
        "precursors": precursors,
        "events": [
            {"gpu": e.gpu, "t": e.t, "code": e.code} for e in events
        ],
        "label_scheme": {
            "event_definition": "rising edge of XID_ERRORS gauge "
            "(0/clear -> nonzero, or change to a different nonzero code)",
            "severity_groups": XID_SEVERITY,
            "horizons_seconds": horizons_seconds,
            "baseline_seconds": baseline_seconds,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="kalos util dir with the CSVs")
    ap.add_argument("--out", default=None, help="write full JSON here")
    ap.add_argument(
        "--horizons",
        default="60,300,600",
        help="comma-separated lead horizons in seconds (default 1/5/10 min)",
    )
    ap.add_argument(
        "--baseline-seconds",
        type=int,
        default=7200,
        help="how far before each fault to draw the per-GPU baseline (default 2h)",
    )
    ap.add_argument(
        "--lead-for-baseline",
        type=int,
        default=600,
        help="exclude this final lead window from the baseline (default 10min)",
    )
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    result = characterize(
        args.data_dir,
        horizons_seconds=horizons,
        baseline_seconds=args.baseline_seconds,
        lead_for_baseline=args.lead_for_baseline,
    )

    # Console summary.
    obs = result["observed"]
    freq = result["frequency"]
    print(
        f"observed: {obs['n_gpus_total']} GPUs, "
        f"{obs['first_ts']} .. {obs['last_ts']} ({obs['span_days']}d)"
    )
    print(
        f"events: {freq['total_events']} on {freq['distinct_event_gpus']} GPUs | "
        f"MTBF {freq['mtbf_gpu_hours_per_event']} GPU-h/event"
    )
    for code, info in freq["by_code"].items():
        print(f"  xid {code:>4}: {info['events']:>5}  {info['meaning']}")
    burst = result["temporal"]["largest_burst"]
    if burst:
        print(
            f"largest burst: {burst['events']} events "
            f"({burst['frac_of_all_events']:.0%} of all) across "
            f"{burst['distinct_nodes']} nodes @ {burst['window'][0]}..{burst['window'][1]}"
        )
        print(f"sporadic (non-burst) events: {result['temporal']['sporadic_events']}")
    print("precursors (frac detectable @2sigma by horizon):")
    for metric, pre in result["precursors"].items():
        if "by_horizon" not in pre:
            print(f"  {metric}: {pre.get('error')}")
            continue
        cells = " ".join(
            f"{h}s={hh['frac_detectable_2sigma']}"
            for h, hh in pre["by_horizon"].items()
        )
        print(f"  {metric}: {cells}  (n={pre['events_with_any_baseline_data']})")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
