"""Labeled early-detection dataset builder (beads lys / r7j).

Each output row is a *prediction point* ``(gpu, t_ref)``: would an alert fired at
``t_ref`` have caught a real fault? The label is 1 iff an Xid onset occurs for
that GPU within the horizon ``(t_ref, t_ref + H]``; features summarize the
lookback window ``[t_ref - lookback, t_ref]`` strictly *before* ``t_ref`` — so
the model only ever sees telemetry available at the moment it would alert.

This supersedes ``scripts/precompute_features.py``, whose job-level builder read
each wide telemetry CSV fully into pandas and re-filtered the whole frame once
per job (O(jobs * rows), and it densified ~80 GB of telemetry), aggregating over
*all* GPU columns with no per-GPU attribution.

It is deliberately thin: the heavy lifting reuses primitives that already exist
and are tested independently —

* cache-safe raw access: ``scripts.lfs_helper.resolve_data_path`` resolves a
  metric to a materialized file, an LFS-pointer-in-tree, or a deleted path
  recovered from ``git show`` — so the builder runs on the droplet where the
  ~80 GB working tree is absent and only ``.git/lfs`` cache objects exist.
* streaming wide->long reads: ``gpusitter.telemetry`` streams one row at a time
  and never materializes the full frame; ``TelemetryStore.load(..., gpus=...)``
  keeps only the sampled GPUs' cells, bounding memory to the sample set.

Label semantics follow the parent lys design: onsets are 0/idle -> nonzero Xid
transitions (XID_ERRORS is a latched gauge, so a sustained nonzero counts once);
a GPU already nonzero on its first observation is left-censored and excluded.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ..telemetry.ingest import iter_long_records
from ..telemetry.normalize import GpuId
from ..telemetry.store import TelemetryStore

XID_METRIC = "XID_ERRORS"

# Default IP-named feature metrics (SM_ACTIVE excluded until a pod/IP alias map
# exists; MEM_CLOCK optional). Mirrors the lys design's initial metric set.
DEFAULT_FEATURE_METRICS = ("GPU_TEMP", "POWER_USAGE", "GPU_UTIL", "MEMORY_TEMP")

_STATS = ("count", "coverage", "present", "mean", "std", "min", "max",
          "last", "delta", "slope")
_META_COLS = ("gpu", "node", "gpu_idx", "t_ref", "event_source",
              "horizon_s", "lookback_s", "label")


# --- Xid onset detection -----------------------------------------------------


class OnsetEvent(NamedTuple):
    """An observed 0/idle -> nonzero Xid transition on one GPU."""

    gpu: GpuId
    t: datetime


def xid_onsets(path: str, *, alias: Optional[Mapping[str, str]] = None) -> List[OnsetEvent]:
    """Stream XID_ERRORS and emit one event per 0/idle -> nonzero transition.

    Rows are time-ordered (verified on the droplet), so per-GPU records arrive in
    time order. Empty cells are skipped by the streaming reader (idle GPU), so a
    transition is observed as ``prev == 0 -> value != 0``. A GPU whose *first*
    observation is already nonzero is left-censored and never produces an onset.
    """
    prev: Dict[str, float] = {}
    onsets: List[OnsetEvent] = []
    for rec in iter_long_records(path, XID_METRIC, alias=alias):
        key = rec.gpu.canonical
        last = prev.get(key)
        if last is None:
            # First observation: a nonzero here is left-censored (we never saw
            # the transition), so it is not an onset.
            prev[key] = rec.value
            continue
        if last == 0.0 and rec.value != 0.0:
            onsets.append(OnsetEvent(gpu=rec.gpu, t=_parse_iso(rec.t)))
        prev[key] = rec.value
    return onsets


# --- Windowed features (pre-reference; missingness explicit) -----------------


def window_features(
    series: Sequence[Tuple[datetime, float]],
    t_ref: datetime,
    lookback_s: float,
    *,
    sample_period_s: float = 15.0,
) -> Dict[str, float]:
    """Aggregate one metric over ``[t_ref - lookback, t_ref]`` (inclusive).

    ``series`` is the full sorted ``[(t, value)]`` for one GPU/metric; only
    samples in the lookback window and at or before ``t_ref`` are used (future
    leakage is impossible). An empty window returns NaN stats with
    ``present = 0`` and ``coverage = 0`` — never zero-filled, because 0 is a
    meaningful telemetry value.
    """
    lo = t_ref - timedelta(seconds=lookback_s)
    win = [(t, v) for t, v in series if lo <= t <= t_ref]
    expected = lookback_s / sample_period_s + 1.0 if sample_period_s > 0 else 0.0
    if not win:
        nan = float("nan")
        return {"count": 0, "coverage": 0.0, "present": 0, "mean": nan,
                "std": nan, "min": nan, "max": nan, "last": nan,
                "delta": nan, "slope": nan}
    win.sort(key=lambda kv: kv[0])
    values = np.array([v for _, v in win], dtype="float64")
    secs = np.array([(t - t_ref).total_seconds() for t, _ in win], dtype="float64")
    slope = float(np.polyfit(secs, values, 1)[0]) if len(values) >= 2 else float("nan")
    return {
        "count": int(len(values)),
        "coverage": float(len(values) / expected) if expected > 0 else 0.0,
        "present": 1,
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "last": float(values[-1]),
        "delta": float(values[-1] - values[0]),
        "slope": slope,
    }


# --- Dataset assembly --------------------------------------------------------


def _onsets_by_gpu(onsets: Iterable[OnsetEvent]) -> Dict[str, List[datetime]]:
    by_gpu: Dict[str, List[datetime]] = {}
    for o in onsets:
        by_gpu.setdefault(o.gpu.canonical, []).append(o.t)
    for times in by_gpu.values():
        times.sort()
    return by_gpu


def _label(gpu_canonical: str, t_ref: datetime, horizon_s: float,
           onsets_by_gpu: Mapping[str, List[datetime]]) -> int:
    """1 iff an onset for this GPU falls in the prediction horizon ``(t_ref, t_ref+H]``."""
    hi = t_ref + timedelta(seconds=horizon_s)
    return int(any(t_ref < t <= hi for t in onsets_by_gpu.get(gpu_canonical, ())))


def _parse_series_cache(store: TelemetryStore, metric: str, gpu: str,
                        cache: Dict[Tuple[str, str], List[Tuple[datetime, float]]]):
    key = (metric, gpu)
    if key not in cache:
        parsed = [(_parse_iso(t), v) for t, v in store.series(metric, gpu)]
        parsed.sort(key=lambda kv: kv[0])
        cache[key] = parsed
    return cache[key]


def _row(gpu: GpuId, t_ref: datetime, horizon_s: float, lookback_s: float,
         label: int, feature_metrics: Sequence[str], store: TelemetryStore,
         cache, sample_period_s: float) -> Tuple[dict, bool]:
    """Build one output row; second element is True if any metric had coverage."""
    row = {
        "gpu": gpu.canonical, "node": gpu.node, "gpu_idx": gpu.index,
        "t_ref": t_ref.isoformat(), "event_source": XID_METRIC,
        "horizon_s": horizon_s, "lookback_s": lookback_s, "label": label,
    }
    any_present = False
    for metric in feature_metrics:
        series = _parse_series_cache(store, metric, gpu.canonical, cache)
        feats = window_features(series, t_ref, lookback_s, sample_period_s=sample_period_s)
        any_present = any_present or bool(feats["present"])
        for stat in _STATS:
            row[f"{metric}_{stat}"] = feats[stat]
    return row, any_present


def build_dataset(
    sources: Mapping[str, str],
    *,
    horizons_s: Sequence[float],
    lookback_s: float,
    neg_offset_s: float = 3600.0,
    control_gpus: Optional[Sequence[str]] = None,
    sample_period_s: float = 15.0,
    alias: Optional[Mapping[str, str]] = None,
) -> List[dict]:
    """Build labeled prediction-point rows from resolved metric CSV paths.

    ``sources`` maps metric name -> readable CSV path (already cache-safe; the
    CLI resolves them via ``lfs_helper.resolve_data_path``). It must contain
    ``XID_METRIC`` (drives labels) plus one or more feature metrics.

    For each horizon H: a positive at ``t_event - H`` per onset; a same-GPU
    pre-event control at ``t_event - neg_offset``; and, for each ``control_gpus``
    id, a time-matched control at every positive ``t_ref``. Every candidate's
    label is computed honestly from the onset set, so a negative that would fall
    inside a positive horizon is dropped (the leakage guard) rather than
    mislabeled. Rows with no telemetry coverage in any feature metric are dropped.
    """
    if XID_METRIC not in sources:
        raise ValueError(f"sources must include {XID_METRIC!r} to derive labels")
    feature_metrics = [m for m in sources if m != XID_METRIC]
    control_gpus = list(control_gpus or [])

    onsets = xid_onsets(sources[XID_METRIC], alias=alias)
    onsets_by_gpu = _onsets_by_gpu(onsets)
    gpu_by_canonical = {o.gpu.canonical: o.gpu for o in onsets}

    # Candidate reference points: (GpuId, t_ref, horizon_s). De-duplicated so the
    # same (gpu, t_ref, H) is never emitted twice.
    candidates: Dict[Tuple[str, datetime, float], GpuId] = {}
    for h in horizons_s:
        positive_refs: List[datetime] = []
        for o in onsets:
            t_pos = o.t - timedelta(seconds=h)
            candidates[(o.gpu.canonical, t_pos, h)] = o.gpu
            positive_refs.append(t_pos)
            t_neg = o.t - timedelta(seconds=neg_offset_s)
            candidates[(o.gpu.canonical, t_neg, h)] = o.gpu
        for cg in control_gpus:
            gpu = gpu_by_canonical.get(cg) or _parse_canonical(cg)
            for t_ref in positive_refs:
                candidates[(cg, t_ref, h)] = gpu

    sample_gpus = sorted({c[0] for c in candidates})
    feature_sources = {m: sources[m] for m in feature_metrics}
    store = TelemetryStore.load(feature_sources, gpus=sample_gpus, alias=alias)

    cache: Dict[Tuple[str, str], List[Tuple[datetime, float]]] = {}
    rows: List[dict] = []
    for (canonical, t_ref, h), gpu in sorted(candidates.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        label = _label(canonical, t_ref, h, onsets_by_gpu)
        row, any_present = _row(gpu, t_ref, h, lookback_s, label, feature_metrics,
                                store, cache, sample_period_s)
        if not any_present:
            continue  # no telemetry in the lookback window -> not a usable point
        rows.append(row)
    return rows


def write_dataset(rows: Sequence[dict], out_path: str) -> str:
    """Write rows to a compact table (parquet if pyarrow is present, else CSV).

    Returns the path actually written. A ``.parquet`` request with no pyarrow
    falls back to a sibling ``.csv`` so a missing optional dep never aborts a run.
    """
    import pandas as pd

    df = pd.DataFrame(list(rows))
    if not df.empty:
        meta = [c for c in _META_COLS if c in df.columns]
        feats = sorted(c for c in df.columns if c not in meta)
        df = df[meta + feats]

    if out_path.endswith(".parquet"):
        try:
            import pyarrow  # noqa: F401
            df.to_parquet(out_path, index=False)
            return out_path
        except Exception:
            out_path = out_path[: -len(".parquet")] + ".csv"
    df.to_csv(out_path, index=False)
    return out_path


# --- helpers -----------------------------------------------------------------


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _parse_canonical(canonical: str) -> GpuId:
    """Rebuild a GpuId from a canonical ``node#index`` string."""
    node, _, idx = canonical.rpartition("#")
    if not node or not idx:
        raise ValueError(f"not a canonical GPU id ('node#index'): {canonical!r}")
    return GpuId(node=node, index=int(idx))
