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
* onset detection: ``gpusitter.rca.job_join.stream_xid_onsets`` — the
  nav-approved empty-aware edge detector. Onsets are non-fault -> fault edges
  where the prior OBSERVED state was healthy (0.0) OR idle (empty cell); reading
  the raw CSV (not the empty-skipping long path) is essential because many kalos
  GPUs go idle -> fault without ever recording a 0.0. Latched faults
  (nonzero -> nonzero) and a GPU already nonzero on first observation
  (left-censored) are excluded.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ..rca.job_join import stream_xid_onsets
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
    """An observed non-fault -> nonzero Xid transition on one GPU."""

    gpu: GpuId
    t: datetime


def xid_onsets(path: str) -> List[OnsetEvent]:
    """Empty-aware Xid onsets from a wide XID_ERRORS CSV.

    Delegates to the nav-approved ``stream_xid_onsets`` (rca.job_join) so the
    onset semantics are defined in exactly one place: an onset is a transition
    into a fault whose prior observed state was healthy (0.0) or idle (empty
    cell). Crucially this catches idle(empty) -> fault transitions that the
    empty-skipping long-record reader would miss.
    """
    return [OnsetEvent(gpu=_parse_canonical(canonical), t=t)
            for t, canonical in stream_xid_onsets(path)]


# --- Windowed features (pre-reference; missingness explicit) -----------------


def _window_from_arrays(
    times: Sequence[datetime],
    values: Sequence[float],
    t_ref: datetime,
    lookback_s: float,
    sample_period_s: float = 15.0,
) -> Dict[str, float]:
    """Aggregate ``[t_ref - lookback, t_ref]`` from time-sorted parallel arrays.

    ``times`` must be ascending; the window is located with two binary searches
    (O(log n) + window size), not a full per-call scan. Future leakage is
    impossible (upper bound is ``t_ref``). An empty window returns NaN stats with
    ``present = 0`` / ``coverage = 0`` — never zero-filled, since 0 is a
    meaningful telemetry value.
    """
    lo = t_ref - timedelta(seconds=lookback_s)
    i = bisect_left(times, lo)
    j = bisect_right(times, t_ref)
    expected = lookback_s / sample_period_s + 1.0 if sample_period_s > 0 else 0.0
    if j <= i:
        nan = float("nan")
        return {"count": 0, "coverage": 0.0, "present": 0, "mean": nan,
                "std": nan, "min": nan, "max": nan, "last": nan,
                "delta": nan, "slope": nan}
    vals = np.array(values[i:j], dtype="float64")
    secs = np.array([(times[k] - t_ref).total_seconds() for k in range(i, j)],
                    dtype="float64")
    slope = float(np.polyfit(secs, vals, 1)[0]) if len(vals) >= 2 else float("nan")
    return {
        "count": int(len(vals)),
        "coverage": float(len(vals) / expected) if expected > 0 else 0.0,
        "present": 1,
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "last": float(vals[-1]),
        "delta": float(vals[-1] - vals[0]),
        "slope": slope,
    }


def window_features(
    series: Sequence[Tuple[datetime, float]],
    t_ref: datetime,
    lookback_s: float,
    *,
    sample_period_s: float = 15.0,
) -> Dict[str, float]:
    """Convenience wrapper over :func:`_window_from_arrays` taking ``[(t, v)]``.

    Sorts defensively, then delegates to the binary-search core; used by tests
    and any caller without pre-split arrays. ``build_dataset`` calls the array
    core directly off a cache to keep the hot path O(log n) per candidate.
    """
    pairs = sorted(series, key=lambda kv: kv[0])
    times = [t for t, _ in pairs]
    values = [v for _, v in pairs]
    return _window_from_arrays(times, values, t_ref, lookback_s,
                               sample_period_s=sample_period_s)


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


def _series_arrays(store: TelemetryStore, metric: str, gpu: str,
                   cache: Dict[Tuple[str, str], Tuple[List[datetime], List[float]]]):
    key = (metric, gpu)
    if key not in cache:
        pairs = sorted(((_parse_iso(t), v) for t, v in store.series(metric, gpu)),
                       key=lambda kv: kv[0])
        cache[key] = ([t for t, _ in pairs], [v for _, v in pairs])
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
        times, values = _series_arrays(store, metric, gpu.canonical, cache)
        feats = _window_from_arrays(times, values, t_ref, lookback_s,
                                    sample_period_s=sample_period_s)
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
    id, a time-matched control at every positive ``t_ref``. **Leakage guard:** a
    negative candidate is *dropped* (not emitted) if an onset for that GPU falls
    inside its horizon — so the negative pool contains only true negatives, even
    when ``neg_offset_s < horizon_s``. Rows with no telemetry coverage in any
    feature metric are dropped. ``alias`` normalizes foreign GPU namespaces for
    the feature store (onsets use the IP-named XID file directly).
    """
    if XID_METRIC not in sources:
        raise ValueError(f"sources must include {XID_METRIC!r} to derive labels")
    feature_metrics = [m for m in sources if m != XID_METRIC]
    control_gpus = list(control_gpus or [])

    onsets = xid_onsets(sources[XID_METRIC])
    onsets_by_gpu = _onsets_by_gpu(onsets)
    gpu_by_canonical = {o.gpu.canonical: o.gpu for o in onsets}

    # Positives: keyed (canonical, t_ref, horizon) -> GpuId. A positive's label
    # is 1 by construction (the onset sits at t_ref + H).
    positives: Dict[Tuple[str, datetime, float], GpuId] = {}
    # Negative candidates, kept only if they are true negatives (leakage guard).
    negatives: Dict[Tuple[str, datetime, float], GpuId] = {}

    for h in horizons_s:
        positive_refs: List[datetime] = []
        for o in onsets:
            positives[(o.gpu.canonical, o.t - timedelta(seconds=h), h)] = o.gpu
            positive_refs.append(o.t - timedelta(seconds=h))
        for o in onsets:
            key = (o.gpu.canonical, o.t - timedelta(seconds=neg_offset_s), h)
            if key in positives:
                continue
            if _label(key[0], key[1], key[2], onsets_by_gpu) == 0:
                negatives[key] = o.gpu
        for cg in control_gpus:
            gpu = gpu_by_canonical.get(cg) or _parse_canonical(cg)
            for t_ref in positive_refs:
                key = (cg, t_ref, h)
                if key in positives or key in negatives:
                    continue
                if _label(key[0], key[1], key[2], onsets_by_gpu) == 0:
                    negatives[key] = gpu

    candidates: Dict[Tuple[str, datetime, float], GpuId] = {**positives, **negatives}
    sample_gpus = sorted({c[0] for c in candidates})
    feature_sources = {m: sources[m] for m in feature_metrics}
    store = TelemetryStore.load(feature_sources, gpus=sample_gpus, alias=alias)

    cache: Dict[Tuple[str, str], Tuple[List[datetime], List[float]]] = {}
    rows: List[dict] = []
    for (canonical, t_ref, h), gpu in sorted(candidates.items(),
                                             key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
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
