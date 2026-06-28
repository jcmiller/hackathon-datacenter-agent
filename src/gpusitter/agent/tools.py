import os
from datetime import datetime, timedelta

from ..detection import classifier, dataset, stream
from ..rca.job_join import (
    correlated_jobs,
    load_incidents,
    stream_xid_onset_records,
)
from ..telemetry.sources import (
    DCGM_FIELD_TO_METRIC,
    metric_csv_relpath,
    resolve_metric_csv,
)
from ..telemetry.timeparse import parse_time_value, window_bounds
from ..telemetry.window import window_stats
from .memory import append_incident, search_incidents

TRACE_CSV = "data/acme-util/data/job_trace/trace_kalos.csv"
SOP_PATH = "data/sop.json"
MODEL_STATE_PATH = "data/model_state.json"
_TICKET = {"n": 0}
_pending_updates: list[dict] = []


def _bound(value):
    """JSON-friendly representation of a window bound (ISO for datetimes)."""
    return value.isoformat() if isinstance(value, datetime) else value


def _xid_code(value):
    """Render an observed Xid onset code as an int when it is integral (e.g. 43)."""
    return int(value) if isinstance(value, float) and value.is_integer() else value


def _metric_stats(metric, lo, hi):
    """Resolve a canonical kalos metric CSV and aggregate over ``[lo, hi]``.

    Every result carries provenance (source path, metric, window, sample count).
    When the raw LFS object is not materialized on this host the metric is
    reported as ``available: False`` with a reason — an explicit, grounded fact
    for the agent, not a stack trace. Misconfigured/non-time-series sources still
    fail loud (``ValueError`` from the source validator propagates).
    """
    try:
        path = resolve_metric_csv(metric)
    except FileNotFoundError as exc:
        return {
            "available": False,
            "reason": "raw data not materialized",
            "metric": metric,
            "source": metric_csv_relpath(metric),
            "window": [_bound(lo), _bound(hi)],
            "detail": str(exc),
        }
    stats = window_stats(path, lo, hi)
    return {
        "available": True,
        "metric": metric,
        "source": path,
        "window": [_bound(lo), _bound(hi)],
        **stats,
    }


def get_telemetry(fail_time, window=120):
    """GPU telemetry around an incident, keyed by the real DCGM field names a
    production fleet emits (dcgm-exporter), read from the canonical kalos metric
    CSVs (LFS-cache resolved). Each value is a window aggregate plus provenance,
    or ``available: False`` when the raw data is not materialized here."""
    lo, hi = window_bounds(fail_time, window)
    return {field: _metric_stats(metric, lo, hi) for field, metric in DCGM_FIELD_TO_METRIC.items()}


def find_correlated_failures(fail_time, window=120, source="jobs"):
    """Find other failures near this incident in time.

    ``source="jobs"`` (default): other FAILED/NODE_FAIL job records within +/-
    ``window`` seconds (job<->job temporal correlation). ``source="xid"``:
    edge-detected per-GPU Xid onsets within +/- ``window`` of the fail_time — the
    preferred early-detection incident source — returning the GPU cohort and the
    observed Xid code(s). Both carry provenance (artifact + count)."""
    if source == "jobs":
        inc = load_incidents(TRACE_CSV)
        corr = correlated_jobs(inc, fail_time, window)
        types = {c["type"] for c in corr}
        return {
            "source": "jobs",
            "artifact": TRACE_CSV,
            "count": len(corr),
            "jobs": [c["job_id"] for c in corr],
            "shared_type": next(iter(types)) if len(types) == 1 else None,
        }
    if source == "xid":
        center = parse_time_value(fail_time)
        if not isinstance(center, datetime):
            raise ValueError(
                "source='xid' requires an ISO fail_time — Xid telemetry is "
                "wall-clock timestamped (e.g. '2023-08-15 15:30:00+08:00')"
            )
        try:
            xid_path = resolve_metric_csv("XID_ERRORS")
        except FileNotFoundError as exc:
            return {
                "source": "xid",
                "available": False,
                "reason": "raw data not materialized",
                "artifact": metric_csv_relpath("XID_ERRORS"),
                "detail": str(exc),
            }
        span = timedelta(seconds=window)
        lo, hi = center - span, center + span
        hits = [
            (t, gpu, code) for (t, gpu, code) in stream_xid_onset_records(xid_path) if lo <= t <= hi
        ]
        gpus = sorted({gpu for _, gpu, _ in hits})
        codes = sorted({_xid_code(code) for _, _, code in hits})
        return {
            "source": "xid",
            "available": True,
            "artifact": xid_path,
            "window": [_bound(lo), _bound(hi)],
            "count": len(hits),
            "gpus": gpus,
            "observed_xid": codes[0] if len(codes) == 1 else (codes or None),
        }
    raise ValueError(f"unknown source: {source!r} (expected 'jobs' or 'xid')")


def check_degradation_trend(fail_time, lookback_hours: int = 4):
    """Examine telemetry in the hours BEFORE the fault to detect gradual degradation.
    A high power_spike_ratio (>1.5) or temp_rise_C (>10) indicates pre-failure stress.
    Reads the canonical kalos POWER_USAGE/GPU_TEMP sources; degrades gracefully when
    the raw data is not materialized (signals computed only from available metrics)."""
    center = parse_time_value(fail_time)
    if isinstance(center, datetime):
        start = center - timedelta(hours=lookback_hours)
        end = center
    else:
        start = center - lookback_hours * 3600
        end = center
    power = _metric_stats("POWER_USAGE", start, end)
    temp = _metric_stats("GPU_TEMP", start, end)
    spike = (
        round(power["max"] / power["mean"], 2)
        if power.get("available") and power.get("mean", 0) > 0
        else 0.0
    )
    rise = (
        round(temp["max"] - temp["min"], 1)
        if temp.get("available") and temp.get("samples", 0) > 0
        else 0.0
    )
    return {
        "lookback_hours": lookback_hours,
        "power": power,
        "power_spike_ratio": spike,
        "temp": temp,
        "temp_rise_C": rise,
        "gradual_degradation_signal": spike > 1.5 or rise > 10,
    }


def search_past_incidents(incident_description: str):
    """Retrieve semantically similar past incidents using embedding search.
    Pass a rich description: Xid error code, telemetry pattern, correlated count, node behavior."""
    hits = search_incidents(incident_description, SOP_PATH)
    return {"count": len(hits), "matches": hits}


def page_technician(node_info, reason):
    """Simulate paging a datacenter technician."""
    _TICKET["n"] += 1
    return {"paged": True, "ticket": f"TKT-{_TICKET['n']:04d}", "node": node_info, "reason": reason}


def record_resolution(
    incident_type,
    summary,
    disposition,
    resolution,
    incident_id: str = "",
    power_spike_ratio: float = 0.0,
    temp_rise_C: float = 0.0,
    correlated_count: int = 0,
):
    """Append this incident + resolution to the SOP memory.
    Pass telemetry metrics so the system can learn to predict dispositions from signals."""
    metrics = {
        "power_spike_ratio": power_spike_ratio,
        "temp_rise_C": temp_rise_C,
        "correlated_count": correlated_count,
    }
    entry = {
        "type": incident_type,
        "summary": summary,
        "disposition": disposition,
        "resolution": resolution,
        "incident_id": incident_id,
        "metrics": metrics,
    }
    append_incident(entry, SOP_PATH)
    _pending_updates.append({"path": SOP_PATH, "entry": entry})
    return {"recorded": True}


def _train_over_history(model_type, features):
    """Fit a failure predictor over the streamed job HISTORY on ``features``.

    Restored original contract (pre-RSI-merge): a feature name absent from the
    streamed records yields an ``unknown feature`` result rather than a stack
    trace; a single-class train/val split is guarded; promotion is gated on the
    incumbent val ROC-AUC."""
    try:
        X, y = dataset.build_xy(stream.HISTORY, features)
    except KeyError:
        return {"trained": False, "reason": "unknown feature"}
    Xtr, ytr, Xval, yval = dataset.time_split(X, y, val_frac=0.3)
    if len(set(ytr)) < 2 or len(set(yval)) < 2:
        return {"trained": False, "reason": "insufficient class balance"}
    est = classifier.fit_candidate(model_type, features, Xtr, ytr)
    val_auc = classifier.auc(est, Xval, yval)
    incumbent_auc = classifier.INCUMBENT.auc if classifier.INCUMBENT else None
    promoted = classifier.maybe_promote(est, model_type, features, val_auc)
    return {
        "trained": True,
        "model_type": model_type,
        "features": features,
        "val_auc": val_auc,
        "incumbent_auc": incumbent_auc,
        "promoted": promoted,
        "version": classifier.INCUMBENT.version,
    }


def train_and_validate(model_type: str = "logreg", features=None):
    """Fit a failure/disposition classifier and promote it if it beats the incumbent.

    Two data sources, selected by ``features``:

    - ``features=None`` (default — the live triage agent's path): fit a *disposition*
      classifier on accumulated SOP entries that carry telemetry metrics
      (power_spike_ratio, temp_rise_C, correlated_count → page/escalate vs restart).
    - ``features=[...]`` (early-detection / online-learning path): fit a *failure*
      predictor over the streamed job HISTORY using the named feature columns
      (label = job state in {NODE_FAIL, FAILED}). An absent column returns
      ``{"trained": False, "reason": "unknown feature"}``.

    Both promote when val ROC-AUC beats the incumbent and return the model card."""
    if features is not None:
        return _train_over_history(model_type, features)
    import json

    if not os.path.exists(SOP_PATH):
        return {"trained": False, "reason": "no SOP entries yet"}
    with open(SOP_PATH) as f:
        entries = json.load(f)
    X, y, features = dataset.build_xy_from_sop(entries)
    if len(X) < 1:
        return {"trained": False, "reason": "no SOP entries with metrics yet"}
    est = classifier.fit_candidate(model_type, features, X, y)

    val_auc = None
    Xtr, ytr, Xval, yval = dataset.time_split_lists(X, y, val_frac=0.3)
    if len(Xval) >= 2 and len(set(ytr)) >= 2 and len(set(yval)) >= 2:
        val_auc = round(classifier.auc_from_lists(est, Xval, yval), 3)

    incumbent_auc = classifier.INCUMBENT.auc if classifier.INCUMBENT else None
    should_promote = classifier.INCUMBENT is None or (
        val_auc is not None and val_auc > incumbent_auc
    )
    promoted = False
    if should_promote:
        promoted = classifier.maybe_promote(est, model_type, features, val_auc or 0.0, len(X))
        if promoted:
            classifier.save_state(MODEL_STATE_PATH)
    return {
        "trained": True,
        "model_type": model_type,
        "features": features,
        "val_auc": val_auc,
        "incumbent_auc": round(incumbent_auc, 3) if incumbent_auc else None,
        "promoted": promoted,
        "n_samples": len(X),
        "version": classifier.INCUMBENT.version if classifier.INCUMBENT else 0,
        "note": "no holdout yet — need more varied incidents for AUC" if val_auc is None else None,
    }


def get_sensory(job_id):
    """Return telemetry-aggregate fields for a job from accumulated history."""
    for r in stream.HISTORY:
        if str(r.get("job_id")) == str(job_id):
            return {k: r[k] for k in r if k.startswith(("power_", "temp_", "util_"))}
    return {}
