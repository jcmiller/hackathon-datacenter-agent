import os
from datetime import datetime, timedelta

from ..detection import stream
from ..detection import dataset
from ..detection import classifier
from ..rca.job_join import correlated_jobs, load_incidents
from ..telemetry.timeparse import parse_time_value
from ..telemetry.window import window_stats
from .memory import search_incidents, append_incident

TRACE_CSV        = "data/acme-util/data/job_trace/trace_kalos.csv"
POWER_CSV        = "data/acme-util/data/utilization/kalos/POWER_USAGE.csv"
TEMP_CSV         = "data/acme-util/data/utilization/kalos/GPU_TEMP.csv"
SOP_PATH         = "data/sop.json"
MODEL_STATE_PATH = "data/model_state.json"
_TICKET = {"n": 0}
_pending_updates: list[dict] = []


def _window(fail_time, seconds):
    """Return (start, end) bounds around fail_time using the telemetry timeparse seam."""
    center = parse_time_value(fail_time)
    if isinstance(center, datetime):
        delta = timedelta(seconds=seconds)
        return center - delta, center + delta
    return center - seconds, center + seconds


def get_telemetry(fail_time, window=120):
    """GPU telemetry around an incident, keyed by the real DCGM field names a
    production fleet emits (dcgm-exporter). Values are window aggregates."""
    lo, hi = _window(fail_time, window)
    return {
        "DCGM_FI_DEV_POWER_USAGE": window_stats(POWER_CSV, lo, hi),
        "DCGM_FI_DEV_GPU_TEMP":    window_stats(TEMP_CSV,  lo, hi),
    }


def find_correlated_failures(fail_time, window=120):
    """Find other node failures near this incident in time."""
    inc = load_incidents(TRACE_CSV)
    corr = correlated_jobs(inc, fail_time, window)
    types = {c["type"] for c in corr}
    return {"count": len(corr), "jobs": [c["job_id"] for c in corr],
            "shared_type": next(iter(types)) if len(types) == 1 else None}


def check_degradation_trend(fail_time, lookback_hours: int = 4):
    """Examine telemetry in the hours BEFORE the fault to detect gradual degradation.
    A high power_spike_ratio (>1.5) or temp_rise_C (>10) indicates pre-failure stress."""
    center = parse_time_value(fail_time)
    if isinstance(center, datetime):
        start = center - timedelta(hours=lookback_hours)
        end = center
    else:
        start = center - lookback_hours * 3600
        end = center
    power = window_stats(POWER_CSV, start, end)
    temp  = window_stats(TEMP_CSV,  start, end)
    spike = round(power["max"] / power["mean"], 2) if power.get("mean", 0) > 0 else 0.0
    rise  = round(temp["max"]  - temp["min"],   1) if temp.get("samples", 0) > 0 else 0.0
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
    return {"paged": True, "ticket": f"TKT-{_TICKET['n']:04d}",
            "node": node_info, "reason": reason}


def record_resolution(incident_type, summary, disposition, resolution,
                      incident_id: str = "",
                      power_spike_ratio: float = 0.0,
                      temp_rise_C: float = 0.0,
                      correlated_count: int = 0):
    """Append this incident + resolution to the SOP memory.
    Pass telemetry metrics so the system can learn to predict dispositions from signals."""
    metrics = {
        "power_spike_ratio": power_spike_ratio,
        "temp_rise_C": temp_rise_C,
        "correlated_count": correlated_count,
    }
    entry = {"type": incident_type, "summary": summary,
             "disposition": disposition, "resolution": resolution,
             "incident_id": incident_id, "metrics": metrics}
    append_incident(entry, SOP_PATH)
    _pending_updates.append({"path": SOP_PATH, "entry": entry})
    return {"recorded": True}


def train_and_validate(model_type: str = "logreg"):
    """Fit a disposition classifier on all SOP entries that have telemetry metrics.
    Features: power_spike_ratio, temp_rise_C, correlated_count → label: page/escalate vs restart.
    Promotes if val ROC-AUC beats the incumbent. Returns the current model card."""
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
    should_promote = (classifier.INCUMBENT is None or
                      (val_auc is not None and val_auc > incumbent_auc))
    promoted = False
    if should_promote:
        promoted = classifier.maybe_promote(est, model_type, features, val_auc or 0.0, len(X))
        if promoted:
            classifier.save_state(MODEL_STATE_PATH)
    return {
        "trained": True, "model_type": model_type, "features": features,
        "val_auc": val_auc, "incumbent_auc": round(incumbent_auc, 3) if incumbent_auc else None,
        "promoted": promoted, "n_samples": len(X),
        "version": classifier.INCUMBENT.version if classifier.INCUMBENT else 0,
        "note": "no holdout yet — need more varied incidents for AUC" if val_auc is None else None,
    }


def get_sensory(job_id):
    """Return telemetry-aggregate fields for a job from accumulated history."""
    for r in stream.HISTORY:
        if str(r.get("job_id")) == str(job_id):
            return {k: r[k] for k in r if k.startswith(("power_", "temp_", "util_"))}
    return {}
