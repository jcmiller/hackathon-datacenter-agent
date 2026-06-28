import backend.stream as stream
import backend.dataset as dataset
import backend.classifier as classifier
from backend.loader import load_incidents, telemetry_window, correlated_failures
from backend.memory import search_incidents, append_incident

TRACE_CSV = "data/acme-util/data/job_trace/trace_kalos.csv"
# Telemetry files mapped to the real DCGM field they represent.
POWER_CSV = "data/acme-util/data/utilization/kalos/POWER_USAGE.csv"  # DCGM_FI_DEV_POWER_USAGE (W)
TEMP_CSV  = "data/acme-util/data/utilization/kalos/GPU_TEMP.csv"     # DCGM_FI_DEV_GPU_TEMP (C)
SOP_PATH  = "data/sop.json"
_TICKET = {"n": 0}
_pending_updates: list[dict] = []

def get_telemetry(fail_time, window=120):
    """GPU telemetry around an incident, keyed by the real DCGM field names a
    production fleet emits (dcgm-exporter). Values are window aggregates."""
    import pandas as pd
    ts = pd.to_datetime(fail_time, utc=True)
    start = (ts - pd.Timedelta(seconds=window)).isoformat()
    end   = (ts + pd.Timedelta(seconds=window)).isoformat()
    return {
        "DCGM_FI_DEV_POWER_USAGE": telemetry_window(POWER_CSV, start, end),
        "DCGM_FI_DEV_GPU_TEMP":    telemetry_window(TEMP_CSV,  start, end),
    }

def find_correlated_failures(fail_time, window=120):
    """Find other node failures near this incident in time."""
    import pandas as pd
    ts = pd.to_datetime(fail_time, utc=True)
    inc = load_incidents(TRACE_CSV)
    corr = correlated_failures(inc, ts, window)
    types = {c["type"] for c in corr}
    return {"count": len(corr), "jobs": [c["job_id"] for c in corr],
            "shared_type": next(iter(types)) if len(types) == 1 else None}

def check_degradation_trend(fail_time, lookback_hours: int = 4):
    """Examine telemetry in the hours BEFORE the fault to detect gradual degradation.
    A high power_spike_ratio (>1.5) or temp_rise_C (>10) indicates pre-failure stress."""
    import pandas as pd
    ts = pd.to_datetime(fail_time, utc=True)
    start = (ts - pd.Timedelta(hours=lookback_hours)).isoformat()
    end = ts.isoformat()
    power = telemetry_window(POWER_CSV, start, end)
    temp  = telemetry_window(TEMP_CSV,  start, end)
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

def record_resolution(incident_type, summary, disposition, resolution):
    """Append this incident + resolution to the SOP memory."""
    entry = {"type": incident_type, "summary": summary,
             "disposition": disposition, "resolution": resolution}
    append_incident(entry, SOP_PATH)
    _pending_updates.append({"path": SOP_PATH, "entry": entry})
    return {"recorded": True}


def get_sensory(job_id):
    """Return telemetry-aggregate fields for a job from accumulated history."""
    for r in stream.HISTORY:
        if str(r.get("job_id")) == str(job_id):
            return {k: r[k] for k in r if k.startswith(("power_", "temp_", "util_"))}
    return {}

def train_and_validate(model_type, features):
    """Fit a candidate on jobs-so-far, score val ROC-AUC, promote if it beats the incumbent."""
    X, y = dataset.build_xy(stream.HISTORY, features)
    Xtr, ytr, Xval, yval = dataset.time_split(X, y, val_frac=0.3)
    if len(set(ytr)) < 2 or len(set(yval)) < 2:
        return {"trained": False, "reason": "insufficient class balance"}
    est = classifier.fit_candidate(model_type, features, Xtr, ytr)
    val_auc = classifier.auc(est, Xval, yval)
    incumbent_auc = classifier.INCUMBENT.auc if classifier.INCUMBENT else None
    promoted = classifier.maybe_promote(est, model_type, features, val_auc)
    return {"trained": True, "model_type": model_type, "features": features,
            "val_auc": val_auc, "incumbent_auc": incumbent_auc,
            "promoted": promoted, "version": classifier.INCUMBENT.version}
