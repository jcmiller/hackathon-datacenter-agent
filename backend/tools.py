from backend.loader import load_incidents, telemetry_window, correlated_failures
from backend.memory import search_incidents, append_incident

TRACE_CSV = "data/trace_kalos.csv"
# Telemetry files mapped to the real DCGM field they represent.
POWER_CSV = "data/util/GPU_AB_Power.csv"   # DCGM_FI_DEV_POWER_USAGE (W)
TEMP_CSV = "data/util/gpu_temp_kalos.pkl"  # DCGM_FI_DEV_GPU_TEMP (C)
SOP_PATH = "data/sop.json"
_TICKET = {"n": 0}

def get_telemetry(fail_time, window=120):
    """GPU telemetry around an incident, keyed by the real DCGM field names a
    production fleet emits (dcgm-exporter). Values are window aggregates."""
    return {
        "DCGM_FI_DEV_POWER_USAGE": telemetry_window(POWER_CSV, fail_time - window, fail_time + window),
        "DCGM_FI_DEV_GPU_TEMP": telemetry_window(TEMP_CSV, fail_time - window, fail_time + window),
    }

def find_correlated_failures(fail_time, window=120):
    """Find other node failures near this incident in time."""
    inc = load_incidents(TRACE_CSV)
    corr = correlated_failures(inc, fail_time, window)
    types = {c["type"] for c in corr}
    return {"count": len(corr), "jobs": [c["job_id"] for c in corr],
            "shared_type": next(iter(types)) if len(types) == 1 else None}

def search_past_incidents(incident_type):
    """Retrieve resolved past incidents of the same type."""
    hits = search_incidents(incident_type, SOP_PATH)
    return {"count": len(hits), "matches": hits}

def page_technician(node_info, reason):
    """Simulate paging a datacenter technician."""
    _TICKET["n"] += 1
    return {"paged": True, "ticket": f"TKT-{_TICKET['n']:04d}",
            "node": node_info, "reason": reason}

def record_resolution(incident_type, summary, disposition, resolution):
    """Append this incident + resolution to the SOP memory."""
    append_incident({"type": incident_type, "summary": summary,
                     "disposition": disposition, "resolution": resolution}, SOP_PATH)
    return {"recorded": True}
