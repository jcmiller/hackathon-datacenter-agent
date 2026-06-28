"""Tests for backend/sim.py — run offline, no API key needed."""
import csv, pathlib, textwrap
import pytest
import backend.sim as sim
from fastapi.testclient import TestClient

client = TestClient(sim.app)


def test_record_resolution_emits_pending_update(tmp_path, monkeypatch):
    import backend.tools as tools_mod
    sop = tmp_path / "sop.json"
    monkeypatch.setattr(tools_mod, "SOP_PATH", str(sop))
    tools_mod._pending_updates.clear()

    tools_mod.record_resolution("GPU_HW_FAULT", "test summary", "PAGE_TECHNICIAN", "replaced card")

    assert len(tools_mod._pending_updates) == 1
    upd = tools_mod._pending_updates[0]
    assert upd["path"] == str(sop)
    assert upd["entry"]["type"] == "GPU_HW_FAULT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "job_id","user","node_num","gpu_num","cpu_num","type","state",
    "submit_time","start_time","end_time","duration","queue",
    "gpu_time","fail_time","stop_time",
]

def _write_sample_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "sample.csv"
    rows = [
        # NODE_FAIL row — should appear in SSE stream
        dict(job_id="JOB001", user="alice", node_num=4, gpu_num=8, cpu_num=32,
             type="GPU_TRAIN", state="NODE_FAIL", submit_time=1000, start_time=1010,
             end_time=1100, duration=90, queue="gpu", gpu_time=720,
             fail_time="2023-05-17 11:17:30+00:00", stop_time=1100),
        # COMPLETED row — filtered out by load_incidents
        dict(job_id="JOB002", user="bob", node_num=2, gpu_num=4, cpu_num=16,
             type="GPU_TRAIN", state="COMPLETED", submit_time=2000, start_time=2010,
             end_time=2200, duration=190, queue="gpu", gpu_time=760,
             fail_time="", stop_time=2200),
    ]
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_index_returns_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "root" in r.text.lower()


def test_incidents_sse_streams_fail_row(tmp_path, monkeypatch):
    csv_path = _write_sample_csv(tmp_path)
    monkeypatch.setattr(sim, "TRACE_CSV", str(csv_path))
    monkeypatch.setattr(sim, "STEP_SECONDS", 0)
    # clear any cached incidents so the monkeypatched path is used
    sim._incidents_cache.clear()

    with client.stream("GET", "/api/incidents") as resp:
        body = ""
        for chunk in resp.iter_text():
            body += chunk
            if "JOB001" in body:
                break

    assert "data:" in body
    assert "JOB001" in body
    assert "JOB002" not in body   # COMPLETED row must be filtered


def test_triage_endpoint_wiring(monkeypatch):
    monkeypatch.setattr(sim, "triage", lambda inc: {"disposition": "restart-and-watch"})
    r = client.post("/api/triage", json={"job_id": "JOB001", "state": "NODE_FAIL"})
    assert r.status_code == 200
    assert r.json()["disposition"] == "restart-and-watch"
