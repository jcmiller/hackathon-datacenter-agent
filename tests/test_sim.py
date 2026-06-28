"""Tests for backend/sim.py — run offline, no API key needed."""
import csv, pathlib
import backend.sim as sim
import backend.stream as stream
import backend.classifier as clf
from fastapi.testclient import TestClient

client = TestClient(sim.app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jobs_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    """3 rows: a warm-start failure, a completed job, then a streamed failure."""
    p = tmp_path / "jobs.csv"
    rows = [
        {"job_id": "WARM01", "type": "pretrain", "node_num": 4, "state": "NODE_FAIL"},
        {"job_id": "JOB002", "type": "pretrain", "node_num": 2, "state": "COMPLETED"},
        {"job_id": "JOB001", "type": "pretrain", "node_num": 4, "state": "NODE_FAIL"},
    ]
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_index_returns_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "incident" in r.text.lower()


def test_incidents_sse_streams_fail_row(tmp_path, monkeypatch):
    csv_path = _write_jobs_csv(tmp_path)
    monkeypatch.setattr(sim, "JOBS_CSV", str(csv_path))
    monkeypatch.setattr(sim, "WARM_START_INCIDENTS", 1)  # warm-start consumes WARM01 only
    monkeypatch.setattr(sim, "STEP_SECONDS", 0)
    sim._started["done"] = False
    stream.reset_history()

    with client.stream("GET", "/incidents") as resp:
        body = ""
        for chunk in resp.iter_text():
            body += chunk
            if "JOB001" in body:
                break

    assert "data:" in body
    assert "JOB001" in body          # streamed failure emitted
    assert "JOB002" not in body      # COMPLETED row filtered
    assert "WARM01" not in body      # warm-start failure is history, not streamed
    # HISTORY = 1 warm-start + 2 streamed rows; guards against double-population
    assert len(stream.HISTORY) == 3


def test_triage_endpoint_wiring(monkeypatch):
    monkeypatch.setattr(sim, "triage", lambda inc: "restart-and-watch")
    r = client.post("/triage", json={"job_id": "JOB001", "state": "NODE_FAIL"})
    assert r.status_code == 200
    assert r.json()["disposition"] == "restart-and-watch"


def test_model_endpoint_reflects_incumbent():
    clf.reset()
    clf.maybe_promote(None, "gboost", ["power_max"], 0.91)
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1 and body["model_type"] == "gboost" and body["auc"] == 0.91


def test_model_endpoint_empty_when_no_model():
    clf.reset()
    assert client.get("/model").json()["version"] == 0
