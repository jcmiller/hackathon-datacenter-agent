"""Tests for gpusitter.app.sim — run offline, no API key needed."""

import csv
import pathlib

from fastapi.testclient import TestClient

import gpusitter.app.sim as sim
import gpusitter.detection.classifier as clf

client = TestClient(sim.app)


def test_record_resolution_emits_pending_update(tmp_path, monkeypatch):
    import gpusitter.agent.memory as mem_mod
    import gpusitter.agent.tools as tools_mod

    sop = tmp_path / "sop.json"
    vec_path = tmp_path / "sop_vectors.json"
    monkeypatch.setattr(tools_mod, "SOP_PATH", str(sop))
    monkeypatch.setattr(mem_mod, "_SOP_VECTORS", str(vec_path))
    monkeypatch.setattr(mem_mod, "_embed", lambda text: [0.1, 0.2, 0.3])
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
    "job_id",
    "user",
    "node_num",
    "gpu_num",
    "cpu_num",
    "type",
    "state",
    "submit_time",
    "start_time",
    "end_time",
    "duration",
    "queue",
    "gpu_time",
    "fail_time",
    "stop_time",
]


def _write_sample_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "sample.csv"
    rows = [
        # NODE_FAIL row — should appear in SSE stream
        {
            "job_id": "JOB001",
            "user": "alice",
            "node_num": 4,
            "gpu_num": 8,
            "cpu_num": 32,
            "type": "GPU_TRAIN",
            "state": "NODE_FAIL",
            "submit_time": 1000,
            "start_time": 1010,
            "end_time": 1100,
            "duration": 90,
            "queue": "gpu",
            "gpu_time": 720,
            "fail_time": "2023-05-17 11:17:30+00:00",
            "stop_time": 1100,
        },
        # COMPLETED row — filtered out by load_incidents
        {
            "job_id": "JOB002",
            "user": "bob",
            "node_num": 2,
            "gpu_num": 4,
            "cpu_num": 16,
            "type": "GPU_TRAIN",
            "state": "COMPLETED",
            "submit_time": 2000,
            "start_time": 2010,
            "end_time": 2200,
            "duration": 190,
            "queue": "gpu",
            "gpu_time": 760,
            "fail_time": "",
            "stop_time": 2200,
        },
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
    sim._incidents_cache.clear()

    with client.stream("GET", "/api/incidents") as resp:
        body = ""
        for chunk in resp.iter_text():
            body += chunk
            if "JOB001" in body:
                break

    assert "data:" in body
    assert "JOB001" in body
    assert "JOB002" not in body


def test_triage_endpoint_wiring(monkeypatch):
    def fake_stream(inc):
        yield {"type": "tool_call", "tool": "get_telemetry", "args": "{}"}
        yield {
            "type": "disposition",
            "disposition": "RESTART_AND_WATCH",
            "action": "ok",
            "ticket": None,
            "summary": "ok...",
        }

    monkeypatch.setattr(sim, "triage_stream", fake_stream)
    with client.stream("POST", "/api/triage", json={"job_id": "J1"}) as r:
        body = "".join(r.iter_text())
    assert "tool_call" in body
    assert "RESTART_AND_WATCH" in body


def test_triage_stream_yields_events(monkeypatch):

    import gpusitter.agent.agent as agent_mod

    class FakeSession:
        id = "s1"

    class FakeSessionService:
        def create_session_sync(self, **kwargs):
            return FakeSession()

    class FakeRunner:
        session_service = FakeSessionService()

        def run(self, **kwargs):
            from unittest.mock import MagicMock

            tc = MagicMock()
            tc.name = "get_telemetry"
            tc.args = {"fail_time": 100}
            ev1 = MagicMock()
            ev1.get_function_calls = lambda: [tc]
            ev1.get_function_responses = lambda: []
            ev1.content = None
            tc2 = MagicMock()
            tc2.name = "check_degradation_trend"
            tc2.args = {"fail_time": "2023-05-17T11:17:30+00:00"}
            ev1b = MagicMock()
            ev1b.get_function_calls = lambda: [tc2]
            ev1b.get_function_responses = lambda: []
            ev1b.content = None
            tr = MagicMock()
            tr.response = {"DCGM_FI_DEV_POWER_USAGE": 200}
            ev2 = MagicMock()
            ev2.get_function_calls = lambda: []
            ev2.get_function_responses = lambda: [tr]
            ev2.content = None
            part = MagicMock()
            part.text = "restart recommended"
            content = MagicMock()
            content.parts = [part]
            ev3 = MagicMock()
            ev3.get_function_calls = lambda: []
            ev3.get_function_responses = lambda: []
            ev3.content = content
            return [ev1, ev1b, ev2, ev3]

    monkeypatch.setattr(agent_mod, "InMemoryRunner", lambda **kw: FakeRunner())
    events = list(agent_mod.triage_stream({"job_id": "J1", "fail_time": 100}))
    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "observation" in types
    assert "disposition" in types


def test_model_endpoint_reflects_incumbent():
    clf.reset()
    clf.maybe_promote(None, "logreg", ["power_spike_ratio"], 0.87, n_samples=5)
    r = client.get("/api/model")
    assert r.status_code == 200
    body = r.json()
    assert body["model"]["version"] == 1
    assert body["model"]["model_type"] == "logreg"
    clf.reset()
