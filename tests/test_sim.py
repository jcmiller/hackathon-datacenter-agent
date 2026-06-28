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


# ---------------------------------------------------------------------------
# /api/monitor — operational reactive trigger (bead i6k)
# ---------------------------------------------------------------------------


def _monitor_feature_df():
    """Small labeled feature table (lys/r7j substrate) with a learnable temp signal."""
    from datetime import datetime, timedelta, timezone

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    t0 = datetime(2023, 8, 15, tzinfo=timezone(timedelta(hours=8)))
    for i in range(40):
        onset = t0 + timedelta(hours=i)
        rows.append(
            {
                "gpu": f"g{i}#0",
                "t_ref": (onset - timedelta(seconds=300)).isoformat(),
                "horizon_s": 300.0,
                "label": 1,
                "temp_last": float(rng.normal(78, 7)),
            }
        )
        for k in range(6):
            t = t0 + timedelta(hours=i, seconds=int(rng.integers(-2000, 2000)))
            rows.append(
                {
                    "gpu": f"decoy{k}#0",
                    "t_ref": t.isoformat(),
                    "horizon_s": 300.0,
                    "label": 0,
                    "temp_last": float(rng.normal(63, 7)),
                }
            )
    return pd.DataFrame(rows).sort_values("t_ref").reset_index(drop=True)


def test_monitor_unavailable_without_incumbent(tmp_path, monkeypatch):
    monkeypatch.setattr(sim, "MONITOR_REGISTRY_PATH", str(tmp_path / "empty_reg"))
    r = client.get("/api/monitor")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert "incumbent" in body["reason"]


def test_monitor_unavailable_without_dataset(tmp_path, monkeypatch):
    # A registry WITH an incumbent but a missing feature table -> honest unavailable.
    from gpusitter.detection.harness import CandidateSpec, ModelRegistry, run_round

    df = _monitor_feature_df()
    data = tmp_path / "train.csv"
    df.to_csv(data, index=False)
    reg_dir = tmp_path / "reg"
    reg = ModelRegistry(str(reg_dir))
    run_round(CandidateSpec("logreg", ("temp_last",)), df, reg, dataset_path=str(data))

    monkeypatch.setattr(sim, "MONITOR_REGISTRY_PATH", str(reg_dir))
    monkeypatch.setattr(sim, "MONITOR_DATA_PATH", str(tmp_path / "missing.csv"))
    body = client.get("/api/monitor").json()
    assert body["available"] is False
    assert "feature table not found" in body["reason"]


def test_monitor_returns_scores_and_miss_grid(tmp_path, monkeypatch):
    from gpusitter.detection.harness import CandidateSpec, ModelRegistry, run_round

    df = _monitor_feature_df()
    data = tmp_path / "early_detection.csv"
    df.to_csv(data, index=False)
    reg_dir = tmp_path / "reg"
    reg = ModelRegistry(str(reg_dir))
    run_round(CandidateSpec("logreg", ("temp_last",)), df, reg, dataset_path=str(data))
    assert reg.incumbent is not None, "fixture must promote a usable incumbent"

    monkeypatch.setattr(sim, "MONITOR_REGISTRY_PATH", str(reg_dir))
    monkeypatch.setattr(sim, "MONITOR_DATA_PATH", str(data))

    body = client.get("/api/monitor", params={"budget": 0.10, "horizon": 300.0}).json()
    assert body["available"] is True
    assert body["model_version"] >= 1
    assert body["n_onsets"] == 40
    assert body["rows"], "per-row scores must be exposed to the dashboard"
    row = body["rows"][0]
    assert {"risk_score", "alert_flag", "model_version", "gpu", "t_ref"} <= set(row)
    grid = body["budgets"][0]["grid"]["by_horizon"]["300"]
    assert grid["n_onsets"] == 40
    assert 0.0 <= grid["recall"] <= 1.0
    # The miss detector actually distinguishes: a skilled incumbent at 10% budget
    # catches a non-trivial share of onsets (more than the budget floor).
    assert grid["caught"] > 4, grid
