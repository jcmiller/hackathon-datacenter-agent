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


def test_model_endpoint_provisional_fallback(tmp_path, monkeypatch):
    """With no rigorous registry AND no committed fixture, the in-process triage fit
    surfaces — but badged provisional so it cannot be mistaken for the incumbent."""
    monkeypatch.setattr(sim, "MONITOR_REGISTRY_PATH", str(tmp_path / "empty_reg"))
    monkeypatch.setattr(sim, "MONITOR_DATA_PATH", str(tmp_path / "no_data.parquet"))
    monkeypatch.setattr(sim, "MONITOR_FIXTURE_DATA_PATH", str(tmp_path / "no_fixture.csv"))
    clf.reset()
    clf.maybe_promote(None, "logreg", ["power_spike_ratio"], 0.87, n_samples=5)
    r = client.get("/api/model")
    assert r.status_code == 200
    body = r.json()
    assert body["model"]["version"] == 1
    assert body["model"]["model_type"] == "logreg"
    # Provenance: explicitly NOT the rigorous registry.
    assert body["source"] == "in_process"
    assert body["rigorous"] is False
    assert "registry" in body["note"]
    clf.reset()


def test_model_endpoint_serves_registry_incumbent(tmp_path, monkeypatch):
    """The canonical /api/model card is the rigorous registry incumbent — the SAME
    model /api/monitor scores — not the weak maybe_promote number (AC#1/#2/#4)."""
    from gpusitter.detection.harness import CandidateSpec, ModelRegistry, run_round

    df = _monitor_feature_df()
    data = tmp_path / "early_detection.csv"
    df.to_csv(data, index=False)
    reg_dir = tmp_path / "reg"
    reg = ModelRegistry(str(reg_dir))
    _, result = run_round(CandidateSpec("logreg", ("temp_last",)), df, reg, dataset_path=str(data))
    assert reg.incumbent is not None, "fixture must promote a usable incumbent"

    monkeypatch.setattr(sim, "MONITOR_REGISTRY_PATH", str(reg_dir))
    monkeypatch.setattr(sim, "MONITOR_DATA_PATH", str(data))

    # A stale, contradictory in-process fit must NOT win over the registry.
    clf.reset()
    clf.maybe_promote(None, "tree", ["power_spike_ratio"], 0.99, n_samples=5)

    body = client.get("/api/model").json()
    assert body["source"] == "registry"
    assert body["rigorous"] is True
    card = body["model"]
    assert card["primary_metric"] == "roc_auc"
    # Canonical, not the weak path: registry AUC, registry model_type, registry version.
    assert card["val_auc"] == round(result.candidate_value, 3)
    assert card["model_type"] == "logreg"
    assert card["model_type"] != "tree"
    assert card["holdout_id"]

    # Unified identity: /api/model and /api/monitor report the same model version.
    mon = client.get("/api/monitor").json()
    assert mon["available"] is True
    assert card["version"] == mon["model_version"]
    clf.reset()


def test_model_monitor_fixture_parity_off_droplet(tmp_path, monkeypatch):
    """Off-droplet (no real registry, no real feature table), /api/model and
    /api/monitor must serve the SAME committed fixture-backed model (bead aow + jds).

    Regression for the rejection: previously /api/monitor showed the fixture model
    while /api/model returned model=null because it ignored jds's fixture resolver.
    Uses the committed fixture only — no droplet artifacts."""
    # Force "off the droplet": the real artifacts are absent.
    monkeypatch.setattr(sim, "MONITOR_REGISTRY_PATH", str(tmp_path / "no_registry"))
    monkeypatch.setattr(sim, "MONITOR_DATA_PATH", str(tmp_path / "no_data.parquet"))
    # MONITOR_FIXTURE_* keep their committed defaults — the portable demo artifact.
    clf.reset()
    # Even a stale in-process fit must not shadow the fixture-backed registry.
    clf.maybe_promote(None, "tree", ["power_spike_ratio"], 0.99, n_samples=5)

    model = client.get("/api/model").json()
    monitor = client.get("/api/monitor").json()

    # Both are the committed fixture, both badged illustrative.
    assert model["source"] == "registry"
    assert model["rigorous"] is True
    assert model["fixture"] is True
    assert "fixture_note" in model
    assert monitor["available"] is True
    assert monitor["fixture"] is True

    # Parity: one model on both surfaces — not a fixture-vs-null (or fixture-vs-real) mix.
    assert model["model"] is not None
    assert model["model"]["version"] == monitor["model_version"]
    assert model["model"]["model_type"] == "logreg"  # the fixture incumbent, not the tree fit
    clf.reset()


# ---------------------------------------------------------------------------
# /api/predict-gpu — per-GPU failure-likelihood (scope A)
# ---------------------------------------------------------------------------


def test_predict_gpu_scores_real_telemetry_window():
    """A GPU with a bundled telemetry window is scored by the real incumbent."""
    r = client.post("/api/predict-gpu", json={"incident_id": "INC-001"})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert 0.0 <= body["likelihood"] <= 1.0
    feats = body["features"]
    for name in (
        "GPU_TEMP_last",
        "GPU_TEMP_mean",
        "GPU_TEMP_slope",
        "POWER_USAGE_last",
        "MEMORY_TEMP_last",
    ):
        assert name in feats
    # MEMORY_TEMP is an honest proxy for GPU_TEMP (no memory series exists).
    assert feats["MEMORY_TEMP_last"] == feats["GPU_TEMP_last"]
    # Slope must be in training units (degrees per SECOND, ~±0.05). Computing it
    # per sample-index instead inflates it ~15x and saturates predict_proba to 0.
    assert abs(feats["GPU_TEMP_slope"]) < 1.0
    assert body["label"] in {"alert", "watch", "ok"}
    assert body["note"]


def test_predict_gpu_slope_is_per_second_not_per_index():
    """Regression: a hot, rising GPU must not score ~0 from a slope-unit bug.

    INC-009 (Xid 94) climbs from ~62C to ~74C over the window. With the slope
    computed per sample-index the model saturated to ~2e-7; per-second it lands
    in a sane mid-range.
    """
    body = client.post("/api/predict-gpu", json={"incident_id": "INC-009"}).json()
    assert body["available"] is True
    assert abs(body["features"]["GPU_TEMP_slope"]) < 0.5  # per-second, small
    assert body["likelihood"] > 0.01  # not saturated to zero


def test_predict_gpu_unknown_incident_is_unavailable():
    r = client.post("/api/predict-gpu", json={"incident_id": "NOPE-999"})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert "telemetry" in body["reason"]


# ---------------------------------------------------------------------------
# /api/learning-curve — self-improving classifier learning curve artifact
# ---------------------------------------------------------------------------


def test_learning_curve_serves_bundled_artifact():
    r = client.get("/api/learning-curve")
    assert r.status_code == 200
    body = r.json()
    for key in ("curve", "baseline_v0", "real_data_reference", "dataset", "honest_note"):
        assert key in body
    assert isinstance(body["curve"], list) and body["curve"]
    first = body["curve"][0]
    assert "version" in first
    assert "roc_auc" in first


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


def _disable_fixture(monkeypatch, tmp_path):
    """Point the committed-demo-fixture fallback at empty paths.

    The jds contract change: /api/monitor now falls back to a committed demo
    fixture when the real artifacts are absent. To exercise the honest
    ``available:false`` floor we must also remove the fixture — otherwise the
    fallback (correctly) serves the demo.
    """
    monkeypatch.setattr(sim, "MONITOR_FIXTURE_DATA_PATH", str(tmp_path / "no_fixture.csv"))
    monkeypatch.setattr(sim, "MONITOR_FIXTURE_REGISTRY_PATH", str(tmp_path / "no_fixture_reg"))


def test_monitor_unavailable_when_all_artifacts_absent(tmp_path, monkeypatch):
    # No real registry/data AND no committed fixture -> honest unavailable.
    monkeypatch.setattr(sim, "MONITOR_REGISTRY_PATH", str(tmp_path / "empty_reg"))
    monkeypatch.setattr(sim, "MONITOR_DATA_PATH", str(tmp_path / "missing.parquet"))
    _disable_fixture(monkeypatch, tmp_path)
    r = client.get("/api/monitor")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body.get("fixture") is None  # honest floor, not a degraded fixture render
    assert "demo fixture" in body["reason"]


def test_monitor_unavailable_without_dataset(tmp_path, monkeypatch):
    # A registry WITH an incumbent but a missing feature table AND no fixture ->
    # honest unavailable (the real-artifact branch needs BOTH halves present).
    from gpusitter.detection.harness import CandidateSpec, ModelRegistry, run_round

    df = _monitor_feature_df()
    data = tmp_path / "train.csv"
    df.to_csv(data, index=False)
    reg_dir = tmp_path / "reg"
    reg = ModelRegistry(str(reg_dir))
    run_round(CandidateSpec("logreg", ("temp_last",)), df, reg, dataset_path=str(data))

    monkeypatch.setattr(sim, "MONITOR_REGISTRY_PATH", str(reg_dir))
    monkeypatch.setattr(sim, "MONITOR_DATA_PATH", str(tmp_path / "missing.csv"))
    _disable_fixture(monkeypatch, tmp_path)
    body = client.get("/api/monitor").json()
    assert body["available"] is False


def test_monitor_renders_off_droplet_from_committed_fixture():
    # No monkeypatching: the real droplet artifacts (data/early_detection.parquet,
    # models/early_detection) do NOT exist in the worktree, so the endpoint must
    # serve the COMMITTED demo fixture and render the full monitor surface. This is
    # the bead's load-bearing smoke (AC #1, #3, #5).
    body = client.get("/api/monitor", params={"budget": 0.10}).json()
    assert body["available"] is True
    assert body["fixture"] is True, "off-droplet must serve the committed fixture"
    assert "early-detection-eval.md" in body["fixture_note"], "fixture must be labeled illustrative"
    assert body["model_version"] >= 1
    assert body["n_onsets"] > 0
    assert body["rows"], "per-row scores must be exposed to the dashboard"
    row = body["rows"][0]
    assert {"risk_score", "alert_flag", "model_version", "gpu", "t_ref"} <= set(row)
    grid = body["budgets"][0]["grid"]["by_horizon"]
    recalls = [grid[h]["recall"] for h in ("60", "300", "600")]
    assert all(0.0 <= rcl <= 1.0 for rcl in recalls)
    assert any(rcl > 0.0 for rcl in recalls), grid


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
