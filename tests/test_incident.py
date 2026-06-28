"""Canonical incident model for the RCA agent (bead 8m6).

The agent's incident model is an empty-aware per-GPU Xid onset (rca/d8z) surfaced
operationally as an i6k miss. These tests pin the OBSERVED-vs-INFERRED separation
and the adapters that build the canonical incident from each real source.
"""

from datetime import datetime, timedelta, timezone

from gpusitter.agent.incident import (
    canonical_incident,
    format_incident_prompt,
    incident_from_miss,
    incident_from_onset,
)

BASE = datetime(2023, 8, 15, 15, 30, 0, tzinfo=timezone(timedelta(hours=8)))


def _make_miss(**overrides):
    from gpusitter.detection.monitor import MissEvent

    kw = {
        "gpu": "172.31.15.220-4",
        "onset_t": BASE,
        "horizon_s": 600.0,
        "model_version": 3,
        "budget": 0.05,
        "caught": False,
        "n_alerts_in_window": 0,
        "prior_scores": [(BASE.isoformat(), 0.12, False)],
        "pre_event_features": {"power_max": 410.0},
    }
    kw.update(overrides)
    return MissEvent(**kw)


# --- incident_from_miss: the operational (i6k) incident -----------------------


def test_miss_becomes_incident_with_inferred_xid_and_detection_block():
    inc = incident_from_miss(_make_miss())
    assert inc["source"] == "i6k_miss"
    # A MissEvent does NOT carry the Xid code — onset location/time are observed,
    # the code is not. So observed.xid must be None (the agent recovers it via xid).
    assert inc["observed"]["xid"] is None
    assert inc["observed"]["gpu"] == "172.31.15.220-4"
    assert inc["observed"]["onset_t"] == BASE.isoformat()
    # fail_time is the onset wall-clock the tools key on.
    assert inc["fail_time"] == BASE.isoformat()
    det = inc["detection"]
    assert det is not None
    assert det["missed"] is True
    assert det["budget"] == 0.05
    assert det["horizon_s"] == 600.0
    assert det["model_version"] == 3
    assert det["prior_scores"] == [(BASE.isoformat(), 0.12, False)]


def test_caught_onset_is_not_a_miss():
    inc = incident_from_miss(_make_miss(caught=True, n_alerts_in_window=2))
    assert inc["detection"]["missed"] is False


# --- incident_from_onset: a raw rca onset (may carry the code) ----------------


def test_onset_with_code_is_observed_fact():
    inc = incident_from_onset("172.31.15.220-4", BASE, code=43.0)
    assert inc["source"] == "xid_onset"
    assert inc["observed"]["xid"] == 43  # integral float coerced to int
    assert inc["observed"]["gpu"] == "172.31.15.220-4"
    assert inc["detection"] is None


def test_onset_without_code_leaves_xid_unobserved():
    inc = incident_from_onset("172.31.15.220-4", BASE)
    assert inc["observed"]["xid"] is None


# --- canonical_incident: normalize arbitrary raw dicts ------------------------


def test_dashboard_fixture_lifts_observed_xid():
    raw = {
        "id": "INC-001",
        "ts": "2023-08-17 06:00:30+08:00",
        "gpu": {"node": "172.31.15.220", "idx": 4},
        "xid": 43,
        "severity": "warn",
        "correlatedCount": 115,
    }
    inc = canonical_incident(raw)
    assert inc["incident_id"] == "INC-001"
    assert inc["observed"]["xid"] == 43
    assert inc["observed"]["gpu"] == "172.31.15.220-4"
    assert inc["fail_time"] == "2023-08-17 06:00:30+08:00"
    assert inc["source"] == "xid_onset"


def test_job_trace_record_has_no_observed_xid():
    # rca.job_join.load_incidents shape: counts, not a sensed Xid code.
    raw = {
        "job_id": "job-77",
        "type": "training",
        "node_num": "1",
        "gpu_num": "8",
        "state": "FAILED",
        "fail_time": 1000,
    }
    inc = canonical_incident(raw)
    assert inc["incident_id"] == "job-77"
    assert inc["observed"]["xid"] is None
    assert inc["observed"]["gpu"] is None  # no per-GPU id in the trace
    assert inc["fail_time"] == 1000
    assert inc["source"] == "job_trace"


def test_canonical_incident_is_idempotent():
    inc = incident_from_onset("g0", BASE, code=79)
    assert canonical_incident(inc) == inc


def test_empty_string_xid_is_not_observed():
    inc = canonical_incident({"id": "X", "xid": ""})
    assert inc["observed"]["xid"] is None


# --- format_incident_prompt: observed vs no-Xid fallback (prompt snapshots) ---


def test_prompt_observed_xid_threads_the_code():
    inc = incident_from_onset("172.31.15.220-4", BASE, code=43)
    prompt = format_incident_prompt(inc)
    assert "43" in prompt
    assert "OBSERVED" in prompt
    # An observed code must be presented as ground truth, never as a thing to infer.
    assert "not directly observed" not in prompt
    assert "172.31.15.220-4" in prompt


def test_prompt_no_xid_forbids_asserting_a_code():
    inc = incident_from_miss(_make_miss())
    prompt = format_incident_prompt(inc)
    assert "not directly observed" in prompt
    # The miss framing is what the agent reasons over.
    assert "MISS" in prompt or "miss" in prompt
    assert "600" in prompt  # horizon surfaced


def test_prompt_distinguishes_the_two_cases():
    observed = format_incident_prompt(incident_from_onset("g", BASE, code=94))
    inferred = format_incident_prompt(incident_from_onset("g", BASE))
    assert observed != inferred
