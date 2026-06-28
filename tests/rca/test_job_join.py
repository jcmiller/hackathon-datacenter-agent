"""Temporal RCA join: FAILED jobs vs Xid onset events (cross-timezone)."""

import textwrap
from datetime import datetime

from gpusitter.rca.job_join import (
    coincidence,
    load_failed_jobs,
    stream_xid_onsets,
    xid_onset_events,
    xid_sample_span,
)
from gpusitter.telemetry.store import TelemetryStore

# Onset at 13:57:15 +08:00  ==  05:57:15 UTC.
ONSET_LOCAL = "2023-08-29 13:57:15+08:00"


def _parse_iso(ts):
    return datetime.fromisoformat(ts)


def _store(tmp_path):
    xid = tmp_path / "XID_ERRORS.csv"
    # Span 13:00..14:30 (+08:00) so a job can be in-window yet far from the onset.
    xid.write_text(
        textwrap.dedent(
            f"""\
            Time,172.31.0.5-3
            2023-08-29 13:00:00+08:00,0.0
            2023-08-29 13:57:00+08:00,0.0
            {ONSET_LOCAL},43.0
            2023-08-29 14:30:00+08:00,43.0
            """
        )
    )
    return TelemetryStore.load({"XID_ERRORS": str(xid)})


def _trace(tmp_path):
    # header mirrors trace_kalos; state is field 9, fail_time field 13.
    p = tmp_path / "trace_kalos.csv"
    p.write_text(
        "job_id,user,node_num,gpu_num,cpu_num,mem_per_pod_GB,shared_mem_per_pod,"
        "type,state,submit_time,start_time,end_time,fail_time,stop_time,duration,"
        "queue,gpu_time\n"
        # near: 05:57:10 UTC == 13:57:10 +08:00, ~5s before the onset -> MATCH
        "jNEAR,u,8,64,1,1,1,Other,FAILED,,,,2023-08-29 05:57:10+00:00,,1,1,1\n"
        # far: 05:10:00 UTC == 13:10 +08:00 — in the telemetry span but ~47 min
        # from the 13:57 onset -> NO MATCH.
        "jFAR,u,8,64,1,1,1,Other,FAILED,,,,2023-08-29 05:10:00+00:00,,1,1,1\n"
        # COMPLETED job -> excluded entirely
        "jOK,u,8,64,1,1,1,Other,COMPLETED,,,,,,1,1,1\n"
        # FAILED but outside the telemetry window (May) -> filtered out
        "jMAY,u,8,64,1,1,1,Other,FAILED,,,,2023-05-17 11:00:58+00:00,,1,1,1\n"
    )
    return str(p)


def test_load_failed_jobs_filters_state_and_missing_fail_time(tmp_path):
    jobs = load_failed_jobs(_trace(tmp_path))
    ids = {j.job_id for j in jobs}
    assert "jOK" not in ids  # not FAILED
    assert {"jNEAR", "jFAR", "jMAY"} == ids  # all FAILED with a fail_time
    # fail_time parsed as a tz-aware datetime (UTC offset preserved).
    near = next(j for j in jobs if j.job_id == "jNEAR")
    assert near.fail_dt.utcoffset().total_seconds() == 0


def test_window_filter_keeps_only_telemetry_span(tmp_path):
    store = _store(tmp_path)
    jobs = load_failed_jobs(_trace(tmp_path))
    onsets = xid_onset_events(store)
    results, rate = coincidence(onsets, jobs, w_minutes=5, telemetry_span=store_span(store))
    ids = {r["job_id"] for r in results}
    assert "jMAY" not in ids  # May is outside the Aug telemetry window


def test_coincidence_matches_across_timezone(tmp_path):
    store = _store(tmp_path)
    jobs = load_failed_jobs(_trace(tmp_path))
    onsets = xid_onset_events(store)
    results, rate = coincidence(onsets, jobs, w_minutes=5, telemetry_span=store_span(store))
    by_id = {r["job_id"]: r for r in results}
    # UTC fail_time within 5 min of a +08:00 onset must match (cross-tz compare).
    assert by_id["jNEAR"]["matched"] is True
    assert by_id["jNEAR"]["n_onsets"] >= 1
    # Hours away -> no match.
    assert by_id["jFAR"]["matched"] is False
    assert by_id["jFAR"]["n_onsets"] == 0
    # rate = matched / in-window jobs (jNEAR matched, jFAR not) = 0.5
    assert rate == 0.5


def test_xid_onset_events_are_edge_detected(tmp_path):
    onsets = xid_onset_events(_store(tmp_path))
    # exactly one onset (0->43), not three latched samples.
    assert len(onsets) == 1
    assert onsets[0][1] == "172.31.0.5#3"


def test_xid_sample_span_is_wider_than_onset_span(tmp_path):
    # Sample coverage runs T0..T2; the only onset is at T1. The span must reflect
    # the full sample window (T0..T2), not the onset (T1..T1).
    xid = tmp_path / "XID_ERRORS.csv"
    xid.write_text(
        "Time,a-0\n"
        "2023-08-29 13:00:00+08:00,0.0\n"
        "2023-08-29 13:57:15+08:00,43.0\n"
        "2023-08-29 14:30:00+08:00,43.0\n"
    )
    lo, hi = xid_sample_span(str(xid))
    assert lo == _parse_iso("2023-08-29 13:00:00+08:00")
    assert hi == _parse_iso("2023-08-29 14:30:00+08:00")


def test_job_in_sample_window_outside_onset_span_counts_unmatched(tmp_path):
    # Regression for the validate_rca span bug: a FAILED job inside the telemetry
    # SAMPLE window but far from any onset must stay in the denominator (counted)
    # and unmatched — not be dropped by using the narrower onset span.
    xid = tmp_path / "XID_ERRORS.csv"
    xid.write_text(
        "Time,a-0\n"
        "2023-08-29 13:00:00+08:00,0.0\n"
        "2023-08-29 13:57:15+08:00,43.0\n"  # the only onset
        "2023-08-29 14:30:00+08:00,43.0\n"  # sample continues, no onset
    )
    onsets = stream_xid_onsets(str(xid))
    span = xid_sample_span(str(xid))
    # job fails at 14:25 +08:00 == 06:25 UTC: in [13:00,14:30] span, ~28 min from
    # the 13:57 onset -> in-window, unmatched.
    from gpusitter.rca.job_join import FailedJob

    job = FailedJob("jTAIL", _parse_iso("2023-08-29 06:25:00+00:00"))
    results, rate = coincidence(onsets, [job], w_minutes=5, telemetry_span=span)
    assert len(results) == 1  # counted (not dropped)
    assert results[0]["matched"] is False
    assert rate == 0.0


def test_stream_xid_onsets_is_empty_aware(tmp_path):
    # a: 0 -> 43 (healthy->fault, onset). b: nonzero at first observation
    # (pre-trace history unknown -> NOT counted). c: empty(idle) -> 43
    # (idle->fault, a REAL onset that a 0-only rule would miss). d: latched
    # 43 -> 43 (not an onset).
    xid = tmp_path / "XID_ERRORS.csv"
    xid.write_text(
        "Time,a-0,b-0,c-0,d-0\n"
        "2023-08-29 13:57:00+08:00,0.0,43.0,,43.0\n"
        "2023-08-29 13:57:15+08:00,43.0,43.0,43.0,43.0\n"
    )
    onsets = stream_xid_onsets(str(xid))
    gpus = {g for _, g in onsets}
    assert gpus == {"a#0", "c#0"}  # 0->fault and idle->fault; not b (start) or d (latched)


# Small helper the tests share; also exercised as part of the public surface.
def store_span(store):
    from gpusitter.rca.job_join import telemetry_span as _span

    return _span(store)
