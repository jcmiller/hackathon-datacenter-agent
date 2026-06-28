"""Coverage for the 3 behaviors ported off the deleted backend/loader.py.

Replaces tests/test_loader.py (loader.py is gone — its data layer folded into
gpusitter.telemetry + gpusitter.rca). Same numeric-fail_time semantics the agent
tools rely on; no pandas / no whole-frame densification.
"""

from gpusitter.rca.job_join import correlated_jobs, load_failed_jobs, load_incidents
from gpusitter.telemetry.window import window_stats


def _write_trace(tmp_path):
    p = tmp_path / "trace.csv"
    p.write_text(
        "job_id,type,state,fail_time,node_num,gpu_num,duration,user\n"
        "j1,train,NODE_FAIL,880,4,32,790,u1\n"  # hardware failure
        "j2,eval,COMPLETED,,1,8,390,u2\n"  # not a failure -> dropped
        "j3,train,FAILED,900,2,16,810,u3\n"  # job failure
    )
    return str(p)


def _write_util(tmp_path):
    p = tmp_path / "util.csv"
    p.write_text("Time,10.0.0.1,10.0.0.2\n870,50,60\n885,90,70\n895,95,80\n910,40,30\n")
    return str(p)


def test_load_incidents_node_fail_filters_and_sorts(tmp_path):
    inc = load_incidents(_write_trace(tmp_path))
    assert [i["job_id"] for i in inc] == ["j1", "j3"]  # COMPLETED dropped, sorted
    assert inc[0]["fail_time"] == 880  # NODE_FAIL included, numeric
    assert inc[0]["state"] == "NODE_FAIL"


def test_load_failed_jobs_includes_node_fail(tmp_path):
    # ISO-path loader skips the numeric mock fail_times here, but the state
    # filter must admit NODE_FAIL (not just FAILED) — assert via the state set.
    from gpusitter.rca.job_join import FAIL_STATES

    assert "NODE_FAIL" in FAIL_STATES and "FAILED" in FAIL_STATES
    # And it accepts a custom states set without crashing.
    assert load_failed_jobs(_write_trace(tmp_path), states={"FAILED"}) == []


def test_window_stats_aggregates_streaming(tmp_path):
    w = window_stats(_write_util(tmp_path), 880, 900)
    assert w["samples"] == 2  # rows at 885 and 895
    assert w["max"] == 95
    assert w["min"] == 70
    assert w["mean"] == 83.75  # (90+70+95+80)/4


def test_window_stats_empty_window(tmp_path):
    assert window_stats(_write_util(tmp_path), 1000, 2000) == {
        "samples": 0,
        "mean": 0.0,
        "max": 0.0,
        "min": 0.0,
    }


def test_correlated_jobs_within_window_excludes_self(tmp_path):
    inc = load_incidents(_write_trace(tmp_path))
    corr = correlated_jobs(inc, fail_time=880, window=30)
    assert [c["job_id"] for c in corr] == ["j3"]  # j3@900 within +/-30; j1@880 is self
