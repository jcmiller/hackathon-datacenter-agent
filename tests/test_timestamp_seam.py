"""Regression: agent runtime tools must handle ISO timestamps (real Kalos /
AcmeTrace), not just the numeric fixtures — and never silently drop bad rows.

These exercise the RUNTIME path (gpusitter.agent.tools.get_telemetry /
find_correlated_failures), which is where real ISO data flows; the mock tests use
numeric time and miss this. See bead 65e (timestamp seam).
"""

import pytest

import gpusitter.agent.tools as tools
from gpusitter.telemetry.window import window_stats

ISO = "2023-08-15 15:30:30+08:00"


def test_get_telemetry_accepts_iso_time(tmp_path, monkeypatch):
    # Real Kalos Time is ISO. Before the fix window_stats float(row[0]) raised and
    # was silently skipped -> samples:0; now it parses ISO and aggregates.
    pw = tmp_path / "power.csv"
    pw.write_text(
        "Time,n1\n"
        "2023-08-15 15:30:00+08:00,10\n"
        "2023-08-15 15:30:30+08:00,90\n"
        "2023-08-15 15:32:00+08:00,10\n"   # outside +/-60s window
    )
    monkeypatch.setattr(tools, "POWER_CSV", str(pw))
    monkeypatch.setattr(tools, "TEMP_CSV", str(pw))
    out = tools.get_telemetry(fail_time=ISO, window=60)
    assert out["DCGM_FI_DEV_POWER_USAGE"]["samples"] == 2   # not 0 (silent-skip bug)
    assert out["DCGM_FI_DEV_POWER_USAGE"]["max"] == 90


def test_find_correlated_failures_accepts_iso_fail_time(tmp_path, monkeypatch):
    p = tmp_path / "trace.csv"
    p.write_text(
        "job_id,type,state,fail_time,node_num,gpu_num,duration,user\n"
        "a,train,NODE_FAIL,2023-08-15 15:30:00+08:00,1,8,1,u\n"
        "b,train,NODE_FAIL,2023-08-15 15:31:00+08:00,1,8,1,u\n"   # 60s later
    )
    monkeypatch.setattr(tools, "TRACE_CSV", str(p))
    out = tools.find_correlated_failures(fail_time="2023-08-15 15:30:00+08:00", window=120)
    # Before fix: float(ISO) raised -> all jobs silently dropped -> count 0.
    assert out["count"] == 1 and out["jobs"] == ["b"] and out["shared_type"] == "train"


def test_window_stats_fails_loud_on_bad_time(tmp_path):
    # A present-but-unparseable Time must raise, not quietly return samples:0.
    p = tmp_path / "bad.csv"
    p.write_text("Time,n1\nnot_a_timestamp,5\n")
    with pytest.raises(ValueError):
        window_stats(str(p), 0, 100)


def test_window_stats_iso_bounds_direct(tmp_path):
    from datetime import datetime

    p = tmp_path / "util.csv"
    p.write_text(
        "Time,n1,n2\n"
        "2023-08-15 15:30:00+08:00,50,60\n"
        "2023-08-15 15:30:30+08:00,90,70\n"
        "2023-08-15 15:31:30+08:00,40,30\n"
    )
    lo = datetime.fromisoformat("2023-08-15 15:29:45+08:00")
    hi = datetime.fromisoformat("2023-08-15 15:30:45+08:00")
    w = window_stats(str(p), lo, hi)
    assert w["samples"] == 2 and w["max"] == 90 and w["min"] == 50
