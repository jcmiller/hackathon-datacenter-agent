import pytest

import gpusitter.agent.tools as tools


def _resolver(mapping):
    """Fake resolve_metric_csv that maps canonical metric -> CSV path."""

    def resolve(metric, **_kw):
        try:
            return mapping[metric]
        except KeyError as err:  # mimic an un-fetched LFS object
            raise FileNotFoundError(f"LFS object cache not found for {metric}") from err

    return resolve


def test_get_telemetry(tmp_path, monkeypatch):
    import pandas as pd

    pw = tmp_path / "POWER_USAGE.csv"
    tp = tmp_path / "GPU_TEMP.csv"
    pd.DataFrame({"Time": [100, 110, 200], "n1": [10, 90, 10]}).to_csv(pw, index=False)
    pd.DataFrame({"Time": [100, 110, 200], "n1": [40, 42, 41]}).to_csv(tp, index=False)
    monkeypatch.setattr(
        tools, "resolve_metric_csv", _resolver({"POWER_USAGE": str(pw), "GPU_TEMP": str(tp)})
    )
    out = tools.get_telemetry(fail_time=110, window=20)
    power = out["DCGM_FI_DEV_POWER_USAGE"]
    assert power["max"] == 90
    assert out["DCGM_FI_DEV_GPU_TEMP"]["samples"] == 2
    # provenance: canonical source path + metric + window are surfaced.
    assert power["available"] is True
    assert power["source"] == str(pw) and power["metric"] == "POWER_USAGE"
    assert power["window"] == [90, 130]


def test_get_telemetry_missing_data_is_explicit(monkeypatch):
    # Raw LFS object not materialized -> available:false + reason, not a crash.
    monkeypatch.setattr(tools, "resolve_metric_csv", _resolver({}))
    out = tools.get_telemetry(fail_time=110, window=20)
    power = out["DCGM_FI_DEV_POWER_USAGE"]
    assert power["available"] is False
    assert power["reason"] == "raw data not materialized"
    assert power["source"] == "data/utilization/kalos/POWER_USAGE.csv"


def test_find_correlated_failures(tmp_path, monkeypatch):
    import pandas as pd

    p = tmp_path / "trace.csv"
    pd.DataFrame(
        [
            {
                "job_id": "a",
                "type": "train",
                "node_num": 1,
                "gpu_num": 8,
                "state": "NODE_FAIL",
                "fail_time": 1000,
                "duration": 1,
                "user": "u",
            },
            {
                "job_id": "b",
                "type": "train",
                "node_num": 1,
                "gpu_num": 8,
                "state": "NODE_FAIL",
                "fail_time": 1050,
                "duration": 1,
                "user": "u",
            },
        ]
    ).to_csv(p, index=False)
    monkeypatch.setattr(tools, "TRACE_CSV", str(p))
    out = tools.find_correlated_failures(fail_time=1000, window=120)
    assert out["source"] == "jobs" and out["count"] == 1
    assert out["jobs"] == ["b"] and out["shared_type"] == "train"


def test_find_correlated_failures_xid_source(tmp_path, monkeypatch):
    # source="xid": edge-detected Xid onsets near the fail_time, with the code.
    xid = tmp_path / "XID_ERRORS.csv"
    xid.write_text(
        "Time,172.31.0.5-3,172.31.0.6-1\n"
        "2023-08-15 15:30:00+08:00,0.0,0.0\n"
        "2023-08-15 15:30:30+08:00,43.0,0.0\n"  # onset on gpu .5-3 (code 43)
        "2023-08-15 15:30:45+08:00,43.0,43.0\n"  # onset on gpu .6-1 (code 43)
    )
    monkeypatch.setattr(tools, "resolve_metric_csv", _resolver({"XID_ERRORS": str(xid)}))
    out = tools.find_correlated_failures(
        fail_time="2023-08-15 15:30:30+08:00", window=120, source="xid"
    )
    assert out["source"] == "xid" and out["available"] is True
    assert out["count"] == 2
    assert out["gpus"] == ["172.31.0.5#3", "172.31.0.6#1"]
    assert out["observed_xid"] == 43  # integral code, single distinct value
    assert out["artifact"] == str(xid)


def test_find_correlated_failures_xid_window_excludes_far_onset(tmp_path, monkeypatch):
    # An onset outside +/- window must NOT be counted (transformation: narrow window).
    xid = tmp_path / "XID_ERRORS.csv"
    xid.write_text(
        "Time,172.31.0.5-3\n"
        "2023-08-15 15:30:00+08:00,0.0\n"
        "2023-08-15 15:40:00+08:00,43.0\n"  # 10 min after fail_time
    )
    monkeypatch.setattr(tools, "resolve_metric_csv", _resolver({"XID_ERRORS": str(xid)}))
    out = tools.find_correlated_failures(
        fail_time="2023-08-15 15:30:00+08:00", window=120, source="xid"
    )
    assert out["count"] == 0 and out["gpus"] == [] and out["observed_xid"] is None


def test_find_correlated_failures_xid_missing_data(monkeypatch):
    monkeypatch.setattr(tools, "resolve_metric_csv", _resolver({}))
    out = tools.find_correlated_failures(
        fail_time="2023-08-15 15:30:30+08:00", window=120, source="xid"
    )
    assert out["available"] is False
    assert out["artifact"] == "data/utilization/kalos/XID_ERRORS.csv"


def test_find_correlated_failures_xid_requires_iso(monkeypatch):
    # Numeric fail_time can't be matched against wall-clock Xid telemetry.
    monkeypatch.setattr(tools, "resolve_metric_csv", _resolver({"XID_ERRORS": "/nope"}))
    with pytest.raises(ValueError):
        tools.find_correlated_failures(fail_time=1000, window=120, source="xid")


def test_find_correlated_failures_unknown_source():
    with pytest.raises(ValueError):
        tools.find_correlated_failures(fail_time=1000, source="bogus")


def test_page_and_record(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "SOP_PATH", str(tmp_path / "sop.json"))
    assert tools.page_technician("node 4", "drain+replace")["paged"] is True
    assert tools.record_resolution("train", "s", "page_technician", "replaced")["recorded"] is True
    assert tools.search_past_incidents("train")["count"] == 1
