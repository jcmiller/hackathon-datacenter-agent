import pandas as pd

from gpusitter.detection import stream
from gpusitter.detection.stream import stream_jobs, warm_start


def _write(tmp_path):
    rows = [{"job_id": i, "state": "COMPLETED"} for i in range(5)]
    rows[2]["state"] = "NODE_FAIL"
    rows[4]["state"] = "FAILED"
    p = tmp_path / "jobs.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p), rows


def test_warm_start_includes_nth_failure(tmp_path):
    p, rows = _write(tmp_path)
    pre = warm_start(p, 1)
    assert len(pre) == 3 and pre[-1]["state"] == "NODE_FAIL"  # first failure at index 2


def test_stream_jobs_from_index(tmp_path):
    p, rows = _write(tmp_path)
    out = list(stream_jobs(p, 3))
    assert [r["job_id"] for r in out] == [3, 4]


# --- HISTORY population tests ---


def test_warm_start_populates_history(tmp_path):
    stream.reset_history()
    p, rows = _write(tmp_path)
    prefix = warm_start(p, 1)
    assert stream.HISTORY == prefix


def test_stream_jobs_appends_to_history(tmp_path):
    stream.reset_history()
    p, rows = _write(tmp_path)
    yielded = list(stream_jobs(p, 3))
    assert stream.HISTORY == yielded


def test_reset_history_clears(tmp_path):
    stream.reset_history()
    p, rows = _write(tmp_path)
    warm_start(p, 1)
    assert len(stream.HISTORY) > 0
    stream.reset_history()
    assert stream.HISTORY == []
