"""Streaming wide->long melt: sparse-aware, never densifies the frame."""

import csv

import pytest

from gpusitter.telemetry.ingest import LongRecord, iter_long_records
from gpusitter.telemetry.normalize import parse_gpu_id


def test_melt_wide_to_long_changes_shape_and_keeps_values(gpu_temp_csv):
    records = list(iter_long_records(gpu_temp_csv, "GPU_TEMP"))

    # Transformation assertion: wide frame is 3 GPU cols x 2 timestamps = 6 cells;
    # 2 cells are empty (idle GPUs), so the long form has 4 records, not 6.
    # before (dense cells) != after (long records) -> melt actually happened.
    dense_cells = 3 * 2
    assert len(records) == 4
    assert len(records) != dense_cells

    # Exact values survive the melt.
    by_key = {(r.gpu.canonical, r.t): r.value for r in records}
    assert by_key[("172.31.0.5#3", "2023-08-15 00:00:00+08:00")] == 50.0
    assert by_key[("172.31.0.5#4", "2023-08-15 00:00:00+08:00")] == 51.0
    assert by_key[("172.31.0.5#3", "2023-08-15 00:00:15+08:00")] == 52.0
    assert by_key[("172.31.0.9#0", "2023-08-15 00:00:15+08:00")] == 60.0
    assert all(isinstance(r, LongRecord) and r.metric == "GPU_TEMP" for r in records)


def test_empty_cells_are_skipped_not_zero_filled(gpu_temp_csv):
    records = list(iter_long_records(gpu_temp_csv, "GPU_TEMP"))
    keys = {(r.gpu.canonical, r.t) for r in records}
    # idle GPU 172.31.0.9-0 at t0 and 172.31.0.5-4 at t1 were empty -> absent,
    # NOT present with value 0.0.
    assert ("172.31.0.9#0", "2023-08-15 00:00:00+08:00") not in keys
    assert ("172.31.0.5#4", "2023-08-15 00:00:15+08:00") not in keys
    assert all(r.value is not None for r in records)


def test_time_range_filter(power_usage_csv):
    recs = list(
        iter_long_records(
            power_usage_csv,
            "POWER_USAGE",
            time_range=("2023-08-15 00:00:15+08:00", "2023-08-15 00:00:30+08:00"),
        )
    )
    times = {r.t for r in recs}
    assert "2023-08-15 00:00:00+08:00" not in times
    assert times == {"2023-08-15 00:00:15+08:00", "2023-08-15 00:00:30+08:00"}


def test_scan_stops_at_window_end(tmp_path):
    # Early-stop relies on time-ordered rows: once a row exceeds the window end,
    # scanning stops. A later in-range row placed AFTER an out-of-range row is
    # therefore NOT emitted — this pins break-semantics (vs. a plain skip).
    p = tmp_path / "GPU_TEMP.csv"
    p.write_text(
        "Time,172.31.0.5-3\n"
        "2023-08-15 00:00:00+08:00,50.0\n"
        "2023-08-15 00:00:30+08:00,52.0\n"  # > hi -> triggers stop
        "2023-08-15 00:00:15+08:00,51.0\n"  # in range but unreachable after stop
    )
    recs = list(
        iter_long_records(
            str(p),
            "GPU_TEMP",
            time_range=("2023-08-15 00:00:00+08:00", "2023-08-15 00:00:15+08:00"),
        )
    )
    assert [r.t for r in recs] == ["2023-08-15 00:00:00+08:00"]


def test_gpus_filter_uses_canonical_ids(power_usage_csv):
    keep = {"172.31.0.5#3"}
    recs = list(iter_long_records(power_usage_csv, "POWER_USAGE", gpus=keep))
    assert {r.gpu.canonical for r in recs} == keep


def test_alias_applied_during_ingest(sm_active_csv, pod_to_ip_alias):
    recs = list(iter_long_records(sm_active_csv, "SM_ACTIVE", alias=pod_to_ip_alias))
    # pod-named column normalized to the IP canonical id at read time.
    assert parse_gpu_id("172.31.0.5-3") in {r.gpu for r in recs}
    assert "172.31.0.5#3" in {r.gpu.canonical for r in recs}


# --- Pre-resolved kept-column optimization (aieng26hack-fo1) -----------------
#
# iter_long_records resolves the kept GPU columns once from the header instead
# of re-scanning all columns per row. The optimization MUST be output-identical
# to a naive all-columns brute-force melt. The oracle below re-implements the
# original O(rows*all_cols) logic directly so the regression test fails if the
# pre-resolution ever drifts from naive semantics (e.g. wrong column offset,
# broken ragged-row handling, dropped filter).


def _brute_force_melt(path, metric, *, time_range=None, gpus=None, alias=None):
    """Reference melt: scan every column of every in-window row (the old path)."""
    keep = set(gpus) if gpus is not None else None
    lo, hi = time_range if time_range is not None else (None, None)
    out = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col_gpus = [parse_gpu_id(name, alias=alias) for name in header[1:]]
        for row in reader:
            if not row:
                continue
            t = row[0]
            if lo is not None and t < lo:
                continue
            if hi is not None and t > hi:
                break
            for gpu, cell in zip(col_gpus, row[1:], strict=False):
                if cell == "":
                    continue
                if keep is not None and gpu.canonical not in keep:
                    continue
                out.append(LongRecord(t=t, gpu=gpu, metric=metric, value=float(cell)))
    return out


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"gpus": {"172.31.0.5#3"}},
        {"gpus": {"172.31.0.5#3", "172.31.0.9#0"}},
        {"gpus": set()},  # filter keeps nothing
        {"gpus": {"172.31.99.99#7"}},  # filter matches no real column
        {"time_range": ("2023-08-15 00:00:15+08:00", "2023-08-15 00:00:30+08:00")},
        {
            "gpus": {"172.31.0.9#0"},
            "time_range": ("2023-08-15 00:00:15+08:00", "2023-08-15 00:00:30+08:00"),
        },
    ],
)
def test_optimized_reader_matches_brute_force(power_usage_csv, kwargs):
    expected = _brute_force_melt(power_usage_csv, "POWER_USAGE", **kwargs)
    actual = list(iter_long_records(power_usage_csv, "POWER_USAGE", **kwargs))
    # Sequence equality pins both the values AND the emission order.
    assert actual == expected


def test_optimized_reader_matches_brute_force_ragged(tmp_path):
    # Rows shorter than the header (missing trailing columns) must skip the
    # absent cells exactly as the old zip(strict=False) short-stop did — for
    # both an unfiltered read and one whose kept column lands past the row end.
    p = tmp_path / "GPU_TEMP.csv"
    p.write_text(
        "Time,172.31.0.5-3,172.31.0.5-4,172.31.0.9-0\n"
        "2023-08-15 00:00:00+08:00,50.0\n"  # only col 1 present; 2 & 3 ragged-missing
        "2023-08-15 00:00:15+08:00,52.0,53.0,60.0\n"
    )
    for kwargs in ({}, {"gpus": {"172.31.0.9#0"}}, {"gpus": {"172.31.0.5#4"}}):
        expected = _brute_force_melt(str(p), "GPU_TEMP", **kwargs)
        actual = list(iter_long_records(str(p), "GPU_TEMP", **kwargs))
        assert actual == expected


def test_kept_columns_resolved_once_not_per_row(tmp_path, monkeypatch):
    # The whole point of the bead: parse_gpu_id (and thus canonical resolution)
    # runs once per column at header time, NOT once per data cell. With 3 columns
    # and 100 rows the naive path would parse 3*? per row; the optimized path
    # parses exactly 3 (one per header column).
    import gpusitter.telemetry.ingest as ingest_mod

    calls = {"n": 0}
    real_parse = ingest_mod.parse_gpu_id

    def counting_parse(name, alias=None):
        calls["n"] += 1
        return real_parse(name, alias=alias)

    monkeypatch.setattr(ingest_mod, "parse_gpu_id", counting_parse)

    p = tmp_path / "GPU_TEMP.csv"
    rows = "".join(f"2023-08-15 00:00:{i:02d}+08:00,1.0,2.0,3.0\n" for i in range(60))
    p.write_text("Time,172.31.0.5-3,172.31.0.5-4,172.31.0.9-0\n" + rows)

    list(iter_long_records(str(p), "GPU_TEMP", gpus={"172.31.0.5#3"}))
    assert calls["n"] == 3  # one parse per header column, independent of row count
