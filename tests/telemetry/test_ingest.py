"""Streaming wide->long melt: sparse-aware, never densifies the frame."""

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
