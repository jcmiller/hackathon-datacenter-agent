"""TelemetryStore query surface — the contract w28 (env) + eku (Xid) consume."""

from gpusitter.telemetry.store import TelemetryStore


def test_value_and_snapshot_across_metrics(gpu_temp_csv, power_usage_csv):
    store = TelemetryStore.load(
        {"GPU_TEMP": gpu_temp_csv, "POWER_USAGE": power_usage_csv}
    )

    # metric[gpu][t] point lookup.
    assert store.value("GPU_TEMP", "172.31.0.5#3", "2023-08-15 00:00:00+08:00") == 50.0
    assert (
        store.value("POWER_USAGE", "172.31.0.5#3", "2023-08-15 00:00:00+08:00") == 300.0
    )

    # per-(t,gpu) snapshot stitches both metrics for one GPU at one instant.
    snap = store.snapshot("172.31.0.5#3", "2023-08-15 00:00:00+08:00")
    assert snap == {"GPU_TEMP": 50.0, "POWER_USAGE": 300.0}


def test_missing_lookup_returns_none_not_zero(gpu_temp_csv):
    store = TelemetryStore.load({"GPU_TEMP": gpu_temp_csv})
    # idle GPU/time pair was empty in the source -> absent, not 0.0.
    assert store.value("GPU_TEMP", "172.31.0.9#0", "2023-08-15 00:00:00+08:00") is None
    assert store.snapshot("172.31.0.9#0", "2023-08-15 00:00:00+08:00") == {}


def test_id_normalization_joins_two_differently_named_metrics(
    gpu_temp_csv, sm_active_csv, pod_to_ip_alias
):
    # GPU_TEMP is IP-named, SM_ACTIVE is pod-named; the alias proves they are the
    # same physical GPU. Snapshot must merge both metrics under one canonical id.
    store = TelemetryStore.load(
        {"GPU_TEMP": gpu_temp_csv, "SM_ACTIVE": sm_active_csv},
        alias=pod_to_ip_alias,
    )
    snap = store.snapshot("172.31.0.5#3", "2023-08-15 00:00:00+08:00")
    assert snap == {"GPU_TEMP": 50.0, "SM_ACTIVE": 0.80}


def test_without_alias_metrics_do_not_merge(gpu_temp_csv, sm_active_csv):
    # Non-vacuity: drop the alias and the join must collapse — the pod GPU and
    # the IP GPU become two separate canonical ids.
    store = TelemetryStore.load(
        {"GPU_TEMP": gpu_temp_csv, "SM_ACTIVE": sm_active_csv}
    )
    snap = store.snapshot("172.31.0.5#3", "2023-08-15 00:00:00+08:00")
    assert snap == {"GPU_TEMP": 50.0}
    assert "lingjun-pod9-0001#3" in store.gpus()


def test_series_returns_sorted_timeseries(power_usage_csv):
    store = TelemetryStore.load({"POWER_USAGE": power_usage_csv})
    series = store.series("POWER_USAGE", "172.31.0.5#3")
    assert series == [
        ("2023-08-15 00:00:00+08:00", 300.0),
        ("2023-08-15 00:00:15+08:00", 310.0),
        ("2023-08-15 00:00:30+08:00", 305.0),
    ]


def test_window_returns_metrics_over_time_range(gpu_temp_csv, power_usage_csv):
    store = TelemetryStore.load(
        {"GPU_TEMP": gpu_temp_csv, "POWER_USAGE": power_usage_csv}
    )
    win = store.window(
        "172.31.0.5#3",
        "2023-08-15 00:00:00+08:00",
        "2023-08-15 00:00:15+08:00",
    )
    assert win == {
        "2023-08-15 00:00:00+08:00": {"GPU_TEMP": 50.0, "POWER_USAGE": 300.0},
        "2023-08-15 00:00:15+08:00": {"GPU_TEMP": 52.0, "POWER_USAGE": 310.0},
    }
    # 00:00:30 is outside the window.
    assert "2023-08-15 00:00:30+08:00" not in win


def test_load_respects_time_range_and_downsample(power_usage_csv):
    # downsample keeps every 2nd row -> rows 0 and 2 (00 and 30), drops 15.
    store = TelemetryStore.load({"POWER_USAGE": power_usage_csv}, downsample=2)
    ts = store.timestamps("172.31.0.5#3")
    assert ts == ["2023-08-15 00:00:00+08:00", "2023-08-15 00:00:30+08:00"]


def test_gpus_and_metrics_enumerated(gpu_temp_csv, power_usage_csv):
    store = TelemetryStore.load(
        {"GPU_TEMP": gpu_temp_csv, "POWER_USAGE": power_usage_csv}
    )
    assert set(store.metrics()) == {"GPU_TEMP", "POWER_USAGE"}
    assert "172.31.0.5#3" in store.gpus()
    assert "172.31.0.9#0" in store.gpus()
