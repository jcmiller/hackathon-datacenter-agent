import pandas as pd
from scripts.precompute_features import precompute_jobs


def test_precompute_schema_label_and_window(tmp_path):
    trace = tmp_path / "trace.csv"
    pd.DataFrame(
        [
            {
                "job_id": "j1",
                "type": "train",
                "node_num": 1,
                "gpu_num": 8,
                "cpu_num": 16,
                "duration": 90,
                "queue": 5,
                "mem_per_pod_GB": 80,
                "state": "FAILED",
                "start_time": "2023-08-20 10:00:00+00:00",
                "end_time": "2023-08-20 10:05:00+00:00",
                "fail_time": "2023-08-20 10:04:00+00:00",
            },
            {
                "job_id": "j2",
                "type": "eval",
                "node_num": 1,
                "gpu_num": 8,
                "cpu_num": 16,
                "duration": 90,
                "queue": 5,
                "mem_per_pod_GB": 80,
                "state": "COMPLETED",
                "start_time": "2023-08-20 10:00:00+00:00",
                "end_time": "2023-08-20 10:05:00+00:00",
                "fail_time": "",
            },
        ]
    ).to_csv(trace, index=False)
    # power telemetry: Time (ISO) + two per-GPU columns; the 10:02 sample is inside j1's window
    power = tmp_path / "power.csv"
    pd.DataFrame(
        {
            "Time": ["2023-08-20 10:02:00+00:00", "2023-08-20 11:00:00+00:00"],
            "172.31.13.235-0": [100.0, 999.0],
            "172.31.13.235-1": [300.0, 999.0],
        }
    ).to_csv(power, index=False)
    out = tmp_path / "jobs.csv"
    precompute_jobs(str(trace), str(power), None, None, str(out))
    df = pd.read_csv(out)
    assert {
        "job_id",
        "gpu_num",
        "power_mean",
        "power_max",
        "power_std",
        "temp_mean",
        "util_mean",
        "state",
        "fail_time",
    } <= set(df.columns)
    # only the 10:02 sample (100, 300) is in-window; the 11:00 sample (999) is excluded
    assert df.loc[df.job_id == "j1", "power_max"].iloc[0] == 300.0
    assert df.loc[df.job_id == "j1", "power_mean"].iloc[0] == 200.0
    assert df.loc[df.job_id == "j2", "temp_mean"].iloc[0] == 0.0  # no temp csv -> 0.0
