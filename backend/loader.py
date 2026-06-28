import pandas as pd

FAIL_STATES = {"NODE_FAIL", "FAILED"}
_KEEP = ["job_id","type","node_num","gpu_num","state","fail_time","duration","user"]

def load_incidents(path):
    df = pd.read_csv(path)
    df = df[df["state"].isin(FAIL_STATES) & df["fail_time"].notna() & (df["fail_time"] != "")]
    df["fail_time"] = pd.to_datetime(df["fail_time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    df = df.sort_values("fail_time")
    return df[_KEEP].to_dict("records")

def _read_telemetry(path):
    return pd.read_csv(path)

def telemetry_window(csv_path, start, end):
    df = _read_telemetry(csv_path)
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    start_ts = pd.to_datetime(start, utc=True)
    end_ts   = pd.to_datetime(end,   utc=True)
    win = df[(df["Time"] >= start_ts) & (df["Time"] <= end_ts)]
    cols = [c for c in win.columns if c != "Time"]
    vals = win[cols].to_numpy().ravel()
    if vals.size == 0:
        return {"samples": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
    import numpy as np
    vals = vals[~np.isnan(vals.astype(float))]
    if vals.size == 0:
        return {"samples": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
    return {"samples": int(len(win)), "mean": float(vals.astype(float).mean()),
            "max": float(vals.astype(float).max()), "min": float(vals.astype(float).min())}

def correlated_failures(incidents, fail_ts, window):
    """fail_ts is a pd.Timestamp (UTC). incidents have fail_time as ISO string."""
    result = []
    for i in incidents:
        ts = pd.to_datetime(i["fail_time"], utc=True)
        if ts != fail_ts and abs((ts - fail_ts).total_seconds()) <= window:
            result.append(i)
    return result
