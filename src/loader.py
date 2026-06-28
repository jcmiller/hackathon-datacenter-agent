import pandas as pd

FAIL_STATES = {"NODE_FAIL", "FAILED"}
_KEEP = ["job_id","type","node_num","gpu_num","state","fail_time","duration","user"]

def load_incidents(path):
    df = pd.read_csv(path)
    df = df[df["state"].isin(FAIL_STATES) & df["fail_time"].notna()]
    df = df.sort_values("fail_time")
    return df[_KEEP].to_dict("records")

def _read_telemetry(path):
    # AcmeTrace ships CSV for some signals, pickle for others.
    # SECURITY: read_pickle executes arbitrary code — only load .pkl from the
    # trusted InternLM/AcmeTrace release you downloaded yourself. Prefer CSV when both exist.
    return pd.read_pickle(path) if path.endswith(".pkl") else pd.read_csv(path)

def telemetry_window(csv_path, start, end):
    df = _read_telemetry(csv_path)
    win = df[(df["Time"] >= start) & (df["Time"] <= end)]
    cols = [c for c in win.columns if c != "Time"]
    vals = win[cols].to_numpy().ravel()
    if vals.size == 0:
        return {"samples": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
    return {"samples": len(win), "mean": float(vals.mean()),
            "max": float(vals.max()), "min": float(vals.min())}

def correlated_failures(incidents, fail_time, window):
    return [i for i in incidents
            if i["fail_time"] != fail_time
            and abs(i["fail_time"] - fail_time) <= window]
