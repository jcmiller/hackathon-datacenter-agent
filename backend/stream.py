import pandas as pd

FAIL_STATES = {"NODE_FAIL", "FAILED"}
HISTORY = []

def reset_history():
    HISTORY.clear()

def _load(path):
    return pd.read_csv(path).to_dict("records")

def warm_start(path, n_incidents):
    records = _load(path)
    count = 0
    for i, r in enumerate(records):
        if r["state"] in FAIL_STATES:
            count += 1
            if count == n_incidents:
                return records[: i + 1]
    return records

def stream_jobs(path, start_index):
    for r in _load(path)[start_index:]:
        yield r
