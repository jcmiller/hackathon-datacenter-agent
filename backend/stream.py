import pandas as pd
from backend.dataset import FAIL_STATES

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
                prefix = records[: i + 1]
                HISTORY.clear()
                HISTORY.extend(prefix)
                return prefix
    HISTORY.clear()
    HISTORY.extend(records)
    return records

def stream_jobs(path, start_index):
    for r in _load(path)[start_index:]:
        HISTORY.append(r)
        yield r
