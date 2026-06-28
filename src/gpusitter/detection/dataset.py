import pandas as pd

FAIL_STATES = {"NODE_FAIL", "FAILED"}
SOP_FEATURES = ["power_spike_ratio", "temp_rise_C", "correlated_count"]


def build_xy(history, features):
    df = pd.DataFrame(history)
    y = df["state"].isin(FAIL_STATES).astype(int).tolist()
    cols = [f for f in features if f != "type"]
    X = df[cols].copy() if cols else pd.DataFrame(index=df.index)
    if "type" in features:
        X = pd.concat([X, pd.get_dummies(df["type"], prefix="type")], axis=1)
    return X, y


def time_split(X, y, val_frac=0.3):
    k = int(len(y) * (1 - val_frac))
    return X.iloc[:k], y[:k], X.iloc[k:], y[k:]


def build_xy_from_sop(entries):
    """Extract numeric features from SOP entries for disposition classification."""
    rows = [e for e in entries if e.get("metrics")]
    if not rows:
        return [], [], SOP_FEATURES
    X = [[e["metrics"].get(f, 0.0) for f in SOP_FEATURES] for e in rows]
    y = [
        0 if str(e.get("disposition", "")).lower() in ("restart_and_watch", "restart") else 1
        for e in rows
    ]
    return X, y, SOP_FEATURES


def time_split_lists(X, y, val_frac=0.3):
    k = max(1, int(len(y) * (1 - val_frac)))
    return X[:k], y[:k], X[k:], y[k:]
