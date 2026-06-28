import pandas as pd

FAIL_STATES = {"NODE_FAIL", "FAILED"}

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
