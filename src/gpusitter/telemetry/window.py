"""Window aggregate over a wide telemetry CSV — streaming, never densified.

Replaces backend/loader.telemetry_window with the SAME result shape
({samples, mean, max, min}; samples = in-window ROW count, stats over every
non-Time cell) but streams row-by-row instead of ``pd.read_csv``-ing the whole
frame — so it scales to the real ~750 MB kalos files without materializing them
(the densification q2o was built to avoid).
"""

import csv


def _empty():
    return {"samples": 0, "mean": 0.0, "max": 0.0, "min": 0.0}


def window_stats(path: str, start, end) -> dict:
    """Aggregate non-Time cells over rows whose numeric Time is in [start, end]."""
    if path.endswith(".pkl"):
        return _pickle_window_stats(path, start, end)
    rows = n = 0
    total = 0.0
    mx = mn = None
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            next(reader)  # header: Time, <gpu cols...>
        except StopIteration:
            return _empty()
        for row in reader:
            if not row:
                continue
            try:
                t = float(row[0])
            except ValueError:
                continue
            if not (start <= t <= end):
                continue
            rows += 1
            for cell in row[1:]:
                if cell == "":
                    continue
                try:
                    v = float(cell)
                except ValueError:
                    continue
                n += 1
                total += v
                mx = v if mx is None or v > mx else mx
                mn = v if mn is None or v < mn else mn
    if n == 0:
        return _empty()
    return {"samples": rows, "mean": total / n, "max": mx, "min": mn}


def _pickle_window_stats(path, start, end):
    # SECURITY: read_pickle executes arbitrary code — only load .pkl from the
    # trusted InternLM/AcmeTrace release. Prefer CSV when both exist.
    import pandas as pd

    df = pd.read_pickle(path)
    win = df[(df["Time"] >= start) & (df["Time"] <= end)]
    cols = [c for c in win.columns if c != "Time"]
    vals = win[cols].to_numpy().ravel()
    if vals.size == 0:
        return _empty()
    return {
        "samples": len(win),
        "mean": float(vals.mean()),
        "max": float(vals.max()),
        "min": float(vals.min()),
    }
