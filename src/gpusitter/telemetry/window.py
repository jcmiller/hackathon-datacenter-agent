"""Window aggregate over a wide telemetry CSV — streaming, never densified.

Replaces backend/loader.telemetry_window with the SAME result shape
({samples, mean, max, min}; samples = in-window ROW count, stats over every
non-Time cell) but streams row-by-row instead of ``pd.read_csv``-ing the whole
frame — so it scales to the real ~750 MB kalos files without materializing them
(the densification q2o was built to avoid).
"""

import csv

from .timeparse import parse_time_value


def _empty():
    return {"samples": 0, "mean": 0.0, "max": 0.0, "min": 0.0}


def window_stats(path: str, start, end) -> dict:
    """Aggregate non-Time cells over rows whose Time is in [start, end].

    Time may be relative-second floats OR ISO timestamps (real Kalos); ``start``/
    ``end`` must be the same kind (use :func:`window_bounds`). A present-but-
    unparseable Time raises — never silently skipped — so real-data windows can't
    quietly return samples:0.

    Only wide ``Time`` + GPU-column CSVs are handled. Non-time-series artifacts
    (e.g. a ``.pkl`` CDF) are rejected upstream by
    :func:`gpusitter.telemetry.sources.validate_timeseries_csv` before reaching
    here, so there is no pickle branch.
    """
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
            if not row or (len(row) == 1 and row[0].strip() == ""):
                continue  # genuinely blank line, not a data row
            t = parse_time_value(row[0])  # raises on bad Time -> fail loud
            if not (start <= t <= end):
                continue
            rows += 1
            for cell in row[1:]:
                if cell == "":
                    continue  # idle/unallocated cell — sparse, expected
                v = float(cell)
                n += 1
                total += v
                mx = v if mx is None or v > mx else mx
                mn = v if mn is None or v < mn else mn
    if n == 0:
        return _empty()
    return {"samples": rows, "mean": total / n, "max": mx, "min": mn}
