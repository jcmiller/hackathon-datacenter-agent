"""Tests for the early-detection evaluator's honesty guards (bead lys).

The split is the load-bearing claim behind every reported number: if train can
contain a point from the *future* of a test point, the held-out metrics are
inflated and the go/no-go verdict is unsound. These tests pin the
no-future-leakage property the navigator flagged as missing in the first cut.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
from scripts.eval_early_dataset import TRAIN_FRAC, time_ordered_split

TZ = timezone(timedelta(hours=8))  # Kalos fixed +08:00


def _df(rows):
    """rows = [(gpu, seconds_from_base)] -> a frame with ISO t_ref strings."""
    base = datetime(2023, 8, 15, 0, 0, 0, tzinfo=TZ)
    return pd.DataFrame(
        {
            "gpu": [g for g, _ in rows],
            "t_ref": [(base + timedelta(seconds=s)).isoformat(sep=" ") for _, s in rows],
        }
    )


def test_split_is_strictly_time_ordered_no_future_leakage():
    # GPU a#0 spans a *wide* time range (0s and 1000s). A GPU-bucketed split that
    # assigns each GPU atomically by its earliest t_ref would drop a#0 entirely
    # into train, putting a#0@1000s AFTER test rows -> future leakage. The strict
    # t_ref-threshold split must instead place a#0@0s in train and a#0@1000s in
    # test, with every train point strictly before every test point.
    df = _df([("a#0", 0), ("b#0", 100), ("b#0", 200), ("c#0", 300), ("c#0", 400), ("a#0", 1000)])
    train, test = time_ordered_split(df)

    assert train.sum() > 0 and test.sum() > 0
    t = pd.to_datetime(df["t_ref"]).values
    assert t[train].max() < t[test].min(), "train must be strictly before test"
    # a#0 appears on BOTH sides -> proves it is time-, not GPU-, partitioned.
    gpus_train = set(df["gpu"].values[train])
    gpus_test = set(df["gpu"].values[test])
    assert "a#0" in gpus_train and "a#0" in gpus_test


def test_split_respects_train_fraction_roughly():
    rows = [("g#0", s) for s in range(100)]  # 100 distinct ascending times
    df = _df(rows)
    train, test = time_ordered_split(df)
    # ~70% in train; allow a small slack for the threshold landing on a value.
    assert abs(train.sum() - int(TRAIN_FRAC * len(df))) <= 1
    assert train.sum() + test.sum() == len(df)


def test_split_handles_unsorted_input():
    # Input rows out of time order must still split chronologically.
    df = _df([("g#0", 900), ("g#0", 100), ("g#0", 500), ("g#0", 300), ("g#0", 700)])
    train, test = time_ordered_split(df)
    t = pd.to_datetime(df["t_ref"]).values
    assert t[train].max() < t[test].min()
