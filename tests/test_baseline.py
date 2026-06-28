"""Tests for the v0 no-skill baseline (bead 8co).

The baseline is the floor of the learning curve, so its load-bearing property is
that it has *no learned skill*: its held-out ROC-AUC is chance (0.5) even on data
with a strong, learnable signal — a real model on the identical split beats it.
Each test contrasts the no-skill floor against a model that should clear it, so a
baseline that accidentally peeked at the features would fail.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from gpusitter.detection.baseline import BASELINE_NAME, evaluate_baseline
from gpusitter.detection.harness import (
    CandidateSpec,
    ModelRegistry,
    run_round,
    time_ordered_split,
)

TZ = timezone(timedelta(hours=8))
BASE = datetime(2023, 8, 15, 0, 0, 0, tzinfo=TZ)


def _signal_df(n: int = 400, *, signal: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n // 2)
    rng.shuffle(labels)
    if signal:
        temp = np.where(labels == 1, rng.normal(78, 7, n), rng.normal(62, 7, n))
    else:
        temp = rng.normal(70, 7, n)
    return pd.DataFrame(
        {
            "gpu": [f"node{i % 7}#0" for i in range(n)],
            "t_ref": [(BASE + timedelta(seconds=30 * i)).isoformat() for i in range(n)],
            "horizon_s": np.tile([60.0, 300.0], n // 2),
            "label": labels,
            "temp_last": temp,
        }
    )


def test_baseline_is_chance_even_with_a_strong_signal_a_real_model_beats(tmp_path):
    # Data carries a strong learnable signal. The no-skill floor must STILL be ~0.5
    # (it ignores features by design); a real logreg on the SAME split must clear it.
    df = _signal_df(signal=True, seed=1)
    base = evaluate_baseline(df)

    assert base.name == BASELINE_NAME
    assert abs(base.roc_auc - 0.5) < 0.02, f"no-skill floor must be chance, got {base.roc_auc}"

    reg = ModelRegistry(str(tmp_path / "reg"))
    data = str(tmp_path / "d.csv")
    df.to_csv(data, index=False)
    ev, _ = run_round(CandidateSpec("logreg", ("temp_last",)), df, reg, dataset_path=data)
    assert ev.primary_value > base.roc_auc + 0.1, "a real model must rise above the floor"


def test_baseline_avg_precision_collapses_to_base_rate():
    df = _signal_df(signal=True, seed=2)
    base = evaluate_baseline(df)
    # A constant score ranks nothing, so AP == the test base rate (positives / total).
    assert base.avg_precision == pytest.approx(base.base_rate, abs=1e-9)


def test_baseline_scores_on_the_same_split_as_the_harness():
    df = _signal_df(n=300, signal=True, seed=3)
    train_mask, test_mask = time_ordered_split(df, 0.7)
    base = evaluate_baseline(df, train_frac=0.7)
    assert base.n_train == int(train_mask.sum())
    assert base.n_test == int(test_mask.sum())


def test_baseline_refuses_an_unscorable_single_class_holdout():
    # Latest (held-out) rows all positive -> single-class test set -> not scorable.
    n = 200
    cut = int(n * 0.7)
    labels = np.zeros(n, dtype=int)
    labels[:cut] = np.tile([0, 1], cut // 2)
    labels[cut:] = 1
    df = pd.DataFrame(
        {
            "gpu": [f"node{i % 5}#0" for i in range(n)],
            "t_ref": [(BASE + timedelta(seconds=30 * i)).isoformat() for i in range(n)],
            "label": labels,
            "temp_last": np.zeros(n),
        }
    )
    with pytest.raises(ValueError, match="not scorable"):
        evaluate_baseline(df)
