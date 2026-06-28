"""Tests for the operational reactive trigger (bead i6k).

The load-bearing properties: (1) the alert threshold fires ~budget fraction of rows
*derived from the score distribution*, not a constant; (2) a real-signal incumbent
CATCHES onsets a no-skill scorer MISSES — so the miss detector genuinely
distinguishes models, it is not vacuously firing or never firing; (3) onsets are
recovered exactly from the labeled feature table; (4) wider horizons never lose
recall; (5) a MISS drives the injected retrain trainer.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from gpusitter.detection.harness import CandidateSpec, ModelRegistry, run_round
from gpusitter.detection.monitor import (
    DEFAULT_HORIZONS_S,
    Onset,
    RowScorer,
    calibrate_threshold,
    detect_misses,
    horizon_grid,
    monitor_report,
    onsets_from_dataset,
    react_to_misses,
    score_rows,
)

TZ = timezone(timedelta(hours=8))
BASE = datetime(2023, 8, 15, 0, 0, 0, tzinfo=TZ)


def _signal_df(n: int = 400, *, signal: bool, seed: int = 0) -> pd.DataFrame:
    """Labeled early-detection table; positives carry a learnable temp shift iff signal."""
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n // 2)
    rng.shuffle(labels)
    if signal:
        # Learnable but NOT leaking: ~1.5 sigma separation (univariate AUC ~0.85,
        # under the harness's ~0.99 single-feature leakage trip).
        temp = np.where(labels == 1, rng.normal(76, 8, n), rng.normal(64, 8, n))
    else:
        temp = rng.normal(70, 7, n)
    return pd.DataFrame(
        {
            "gpu": [f"node{i % 7}#0" for i in range(n)],
            "t_ref": [(BASE + timedelta(seconds=30 * i)).isoformat() for i in range(n)],
            "horizon_s": np.tile([60.0, 300.0], n // 2),
            "label": labels,
            "temp_last": temp,
            "power_last": rng.normal(250, 30, n),
        }
    )


class _RandomScorer:
    """A no-skill scorer: random risk per row (RowScorer-shaped).

    Unlike a constant scorer (whose tied scores would fire on 100% of rows once
    ``score >= threshold``), random scores honor the SAME alert budget as the real
    model — so the comparison is fair: equal alert rate, and only skill separates
    the catch counts.
    """

    features = ("temp_last",)
    model_version = 0

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def score(self, df):
        return self._rng.random(len(df))


def _fit_scorer(df, tmp_path, features=("temp_last", "power_last")) -> RowScorer:
    reg = ModelRegistry(str(tmp_path / "reg"))
    data = str(tmp_path / "d.csv")
    df.to_csv(data, index=False)
    run_round(CandidateSpec("logreg", features), df, reg, dataset_path=data)
    return RowScorer.from_registry(reg)


# --- alert budget -------------------------------------------------------------


def test_threshold_fires_approximately_budget_fraction():
    scores = np.linspace(0.0, 1.0, 1000)
    for budget in (0.01, 0.05, 0.10):
        thr = calibrate_threshold(scores, budget)
        fired = float((scores >= thr).mean())
        assert abs(fired - budget) < 0.02, f"budget {budget}: fired {fired}"


def test_threshold_is_derived_not_constant():
    # Two different score distributions must yield two different thresholds —
    # proves the threshold tracks the data, not a hand-tuned constant.
    lo = calibrate_threshold(np.linspace(0, 0.2, 500), 0.05)
    hi = calibrate_threshold(np.linspace(0.8, 1.0, 500), 0.05)
    assert hi > lo + 0.5


def test_empty_scores_threshold_fires_nothing():
    assert calibrate_threshold(np.array([]), 0.05) == float("inf")


# --- per-row scoring ----------------------------------------------------------


def test_scored_row_carries_score_flag_and_version(tmp_path):
    df = _signal_df(signal=True, seed=1)
    scorer = _fit_scorer(df, tmp_path)
    result = score_rows(df, scorer, budget=0.10)

    assert len(result.rows) == len(df)
    for r in result.rows:
        assert 0.0 <= r.risk_score <= 1.0
        assert r.model_version == scorer.model_version >= 1
        assert r.alert_flag == (r.risk_score >= result.threshold)
    # ~budget fraction fired (derived threshold), not all/none.
    assert 0.02 < result.alert_rate < 0.30


def test_rows_returned_in_time_order(tmp_path):
    df = _signal_df(signal=True, seed=4)
    scorer = _fit_scorer(df, tmp_path)
    rows = score_rows(df, scorer, budget=0.10).rows
    ts = [r.t_ref for r in rows]
    assert ts == sorted(ts)


# --- onset reconstruction -----------------------------------------------------


def test_onsets_reconstructed_exactly_from_positives():
    # Two positives at different horizons that encode the SAME onset must dedup to one.
    df = pd.DataFrame(
        {
            "gpu": ["n#0", "n#0", "n#1"],
            "t_ref": [
                BASE.isoformat(),
                (BASE - timedelta(seconds=240)).isoformat(),  # 300-60 earlier
                (BASE + timedelta(seconds=1000)).isoformat(),
            ],
            "horizon_s": [60.0, 300.0, 60.0],
            "label": [1, 1, 1],
            "temp_last": [80.0, 80.0, 80.0],
        }
    )
    onsets = onsets_from_dataset(df)
    # rows 0 and 1 both point at BASE+60 for n#0; row 2 -> n#1 at +1060.
    assert Onset("n#0", BASE + timedelta(seconds=60)) in onsets
    assert sum(o.gpu == "n#0" for o in onsets) == 1
    assert len(onsets) == 2


def test_onsets_come_only_from_positives():
    df = _signal_df(signal=True, seed=2)
    onsets = onsets_from_dataset(df)
    n_pos = int((df["label"] == 1).sum())
    assert onsets, "fixture must contain positives"
    # Onsets derive only from positives, deduped across horizons -> never exceed them.
    assert 0 < len(onsets) <= n_pos
    # An all-negative table yields no onsets (transformation: positives drive onsets).
    neg = df.copy()
    neg["label"] = 0
    assert onsets_from_dataset(neg) == []


# --- miss detection: a real model catches what a no-skill scorer misses --------


def _separable_df(n_onsets: int = 30, n_decoys: int = 240, seed: int = 0) -> pd.DataFrame:
    """A scenario where catching an onset requires scoring the RIGHT row.

    Each onset lives on its own GPU with one high-temp positive row inside its
    horizon window. Many low-temp decoy negatives live on onset-free GPUs, outside
    every window. A skilled scorer spends its alert budget on the high-temp
    positives (catching onsets); a random scorer wastes most of the budget on the
    decoys (catching few). Separation is kept ~1.1 sigma so the harness does not
    flag it as leakage.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_onsets):
        onset = BASE + timedelta(hours=i)
        t_ref = onset - timedelta(seconds=300)  # positive sits 5 min before onset
        rows.append(
            {
                "gpu": f"g{i}#0",
                "t_ref": t_ref.isoformat(),
                "horizon_s": 300.0,
                "label": 1,
                "temp_last": float(rng.normal(78, 7)),
            }
        )
    for j in range(n_decoys):
        # decoys: onset-free GPUs, low temp, scattered across the same time span
        onset_idx = j % n_onsets
        t = BASE + timedelta(hours=onset_idx, seconds=int(rng.integers(-3000, 3000)))
        rows.append(
            {
                "gpu": f"decoy{j % 12}#0",
                "t_ref": t.isoformat(),
                "horizon_s": 300.0,
                "label": 0,
                "temp_last": float(rng.normal(63, 7)),
            }
        )
    return pd.DataFrame(rows).sort_values("t_ref").reset_index(drop=True)


def test_real_model_catches_onsets_the_noskill_scorer_misses(tmp_path):
    df = _separable_df(seed=7)
    onsets = onsets_from_dataset(df)
    assert len(onsets) == 30, "one onset per positive GPU"

    real = _fit_scorer(df, tmp_path, features=("temp_last",))
    real_result = score_rows(df, real, budget=0.10)
    rand_result = score_rows(df, _RandomScorer(seed=7), budget=0.10)
    # Fair comparison: both honor the ~10% alert budget.
    assert abs(real_result.alert_rate - rand_result.alert_rate) < 0.03

    H = 300.0
    real_caught = sum(
        e.caught
        for e in detect_misses(
            real_result.rows, onsets, horizon_s=H, model_version=real.model_version, budget=0.10
        )
    )
    rand_caught = sum(
        e.caught
        for e in detect_misses(rand_result.rows, onsets, horizon_s=H, model_version=0, budget=0.10)
    )
    # At equal alert budget, the discriminating model catches strictly more real
    # onsets than chance — otherwise the miss detector / scoring is vacuous.
    assert real_caught > rand_caught + 5, (real_caught, rand_caught)


def test_miss_event_carries_prior_scores_and_pre_event_features(tmp_path):
    df = _signal_df(n=600, signal=True, seed=8)
    onsets = onsets_from_dataset(df)
    scorer = _fit_scorer(df, tmp_path)
    rows = score_rows(df, scorer, budget=0.05).rows
    events = detect_misses(
        rows, onsets, horizon_s=600.0, model_version=scorer.model_version, budget=0.05
    )
    # At least one onset has rows in its window; that event must carry the payload.
    with_window = [e for e in events if e.prior_scores]
    assert with_window, "expected onsets with prior rows in-window"
    e = with_window[0]
    assert e.horizon_s == 600.0
    assert "temp_last" in e.pre_event_features
    assert all(len(ps) == 3 for ps in e.prior_scores)


# --- horizon grid -------------------------------------------------------------


def test_grid_records_horizon_and_recall_is_monotone(tmp_path):
    df = _signal_df(n=600, signal=True, seed=9)
    onsets = onsets_from_dataset(df)
    scorer = _fit_scorer(df, tmp_path)
    rows = score_rows(df, scorer, budget=0.10).rows
    grid = horizon_grid(
        rows, onsets, horizons_s=DEFAULT_HORIZONS_S, model_version=scorer.model_version, budget=0.10
    )["by_horizon"]

    recalls = [grid[str(int(h))]["recall"] for h in DEFAULT_HORIZONS_S]
    # Wider horizon admits >= alerts, so recall is non-decreasing.
    assert recalls == sorted(recalls), recalls
    for h in DEFAULT_HORIZONS_S:
        cell = grid[str(int(h))]
        assert cell["horizon_s"] == h
        for miss in cell["misses"]:
            assert miss["horizon_s"] == h  # H recorded on every miss event (AC #4)


def test_monitor_report_is_json_serializable(tmp_path):
    import json

    df = _signal_df(n=500, signal=True, seed=11)
    scorer = _fit_scorer(df, tmp_path)
    report = monitor_report(df, scorer, budgets=(0.05, 0.10))
    blob = json.dumps(report)  # must not raise
    assert report["available"] is True
    assert report["model_version"] >= 1
    assert len(report["budgets"]) == 2
    assert "by_horizon" in report["budgets"][0]["grid"]
    assert "JSON" or blob


# --- miss -> retrain trigger --------------------------------------------------


def test_miss_triggers_injected_trainer(tmp_path):
    df = _signal_df(n=300, signal=True, seed=3)
    reg = ModelRegistry(str(tmp_path / "reg"))
    misses = [
        # one synthetic miss is enough to fire
        _make_miss(),
    ]
    calls = {}

    def fake_trainer(d, r):
        calls["fired"] = (len(d), r)
        return "retrained"

    out = react_to_misses(df, reg, misses, min_misses=1, trainer=fake_trainer)
    assert out.triggered is True
    assert out.result == "retrained"
    assert calls["fired"][0] == len(df)


def test_no_misses_does_not_trigger(tmp_path):
    df = _signal_df(n=100, signal=True, seed=5)
    reg = ModelRegistry(str(tmp_path / "reg"))
    called = {"n": 0}

    def fake_trainer(d, r):
        called["n"] += 1
        return None

    out = react_to_misses(df, reg, [], min_misses=1, trainer=fake_trainer)
    assert out.triggered is False
    assert called["n"] == 0


def test_from_registry_without_incumbent_raises(tmp_path):
    reg = ModelRegistry(str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="no persisted incumbent"):
        RowScorer.from_registry(reg)


def _make_miss():
    from gpusitter.detection.monitor import MissEvent

    return MissEvent(
        gpu="n#0",
        onset_t=BASE,
        horizon_s=600.0,
        model_version=1,
        budget=0.10,
        caught=False,
        n_alerts_in_window=0,
        prior_scores=[],
        pre_event_features={},
    )
