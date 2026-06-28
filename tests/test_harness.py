"""Tests for the eval harness + model registry (bead glf).

The harness is the judge the self-improving loop optimizes against, so these
tests pin the load-bearing properties: the split never leaks the future,
keep-if-better actually rejects a weaker candidate, and a promoted model survives
a process restart as a *usable* model (not just metadata). Each transformation
test sets up the "wrong" state and asserts the corrected state, so deleting the
call under test fails the test.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from gpusitter.detection.harness import (
    CandidateSpec,
    ModelRegistry,
    evaluate_candidate,
    run_round,
    sha256_file,
    time_ordered_split,
)

TZ = timezone(timedelta(hours=8))  # Kalos fixed +08:00
BASE = datetime(2023, 8, 15, 0, 0, 0, tzinfo=TZ)


def _signal_df(n: int = 400, *, signal: bool = True, seed: int = 0) -> pd.DataFrame:
    """Synthetic early-detection table with a tunable warning signal.

    ``temp_last`` carries the signal: positives run hotter. With ``signal=False``
    the feature is pure noise, so a model cannot beat chance — used to prove the
    metrics are real, not vacuous. Rows are spread over ascending ``t_ref`` and
    balanced across two horizons so the per-horizon table and time split are
    exercised. Labels alternate so both classes appear in every time slice (the
    strict split would otherwise hand back a single-class fold).
    """
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n // 2)
    horizons = np.tile([60.0, 300.0], n // 2)
    if signal:
        # Moderate, overlapping separation: a real but imperfect warning signal,
        # so AUC lands ~0.8 (not saturated at 1.0) and weak/strong candidates
        # actually differ. temp_last carries most signal; power_mean is weak.
        temp = np.where(labels == 1, rng.normal(75, 8, n), rng.normal(65, 8, n))
        power = np.where(labels == 1, rng.normal(355, 30, n), rng.normal(345, 30, n))
    else:
        temp = rng.normal(70, 8, n)
        power = rng.normal(350, 30, n)
    return pd.DataFrame(
        {
            "gpu": [f"node{i % 7}#0" for i in range(n)],
            "t_ref": [(BASE + timedelta(seconds=30 * i)).isoformat() for i in range(n)],
            "horizon_s": horizons,
            "label": labels,
            "temp_last": temp,
            "power_mean": power,
        }
    )


# --- split -------------------------------------------------------------------


def test_split_is_strictly_time_ordered():
    df = _signal_df(200)
    train, test = time_ordered_split(df, 0.7)
    t = pd.to_datetime(df["t_ref"]).values
    assert train.sum() > 0 and test.sum() > 0
    assert t[train].max() < t[test].min(), "every train point must precede every test point"
    assert train.sum() + test.sum() == len(df)


def test_split_handles_unsorted_input():
    # Set up the WRONG (shuffled) row order. The cut must still land at the 70th
    # time-percentile, not at a random row -> exactly 84/120 rows in train. This
    # pins the chronological argsort: replace it with positional order and t_cut
    # falls on an arbitrary row, breaking the fraction (the strict-ordering
    # invariant alone holds for any threshold, so it cannot catch that bug).
    df = _signal_df(120).sample(frac=1.0, random_state=3).reset_index(drop=True)
    train, test = time_ordered_split(df, 0.7)
    t = pd.to_datetime(df["t_ref"]).values
    assert t[train].max() < t[test].min()
    assert abs(int(train.sum()) - 84) <= 1, "cut must land at the 70th time-percentile"


# --- metrics -----------------------------------------------------------------


def test_metrics_present_and_signal_beats_baseline():
    ev = evaluate_candidate(CandidateSpec("logreg", ()), _signal_df(400))
    m = ev.metrics
    assert ev.primary_metric == "roc_auc"
    assert m["roc_auc"] is not None and m["avg_precision"] is not None
    # With real signal the held-out AUC must clear the permutation baseline.
    assert m["roc_auc"] > 0.75
    assert m["roc_auc"] > (m["roc_auc_permuted_baseline"] or 0.0) + 0.1
    # Lead-time table is broken out per horizon, with alert-budget P/R.
    assert set(m["per_horizon"]) == {"60", "300"}
    assert any(a["budget"] == 0.05 for a in m["alert_budget"])


def test_no_signal_dataset_does_not_beat_chance():
    # Vacuity guard: when the feature is noise, AUC must collapse toward 0.5.
    ev = evaluate_candidate(CandidateSpec("logreg", ()), _signal_df(400, signal=False))
    assert ev.metrics["roc_auc"] < 0.65


def test_leakage_probe_collapses_on_shuffled_labels():
    ev = evaluate_candidate(CandidateSpec("logreg", ()), _signal_df(400))
    probe = ev.metrics["leakage_probe"]
    assert probe["shuffled_label_auc"] is not None
    assert probe["leaks"] is False
    assert probe["shuffled_label_auc"] < 0.65


def test_evaluate_rejects_single_class_split():
    df = _signal_df(40)
    df["label"] = 0  # no positives -> cannot be judged
    with pytest.raises(ValueError):
        evaluate_candidate(CandidateSpec("logreg", ()), df)


# --- keep-if-better ----------------------------------------------------------


def test_first_candidate_promotes_then_weaker_rejected_stronger_accepted(tmp_path):
    reg = ModelRegistry(str(tmp_path))
    assert reg.has_incumbent() is False

    df = _signal_df(400)
    strong = CandidateSpec("hgb", ())
    weak = CandidateSpec("logreg", ("power_mean",))  # one weak feature

    # First candidate is the baseline -> always promotes to v1.
    ev1, r1 = run_round(strong, df, reg, dataset_path="synthetic.csv", dataset_sha256="x")
    assert r1.promoted is True and r1.version == 1
    v1_value = reg.incumbent.primary_value

    # A clearly weaker candidate must NOT be promoted; incumbent + version unchanged.
    weak_df = _signal_df(400, signal=False)  # noise -> ~chance
    ev2 = evaluate_candidate(weak, weak_df)
    # ensure the weak candidate really is worse than the incumbent
    assert ev2.primary_value < v1_value
    r2 = reg.consider(ev2, dataset_path="synthetic.csv", dataset_sha256="x")
    assert r2.promoted is False
    assert reg.incumbent.version == 1
    assert reg.incumbent.primary_value == v1_value
    assert len(reg.history) == 1


def test_better_candidate_advances_version(tmp_path):
    reg = ModelRegistry(str(tmp_path))
    df = _signal_df(400)
    # Seed a deliberately weak incumbent (single noisy feature) at v1...
    weak = evaluate_candidate(CandidateSpec("logreg", ("power_mean",)), df)
    reg.consider(weak, dataset_path="d.csv", dataset_sha256="x")
    assert reg.incumbent.version == 1
    # ...then a strictly stronger full-feature candidate must take v2.
    strong = evaluate_candidate(CandidateSpec("hgb", ()), df)
    assert strong.primary_value > reg.incumbent.primary_value
    r = reg.consider(strong, dataset_path="d.csv", dataset_sha256="x")
    assert r.promoted is True and reg.incumbent.version == 2
    assert [c.version for c in reg.history] == [1, 2]  # learning curve recorded


# --- persistence / restart ---------------------------------------------------


def test_promoted_model_survives_restart(tmp_path):
    df = _signal_df(400)
    reg = ModelRegistry(str(tmp_path))
    ev, r = run_round(CandidateSpec("hgb", ()), df, reg, dataset_path="d.csv", dataset_sha256="abc")
    assert r.promoted

    # Score the in-memory estimator on the held-out split for a reference value.
    train, test = time_ordered_split(df, 0.7)
    feats = list(ev.features)
    Xte = df.loc[test, feats].to_numpy(dtype="float64")
    before = ev.estimator.predict_proba(Xte)[:, 1]

    # Fresh process: a brand-new registry on the same dir restores the incumbent
    # and the persisted estimator must reproduce identical scores.
    reborn = ModelRegistry(str(tmp_path))
    assert reborn.has_incumbent() is True
    assert reborn.incumbent.version == 1
    assert reborn.incumbent.dataset_sha256 == "abc"
    assert reborn.incumbent.model_type == "hgb"
    assert reborn.incumbent.training_window[0] <= reborn.incumbent.training_window[1]
    loaded = reborn.load_estimator()
    after = loaded.predict_proba(Xte)[:, 1]
    np.testing.assert_allclose(after, before)


def test_empty_registry_reports_no_incumbent(tmp_path):
    reg = ModelRegistry(str(tmp_path / "fresh"))
    assert reg.has_incumbent() is False
    assert reg.incumbent is None
    assert "no persisted incumbent" in reg.describe_incumbent()
    with pytest.raises(ValueError):
        reg.load_estimator()


def test_sha256_file_is_content_addressed(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("label,x\n1,2\n")
    b.write_text("label,x\n1,2\n")
    assert sha256_file(str(a)) == sha256_file(str(b))
    b.write_text("label,x\n0,9\n")
    assert sha256_file(str(a)) != sha256_file(str(b))
