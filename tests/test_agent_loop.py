"""Tests for the agent-authored classifier loop (bead rnh).

The loop is the deliverable, so these tests pin the load-bearing properties of
write -> eval -> reflect -> revise:

* reflection is *real* — its signal_gap separates a planted-signal table from a
  no-signal one (not vacuous);
* revision *uses* the reflection — after the baseline the proposer steers toward
  the high-signal feature and drops noise, and never repeats a candidate;
* keep-if-better integrity survives at the loop level — a weaker candidate after a
  stronger incumbent is refused;
* the loop is honest — on pure noise it manufactures no signal and says so.

Each test sets up the contrasting / wrong state and asserts the corrected state,
so deleting the code under test fails the test. ``logreg`` is used as the model
form throughout (the permutation baseline refits per round; logreg keeps the suite
fast while exercising the identical loop path as hgb).
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from gpusitter.detection.agent_loop import (
    Attempt,
    ReflectiveProposer,
    reflect,
    run_loop,
)
from gpusitter.detection.harness import (
    CandidateSpec,
    ModelRegistry,
    feature_columns,
    run_round,
)

TZ = timezone(timedelta(hours=8))  # Kalos fixed +08:00
BASE = datetime(2023, 8, 15, 0, 0, 0, tzinfo=TZ)
LOGREG = ("logreg",)


def _loop_df(n: int = 400, *, signal: bool = True, seed: int = 0) -> pd.DataFrame:
    """Early-detection table with a strong feature, a weak one, and two noise ones.

    ``temp_last`` carries a strong warning signal (positives clearly hotter),
    ``power_mean`` a weak one, ``noise_a``/``noise_b`` none. With ``signal=False``
    every feature is noise. Labels are balanced but randomly positioned (a periodic
    label sequence would inflate the shuffled-label baseline). The contrast between
    these columns is what the reflective proposer must discover and exploit.
    """
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n // 2)
    rng.shuffle(labels)
    horizons = np.tile([60.0, 300.0], n // 2)
    if signal:
        temp = np.where(labels == 1, rng.normal(78, 7, n), rng.normal(62, 7, n))
        power = np.where(labels == 1, rng.normal(356, 27, n), rng.normal(344, 27, n))
    else:
        temp = rng.normal(70, 7, n)
        power = rng.normal(350, 27, n)
    return pd.DataFrame(
        {
            "gpu": [f"node{i % 7}#0" for i in range(n)],
            "t_ref": [(BASE + timedelta(seconds=30 * i)).isoformat() for i in range(n)],
            "horizon_s": horizons,
            "label": labels,
            "temp_last": temp,
            "power_mean": power,
            "noise_a": rng.normal(0, 1, n),
            "noise_b": rng.normal(5, 2, n),
        }
    )


def _registry(tmp_path, name: str) -> ModelRegistry:
    return ModelRegistry(str(tmp_path / name))


def _data_path(tmp_path, df: pd.DataFrame, name: str) -> str:
    path = str(tmp_path / name)
    df.to_csv(path, index=False)
    return path


def _proposer(df: pd.DataFrame) -> ReflectiveProposer:
    return ReflectiveProposer(tuple(feature_columns(df)), model_types=LOGREG)


# --- the loop runs and builds a real learning curve --------------------------


def test_loop_runs_explores_and_keeps_a_monotone_learning_curve(tmp_path):
    df = _loop_df(seed=0)
    reg = _registry(tmp_path, "reg")
    data = _data_path(tmp_path, df, "d.csv")

    res = run_loop(df, reg, dataset_path=data, proposer=_proposer(df), max_rounds=6)

    # The agent explores more than the single baseline candidate (revision happened)
    # and emits a hypothesis every round (the demo narrative).
    assert len(res.history) >= 3
    assert all(a.hypothesis.strip() for a in res.history)

    # keep-if-better => the promoted primary values are strictly increasing. An empty
    # curve (no promotion) or any flat/declining step would fail.
    vals = [v for _, v in res.learning_curve]
    assert vals, "loop promoted nothing"
    steps = list(zip(vals, vals[1:], strict=False))
    assert all(b > a for a, b in steps), f"not strictly increasing: {vals}"
    assert res.incumbent is not None
    assert res.incumbent.primary_value > 0.5, "real signal should beat chance"


# --- reflection is real, not vacuous -----------------------------------------


def test_reflection_signal_gap_separates_signal_from_noise(tmp_path):
    def gap(*, signal: bool) -> float:
        df = _loop_df(signal=signal, seed=1)
        tag = "s" if signal else "n"
        reg = _registry(tmp_path, f"reg_{tag}")
        data = _data_path(tmp_path, df, f"{tag}.csv")
        spec = CandidateSpec("logreg", tuple(feature_columns(df)))
        ev, promo = run_round(spec, df, reg, dataset_path=data)
        return reflect(0, ev, promo).signal_gap

    signal_gap = gap(signal=True)
    noise_gap = gap(signal=False)

    assert signal_gap is not None and noise_gap is not None
    assert signal_gap > 0.2, f"planted signal should lift AUC over baseline: {signal_gap}"
    assert abs(noise_gap) < 0.1, f"no-signal table must read near-chance: {noise_gap}"
    assert signal_gap > noise_gap + 0.15, "reflection must distinguish the two regimes"


# --- revision uses the reflection --------------------------------------------


def test_proposer_revises_toward_high_signal_features_after_baseline(tmp_path):
    df = _loop_df(seed=2)
    reg = _registry(tmp_path, "reg")
    data = _data_path(tmp_path, df, "d.csv")
    prop = _proposer(df)
    pool = set(feature_columns(df))

    # Round 1: with no history the only thing to try is the full-set baseline.
    spec1, hyp1 = prop.propose(df, [])
    assert set(spec1.features) == pool
    ev, promo = run_round(spec1, df, reg, dataset_path=data)
    history = [Attempt(0, spec1, hyp1, ev, promo, reflect(0, ev, promo))]

    # Round 2: now reflection-driven. The strong feature is kept, pure-noise columns
    # are dropped, and the candidate is a genuine subset (not the full set again).
    spec2, _ = prop.propose(df, history)
    feats2 = set(spec2.features)
    assert "temp_last" in feats2, "must keep the strongest signal"
    assert feats2 != pool, "must actually revise, not re-propose the full set"
    dropped = pool - feats2
    assert dropped, "should drop something"
    assert dropped <= {"power_mean", "noise_a", "noise_b"}, f"must not drop signal: {dropped}"
    assert "noise_a" in dropped or "noise_b" in dropped, "should drop at least one noise feature"


def test_loop_never_proposes_the_same_candidate_twice(tmp_path):
    df = _loop_df(seed=3)
    reg = _registry(tmp_path, "reg")
    data = _data_path(tmp_path, df, "d.csv")

    res = run_loop(df, reg, dataset_path=data, proposer=_proposer(df), max_rounds=8)

    keys = [(a.spec.model_type, frozenset(a.spec.features)) for a in res.history]
    assert len(keys) == len(set(keys)), f"duplicate candidate proposed: {keys}"


# --- keep-if-better integrity holds at the loop level ------------------------


class _ScriptedProposer:
    """A proposer that emits a fixed plan — used to force a strong-then-weak order."""

    def __init__(self, plan: list[tuple[CandidateSpec, str]]):
        self._plan = plan

    def propose(self, df, history):
        i = len(history)
        return self._plan[i] if i < len(self._plan) else None


def test_loop_refuses_a_regression_after_a_stronger_incumbent(tmp_path):
    df = _loop_df(seed=4)
    reg = _registry(tmp_path, "reg")
    data = _data_path(tmp_path, df, "d.csv")

    strong = (CandidateSpec("logreg", ("temp_last", "power_mean")), "strong: real signal")
    weak = (CandidateSpec("logreg", ("noise_a",)), "weak: noise only")
    res = run_loop(
        df, reg, dataset_path=data, proposer=_ScriptedProposer([strong, weak]), max_rounds=4
    )

    assert res.history[0].promotion.promoted, "strong candidate should establish v1"
    assert not res.history[1].promotion.promoted, "weaker candidate must lose keep-if-better"
    assert res.incumbent.version == 1, "incumbent must be unchanged by the regression"
    assert set(res.incumbent.features) == {"temp_last", "power_mean"}


# --- honesty on no-signal data -----------------------------------------------


def test_loop_is_honest_on_noise(tmp_path):
    df = _loop_df(signal=False, seed=5)
    reg = _registry(tmp_path, "reg")
    data = _data_path(tmp_path, df, "d.csv")

    res = run_loop(df, reg, dataset_path=data, proposer=_proposer(df), max_rounds=6)

    gaps = [a.reflection.signal_gap for a in res.history if a.reflection.signal_gap is not None]
    assert gaps, "expected scorable rounds"
    assert max(gaps) < 0.12, f"loop must not manufacture signal on noise: {max(gaps)}"
    assert any("near-chance" in a.reflection.notes for a in res.history), "must flag weak signal"


# --- un-scorable candidates are recorded, never promoted ---------------------


def test_unscorable_round_is_recorded_not_promoted(tmp_path):
    # Build a table whose latest (held-out) rows are all positive -> the time split
    # yields a single-class test set the harness refuses to score.
    n = 200
    cut = int(n * 0.7)
    labels = np.zeros(n, dtype=int)
    labels[:cut] = np.tile([0, 1], cut // 2)  # mixed in the (earlier) train region
    labels[cut:] = 1  # all-positive in the (later) test region
    rng = np.random.default_rng(6)
    df = pd.DataFrame(
        {
            "gpu": [f"node{i % 5}#0" for i in range(n)],
            "t_ref": [(BASE + timedelta(seconds=30 * i)).isoformat() for i in range(n)],
            "horizon_s": np.tile([60.0, 300.0], n // 2),
            "label": labels,
            "temp_last": rng.normal(70, 8, n),
        }
    )
    reg = _registry(tmp_path, "reg")
    data = _data_path(tmp_path, df, "d.csv")

    res = run_loop(df, reg, dataset_path=data, proposer=_proposer(df), max_rounds=3)

    first = res.history[0]
    assert first.evaluation is None and first.promotion is None
    assert "could not be evaluated" in first.reflection.notes
    assert res.incumbent is None, "an unscorable candidate must never be promoted"


# --- CLI demo path -----------------------------------------------------------


def test_cli_runs_the_loop_and_reports_a_learning_curve(tmp_path, capsys):
    import json

    from scripts.agent_loop import main as cli_main

    df = _loop_df(seed=7)
    data = _data_path(tmp_path, df, "early.csv")
    reg_dir = str(tmp_path / "registry")

    rc = cli_main(
        [
            "--registry",
            reg_dir,
            "--data",
            data,
            "--model-types",
            "logreg",
            "--max-rounds",
            "4",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["rounds"] >= 1
    assert out["promotions"] >= 1
    assert out["learning_curve"], "demo must surface the promotion curve"
    assert all(t["hypothesis"] for t in out["transcript"])

    # A fresh registry built from disk restores the promoted incumbent (restart).
    assert ModelRegistry(reg_dir).has_incumbent()
