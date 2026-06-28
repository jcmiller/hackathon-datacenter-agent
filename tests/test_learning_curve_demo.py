"""Tests for the learning-curve demo artifact (bead 8co).

Pins the properties that make the artifact a *demonstration* and an *honest* one:

* the synthetic table is deliberately weak (mirrors Kalos, not a staged 0.95);
* the real-data reference is read from the committed eval json, not fabricated;
* the curve climbs above the v0 floor and is monotone (keep-if-better), and at
  least one worse candidate is shown rejected (the gate is real);
* the CLI writes a well-formed artifact.

Each test sets up the contrasting state, so deleting the code under test fails.
"""

import json

from scripts.learning_curve_demo import (
    DEFAULT_EVAL_REF,
    build_artifact,
    main,
    real_data_reference,
    synthetic_weak_signal,
)

from gpusitter.detection.agent_loop import ReflectiveProposer
from gpusitter.detection.harness import (
    CandidateSpec,
    ModelRegistry,
    feature_columns,
    run_round,
)


def test_synthetic_table_carries_only_weak_signal():
    df = synthetic_weak_signal(n=1200, seed=0)
    assert {"temp_mean", "power_mean", "util_mean", "mem_last"} <= set(df.columns)
    assert 0.12 < df["label"].mean() < 0.24, "base rate should mirror Kalos (~0.18)"

    # A real model lands in the honest weak band — not saturated. If the generator
    # were cranked to an easy signal this upper bound would fail.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        reg = ModelRegistry(f"{tmp}/reg")
        data = f"{tmp}/d.csv"
        df.to_csv(data, index=False)
        ev, _ = run_round(CandidateSpec("logreg", ()), df, reg, dataset_path=data)
    assert 0.55 < ev.primary_value < 0.85, f"signal should be weak-but-real, got {ev.primary_value}"


def test_real_data_reference_is_read_from_the_eval_json():
    ref = real_data_reference(DEFAULT_EVAL_REF)
    assert ref is not None, "the committed eval json should be present"
    assert ref["best_real"]["model"] == "logreg"
    # The real verdict: weak (~0.65) and logreg beats hgb. Both come from the file.
    assert 0.6 < ref["best_real"]["roc_auc"] < 0.72
    assert ref["best_hgb_roc_auc"] < ref["best_real"]["roc_auc"]


def test_real_data_reference_absent_file_returns_none(tmp_path):
    assert real_data_reference(str(tmp_path / "nope.json")) is None


def test_build_artifact_curve_climbs_above_floor_and_is_monotone():
    df = synthetic_weak_signal(n=900, seed=0)
    art = build_artifact(
        df,
        source="synthetic-weak-signal",
        synthetic=True,
        train_frac=0.70,
        primary_metric="roc_auc",
        eval_ref_path=DEFAULT_EVAL_REF,
        proposer=ReflectiveProposer(tuple(feature_columns(df)), model_types=("logreg",)),
    )

    curve = art["curve"]
    assert curve[0]["version"] == "v0"
    assert abs(curve[0]["roc_auc"] - 0.5) < 0.02, "curve must start at the no-skill floor"

    aucs = [p["roc_auc"] for p in curve]
    steps = list(zip(aucs, aucs[1:], strict=False))
    assert all(b >= a for a, b in steps), f"curve must not decline (keep-if-better): {aucs}"
    assert art["final_incumbent"]["roc_auc"] > curve[0]["roc_auc"] + 0.05, "must climb above floor"

    # The gate is real: at least one candidate was rejected.
    assert any(not r["promoted"] for r in art["rounds"]), "expected a rejected candidate"
    assert art["real_data_reference"] is not None
    assert art["dataset"]["synthetic"] is True


def test_cli_writes_a_wellformed_artifact(tmp_path):
    # Drive the CLI on a small real-path dataset (avoids the large default synthetic).
    df = synthetic_weak_signal(n=400, seed=0)
    data = str(tmp_path / "early.csv")
    df.to_csv(data, index=False)
    out = str(tmp_path / "curve.json")

    rc = main(["--data", data, "--out", out, "--quiet"])
    assert rc == 0

    art = json.loads(open(out).read())
    for key in ("curve", "rounds", "baseline_v0", "final_incumbent", "real_data_reference"):
        assert key in art, f"artifact missing {key}"
    assert art["dataset"]["synthetic"] is False
    assert art["curve"][0]["version"] == "v0"
