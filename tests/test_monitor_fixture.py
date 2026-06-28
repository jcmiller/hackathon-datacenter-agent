"""Tests for the committed /api/monitor demo fixture (bead jds).

Load-bearing properties: (1) the committed artifacts actually ship and load as a
*usable* incumbent off the droplet; (2) the synthetic table promotes through the
unmodified keep-if-better judge as honest weak signal (not leakage); (3) the
fixture report is honestly labeled; (4) the registry self-heals (rebuilds from the
committed table) if the pickled estimator cannot be unpickled.
"""

import shutil

import pandas as pd
import pytest

from gpusitter.app import monitor_fixture as mf
from gpusitter.detection import monitor
from gpusitter.detection.harness import (
    CandidateSpec,
    ModelRegistry,
    evaluate_candidate,
    load_dataset,
)


def test_committed_artifacts_exist():
    # The whole point of the bead: these files are tracked, not droplet-only.
    assert mf.FIXTURE_DIR.exists(), (
        "committed fixture dir missing — run scripts/build_monitor_fixture.py"
    )
    assert (mf.FIXTURE_DIR / "features.csv").exists()
    reg_dir = mf.FIXTURE_DIR / "registry"
    assert (reg_dir / "manifest.json").exists()
    assert (reg_dir / "v001.json").exists()
    assert (reg_dir / "v001.pkl").exists()


def test_committed_registry_loads_a_usable_incumbent():
    reg = ModelRegistry(mf.FIXTURE_REGISTRY_PATH)
    assert reg.incumbent is not None, "committed registry must carry a promoted incumbent"
    scorer = monitor.RowScorer.from_registry(reg)  # unpickles the estimator
    df = load_dataset(mf.FIXTURE_DATA_PATH)
    scores = scorer.score(df)
    assert len(scores) == len(df)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_committed_features_match_real_schema():
    df = load_dataset(mf.FIXTURE_DATA_PATH)
    # Real lys/r7j meta schema + real-named feature columns.
    for col in ("gpu", "t_ref", "horizon_s", "label"):
        assert col in df.columns
    for feat in mf.FIXTURE_FEATURES:
        assert feat in df.columns
    assert set(df["label"].unique()) == {0, 1}


def test_fixture_frame_promotes_as_honest_weak_signal():
    # Built through the immutable judge: promotes (real signal) but does NOT leak,
    # and the held-out AUC is a weak demo signal, not a suspicious ~1.0.
    df = mf.build_fixture_frame()
    ev = evaluate_candidate(CandidateSpec("logreg", mf.FIXTURE_FEATURES), df)
    assert ev.leaks is False, ev.metrics["leakage_probe"]
    assert 0.6 < ev.primary_value < 0.95, f"AUC {ev.primary_value} not an honest weak-signal demo"


def test_build_fixture_is_deterministic(tmp_path):
    a = mf.build_fixture(tmp_path / "a")
    b = mf.build_fixture(tmp_path / "b")
    fa = pd.read_csv(tmp_path / "a" / "features.csv")
    fb = pd.read_csv(tmp_path / "b" / "features.csv")
    pd.testing.assert_frame_equal(fa, fb)
    assert a["roc_auc"] == b["roc_auc"]
    assert a["promoted_version"] == b["promoted_version"] == 1


def test_load_fixture_registry_rebuilds_on_unloadable_pickle(tmp_path):
    # Copy the committed fixture, then corrupt the pickle. The loader must rebuild
    # a usable incumbent from the committed CSV rather than going dark.
    shutil.copytree(mf.FIXTURE_DIR, tmp_path / "fx")
    (tmp_path / "fx" / "registry" / "v001.pkl").write_bytes(b"not a pickle")

    reg = mf.load_fixture_registry(
        data_path=str(tmp_path / "fx" / "features.csv"),
        registry_path=str(tmp_path / "fx" / "registry"),
    )
    scorer = monitor.RowScorer.from_registry(reg)  # must not raise
    assert scorer.model_version >= 1


def test_load_fixture_registry_raises_without_source_table(tmp_path):
    with pytest.raises(FileNotFoundError):
        mf.load_fixture_registry(
            data_path=str(tmp_path / "gone.csv"),
            registry_path=str(tmp_path / "empty_reg"),
        )
