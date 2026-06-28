"""End-to-end CLI test for the eval-harness command (bead glf).

Drives the same path a self-improving agent would: a first candidate is promoted
to v1 and persisted; a second, weaker candidate run against the *persisted*
registry (fresh process) is rejected and the incumbent is unchanged. Proves the
promote/persist/restart/keep-if-better contract through the actual entrypoint.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scripts.eval_harness import main as cli_main

from gpusitter.detection.harness import ModelRegistry

TZ = timezone(timedelta(hours=8))
BASE = datetime(2023, 8, 15, tzinfo=TZ)


def _write_csv(path, *, signal: bool, seed: int, n: int = 400) -> None:
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n // 2)
    if signal:
        temp = np.where(labels == 1, rng.normal(75, 8, n), rng.normal(65, 8, n))
    else:
        temp = rng.normal(70, 8, n)
    pd.DataFrame(
        {
            "gpu": [f"node{i % 5}#0" for i in range(n)],
            "t_ref": [(BASE + timedelta(seconds=30 * i)).isoformat() for i in range(n)],
            "horizon_s": np.tile([60.0, 300.0], n // 2),
            "label": labels,
            "temp_last": temp,
        }
    ).to_csv(path, index=False)


def test_cli_promotes_then_rejects_weaker(tmp_path, capsys):
    reg_dir = str(tmp_path / "registry")
    strong = str(tmp_path / "strong.csv")
    weak = str(tmp_path / "weak.csv")
    _write_csv(strong, signal=True, seed=1)
    _write_csv(weak, signal=False, seed=2)

    # Round 1: real signal -> promoted to v1.
    assert cli_main(["--registry", reg_dir, "--data", strong, "--model", "logreg"]) == 0
    out = capsys.readouterr().out
    assert '"promoted": true' in out and '"version": 1' in out

    # Round 2 (fresh ModelRegistry built from disk inside main): noise candidate
    # must lose to the persisted incumbent.
    assert cli_main(["--registry", reg_dir, "--data", weak, "--model", "logreg"]) == 0
    out = capsys.readouterr().out
    assert '"promoted": false' in out

    # Incumbent on disk is still v1 and loads as a usable model.
    reg = ModelRegistry(reg_dir)
    assert reg.incumbent.version == 1
    assert reg.load_estimator().predict_proba is not None


def test_cli_show_reports_empty_registry(tmp_path, capsys):
    assert cli_main(["--registry", str(tmp_path / "none"), "--show"]) == 0
    assert "no persisted incumbent" in capsys.readouterr().out
