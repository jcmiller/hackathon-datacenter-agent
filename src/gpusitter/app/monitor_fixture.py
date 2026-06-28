"""Committed honest demo fixture for ``/api/monitor`` (bead jds).

The operational monitor surface (bead i6k) reads two real droplet artifacts: the
labeled per-GPU feature table ``data/early_detection.parquet`` (gitignored) and a
persisted :class:`~gpusitter.detection.harness.ModelRegistry` at
``models/early_detection/`` (untracked). Both are absent off the droplet, so the
dashboard's miss-detector / monitor / learning-curve story cannot render in a
portable demo.

This module ships a tiny, *clearly illustrative* substitute so the surface works
without the ~80 GB trace:

* a deterministic synthetic labeled feature table that uses the REAL dataset's
  feature-name schema (``GPU_TEMP_*`` / ``POWER_USAGE_*`` / ``MEMORY_TEMP_*``), and
* a prebuilt registry (promoted logreg incumbent: model card + pickled estimator +
  manifest) produced through the unmodified harness ``run_round`` keep-if-better
  path — the immutable judge is used, never bypassed.

**Honesty.** The numbers here are synthetic and exist only to exercise the monitor
report's *shape* (per-row scores, alert budgets, the horizon-grid miss detector).
They are NOT a claim about Kalos. The real held-out evaluation — weak-but-real
linear signal, held-out ROC-AUC ~0.64–0.65, NO-GO verdict — stays canonical in
``docs/early-detection-eval.md``. Every surface that serves the fixture labels it.
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..detection.harness import CandidateSpec, ModelRegistry, run_round
from ..detection.monitor import RowScorer

# Committed artifacts live OUTSIDE the gitignored data/ and models/ trees.
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "early_detection"
FIXTURE_DATA_PATH = str(FIXTURE_DIR / "features.csv")
FIXTURE_REGISTRY_PATH = str(FIXTURE_DIR / "registry")

# Real-schema feature names (a compact subset of the live 40-feature table). The
# learnable signal sits in GPU_TEMP_*; the rest are honest noise. Kept under the
# harness's ~0.99 single-feature leakage trip so the candidate promotes as signal.
FIXTURE_FEATURES: tuple[str, ...] = (
    "GPU_TEMP_last",
    "GPU_TEMP_mean",
    "GPU_TEMP_slope",
    "POWER_USAGE_last",
    "MEMORY_TEMP_last",
)

# Illustrative-only label surfaced in the API payload and the fixture README.
FIXTURE_NOTE = (
    "Illustrative synthetic demo fixture (bead jds) — numbers are NOT real Kalos "
    "results. The real held-out evaluation (ROC-AUC ~0.64–0.65, NO-GO) lives in "
    "docs/early-detection-eval.md."
)

_TZ = timezone(timedelta(hours=8))
_BASE = datetime(2023, 8, 15, 0, 0, 0, tzinfo=_TZ)
_HORIZONS_S = (60.0, 300.0, 600.0)
_NEG_OFFSET_S = 3600.0


def build_fixture_frame(*, n_onsets: int = 36, n_decoys: int = 60, seed: int = 0):
    """Deterministic synthetic labeled early-detection table (lys/r7j schema).

    Mirrors the real builder's row taxonomy so the monitor report is exercised
    honestly:

    * **Positives** — each onset lives on its own GPU; for every horizon ``H`` a
      positive sits at ``t_event - H`` (so it falls inside that GPU's
      ``[t_event-H, t_event)`` alert window) and runs ~1.2σ hotter on GPU_TEMP.
    * **Same-GPU pre-event controls** — a cool negative at ``t_event - 3600`` per
      horizon (a true negative: no onset in its horizon).
    * **Decoy GPUs** — onset-free GPUs carrying cool negatives scattered across the
      same span, so a skilled scorer must spend its budget on the right rows.

    The GPU_TEMP separation is learnable but sub-leakage (univariate AUC ~0.8),
    so the logreg candidate promotes as genuine signal, not a label echo.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)

    def _row(gpu, t_ref, horizon_s, label, hot):
        # ~1.1σ GPU_TEMP separation -> univariate/held-out AUC ~0.75 (weak-but-real,
        # the honest demo target). slope/power/memory carry NO label signal, so the
        # model leans on a single weak feature exactly like the real evaluation.
        temp_mu = 73.0 if hot else 64.0
        temp_last = float(rng.normal(temp_mu, 8.0))
        return {
            "gpu": gpu,
            "node": gpu.split("#")[0],
            "gpu_idx": int(gpu.split("#")[1]),
            "t_ref": t_ref.isoformat(),
            "event_source": "XID_ERRORS",
            "horizon_s": horizon_s,
            "lookback_s": 600.0,
            "label": label,
            "GPU_TEMP_last": temp_last,
            "GPU_TEMP_mean": temp_last - float(rng.normal(1.0, 1.5)),
            "GPU_TEMP_slope": float(rng.normal(0.0, 0.01)),
            "POWER_USAGE_last": float(rng.normal(250.0, 30.0)),
            "MEMORY_TEMP_last": float(rng.normal(70.0, 8.0)),
        }

    # Spread onsets randomly across the window (not monotonically) so the harness's
    # strict time-ordered split gets a representative test set — a sorted timeline
    # would put a near-pure subset in the holdout and inflate AUC toward leakage.
    span_h = float(n_onsets)
    onset_hours = sorted(float(x) for x in rng.uniform(0.0, span_h, size=n_onsets))

    rows = []
    for i, oh in enumerate(onset_hours):
        onset = _BASE + timedelta(hours=oh)
        gpu = f"g{i}#0"
        for h in _HORIZONS_S:
            rows.append(_row(gpu, onset - timedelta(seconds=h), h, 1, hot=True))
            rows.append(_row(gpu, onset - timedelta(seconds=_NEG_OFFSET_S), h, 0, hot=False))
    for j in range(n_decoys):
        gpu = f"decoy{j % 12}#0"
        base_h = float(rng.uniform(0.0, span_h))
        for h in _HORIZONS_S:
            offset = int(rng.integers(-3000, 3000))
            t = _BASE + timedelta(hours=base_h, seconds=offset)
            rows.append(_row(gpu, t, h, 0, hot=False))

    df = pd.DataFrame(rows).sort_values(["t_ref", "gpu", "horizon_s"]).reset_index(drop=True)
    return df


def _write_readme(out_dir: Path) -> None:
    readme = out_dir / "README.md"
    readme.write_text(
        "# early_detection demo fixture (bead jds)\n\n"
        "**Illustrative / synthetic — NOT real Kalos numbers.**\n\n"
        "`features.csv` is a small deterministic synthetic labeled early-detection\n"
        "table (real-schema feature names) and `registry/` is a prebuilt\n"
        "`ModelRegistry` (promoted logreg incumbent: model card + pickled estimator\n"
        "+ manifest). They exist so the dashboard's `/api/monitor` surface renders\n"
        "off the droplet, without the ~80 GB trace or `data/early_detection.parquet`.\n\n"
        "Regenerate with `python scripts/build_monitor_fixture.py`.\n\n"
        "The real held-out evaluation (weak-but-real linear signal, ROC-AUC\n"
        "~0.64–0.65, NO-GO for a standalone predictor) is canonical in\n"
        "`docs/early-detection-eval.md`. Do not cite the fixture as a result.\n"
    )


def build_fixture(out_dir: str | Path = FIXTURE_DIR, *, seed: int = 0) -> dict:
    """Generate the committed demo artifacts: ``features.csv`` + prebuilt registry.

    Writes the synthetic table, then promotes a logreg incumbent through the
    unmodified harness ``run_round`` (immutable judge, keep-if-better). Returns a
    summary dict for the build CLI. Re-runnable and deterministic.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build_fixture_frame(seed=seed)

    data_path = out / "features.csv"
    df.to_csv(data_path, index=False)

    # Fresh single-version registry: a stale incumbent from a prior build carries a
    # different holdout id and keep-if-better would refuse the rebuild outright.
    reg_dir = out / "registry"
    if reg_dir.exists():
        shutil.rmtree(reg_dir)
    reg = ModelRegistry(str(reg_dir))
    ev, result = run_round(
        CandidateSpec("logreg", FIXTURE_FEATURES),
        df,
        reg,
        dataset_path=str(data_path),
        created_at=_BASE.isoformat(),
    )
    if not result.promoted:
        raise RuntimeError(f"fixture candidate failed to promote: {result.reason}")
    _write_readme(out)
    return {
        "data_path": str(data_path),
        "registry_path": str(reg_dir),
        "n_rows": int(len(df)),
        "n_positive": int(df["label"].sum()),
        "promoted_version": result.version,
        "roc_auc": ev.primary_value,
        "features": list(FIXTURE_FEATURES),
    }


def load_fixture_registry(
    data_path: str = FIXTURE_DATA_PATH,
    registry_path: str = FIXTURE_REGISTRY_PATH,
) -> ModelRegistry:
    """Return a registry whose incumbent estimator is loadable.

    Loads the committed registry; if it has no incumbent or the pickled estimator
    cannot be unpickled (e.g. a future scikit-learn drift breaks the demo pickle),
    rebuild deterministically from the committed ``features.csv`` into a temp cache
    dir so the demo never silently goes dark. Raises if even the source table is
    missing — there is then nothing honest to serve.
    """
    reg = ModelRegistry(registry_path)
    if reg.incumbent is not None:
        try:
            RowScorer.from_registry(reg)  # forces the unpickle
            return reg
        except Exception:
            pass  # committed pickle unusable -> rebuild below

    if not Path(data_path).exists():
        raise FileNotFoundError(f"fixture feature table missing at {data_path}")
    # Rebuild fresh from the committed table (no caching — avoids serving a stale
    # rebuild if the source table later changes). Only reached when the committed
    # pickle is unusable, so the one-off logreg fit is acceptable.
    import pandas as pd

    cache_dir = Path(tempfile.mkdtemp(prefix="gpusitter_monitor_fixture_")) / "registry"
    rebuilt = ModelRegistry(str(cache_dir))
    df = pd.read_csv(data_path)
    _, result = run_round(
        CandidateSpec("logreg", FIXTURE_FEATURES),
        df,
        rebuilt,
        dataset_path=data_path,
        created_at=_BASE.isoformat(),
    )
    if not result.promoted:
        raise RuntimeError(f"fixture rebuild failed to promote: {result.reason}")
    return rebuilt
