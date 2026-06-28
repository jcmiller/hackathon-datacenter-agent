"""Eval harness + filesystem-backed model registry (bead glf).

This is the *judge* the self-improving loop optimizes against. An agent (bead rnh)
authors only a :class:`CandidateSpec` — a model type plus a feature list. The
harness owns everything that decides whether a candidate is better than the
incumbent: the held-out split, the metrics, the leakage probes, and the
keep-if-better promotion. The candidate path cannot edit any of those, so the
gate is not theatre (Design: "separation of powers / immutable evaluator").

Why a new module instead of extending ``classifier.py``: that module is the live
triage agent's *in-process* incumbent (consumed by ``app/sim.py`` and
``agent/tools.py``) and only persists a metadata card — a model fitted there
cannot be reloaded and used after a restart. This harness is the offline
early-detection evaluator: it pickles every promoted estimator alongside a model
card so a process restart restores a *usable* incumbent, not just its metadata.

Integrity guards (Design):

* **Strict time-ordered split.** A single ``t_ref`` threshold with
  ``max(train.t_ref) < min(test.t_ref)`` — every training point is strictly in the
  past of every test point, so a candidate can never see future telemetry.
* **No-signal permutation baseline.** The identical model retrained on shuffled
  train labels and scored on the real held-out labels. Real signal is the *gap*
  above this control, not the absolute AUC.
* **Shuffle-label leakage probe.** On the real candidate's own train labels
  shuffled, held-out AUC must collapse to ~0.5; a probe AUC that stays high means
  a feature is leaking the label and the candidate is flagged before promotion is
  even considered.
* **Keep-if-better.** A candidate replaces the incumbent only if it beats it on
  the declared primary metric (default held-out ROC-AUC). Every promotion bumps
  the version and is persisted, giving the v1->v2->v3 learning curve.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Mirror the labeled-dataset schema (gpusitter.detection.early_dataset._META_COLS).
META_COLS = ("gpu", "node", "gpu_idx", "t_ref", "event_source", "horizon_s", "lookback_s", "label")
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_PRIMARY_METRIC = "roc_auc"
ALERT_BUDGETS = (0.01, 0.05, 0.10)  # fraction of held-out points we may alert on
N_PERMUTATIONS = 8  # label shuffles averaged for the no-signal baseline
MANIFEST_NAME = "manifest.json"


# --- candidate authored by the agent (it may set ONLY these) -----------------


class CandidateSpec(NamedTuple):
    """The only thing the agent (rnh) controls: a model type and its features."""

    model_type: str
    features: tuple[str, ...]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All non-meta columns — the full candidate feature set when none is named."""
    return [c for c in df.columns if c not in META_COLS]


def _build_model(model_type: str):
    """Construct a fresh estimator. NaN missingness is explicit in the features.

    ``logreg`` median-imputes + scales (it cannot consume NaN); ``hgb`` consumes
    NaN natively. Kept identical to the lys eval models so harness numbers and the
    standalone eval report agree.
    """
    if model_type == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
    if model_type == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
    raise ValueError(f"unknown model_type {model_type!r}; expected 'logreg' or 'hgb'")


# --- provenance --------------------------------------------------------------


def sha256_file(path: str) -> str:
    """Streaming SHA-256 of the source dataset, recorded on every promoted card."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- strict time-ordered split (the no-future-leakage guarantee) -------------


def time_ordered_split(
    df: pd.DataFrame, train_frac: float = DEFAULT_TRAIN_FRAC
) -> tuple[np.ndarray, np.ndarray]:
    """Strict time-ordered split on ``t_ref``: earliest ``train_frac`` -> train.

    A single ``t_ref`` threshold ``t_cut`` (the t_ref of the first test row in time
    order): **train = rows strictly before ``t_cut``, test = rows at/after it**, so
    ``max(train.t_ref) < t_cut <= min(test.t_ref)``. Unsorted input is handled (the
    order is computed, not assumed). Returns positional boolean masks aligned to
    ``df`` row order.
    """
    t = pd.to_datetime(df["t_ref"]).values
    order = np.argsort(t, kind="mergesort")  # stable, chronological
    n = len(df)
    k = max(1, int(round(train_frac * n)))
    t_cut = t[order[k]] if k < n else t[order[-1]]
    train_mask = t < t_cut
    return train_mask, ~train_mask


# --- metrics -----------------------------------------------------------------


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) >= 2 else None


def _safe_ap(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(average_precision_score(y, s)) if len(np.unique(y)) >= 2 else None


def _alert_budget_metrics(y_true: np.ndarray, scores: np.ndarray, budget: float) -> dict:
    """Precision/recall when alerting on the top ``budget`` fraction of scores."""
    n = len(scores)
    k = max(1, int(round(budget * n)))
    flagged = np.argsort(scores)[::-1][:k]
    tp = int(y_true[flagged].sum())
    total_pos = int(y_true.sum())
    return {
        "budget": budget,
        "k": k,
        "precision": tp / k if k else float("nan"),
        "recall": tp / total_pos if total_pos else float("nan"),
        "flagged_positive": tp,
    }


def _fit_score(model, Xtr, ytr, Xte) -> np.ndarray:
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def _per_horizon_table(
    df: pd.DataFrame, test_mask: np.ndarray, scores: np.ndarray
) -> dict[str, dict]:
    """Lead-time table: held-out metrics sliced by horizon.

    A positive sits exactly ``horizon_s`` seconds before its onset, so recall at
    horizon H *is* the fraction of real faults catchable with H seconds of lead at
    that alert budget. ``scores`` is aligned to the test rows (in ``df`` order).
    """
    if "horizon_s" not in df.columns:
        return {}
    yte_full = df["label"].to_numpy(dtype="int")
    horizon_full = df["horizon_s"].to_numpy()
    test_idx = np.flatnonzero(test_mask)
    table: dict[str, dict] = {}
    for h in sorted(np.unique(horizon_full[test_mask])):
        sel = test_idx[horizon_full[test_idx] == h]
        # positions of these rows within the scores array (scores is test-ordered)
        pos_in_test = np.searchsorted(test_idx, sel)
        y_h = yte_full[sel]
        s_h = scores[pos_in_test]
        table[str(int(h))] = {
            "n_test": int(len(sel)),
            "pos_test": int(y_h.sum()),
            "roc_auc": _safe_auc(y_h, s_h),
            "avg_precision": _safe_ap(y_h, s_h),
            "alert_budget": [_alert_budget_metrics(y_h, s_h, b) for b in ALERT_BUDGETS],
        }
    return table


# --- candidate evaluation (the immutable judge) ------------------------------

# Row-identity columns that fingerprint a held-out set. A prediction point is
# keyed by (gpu, t_ref, horizon); label is included so relabeling the holdout also
# changes its identity. Columns absent from a generic dataset are simply skipped.
_HOLDOUT_ID_COLS = ("gpu", "node", "gpu_idx", "t_ref", "horizon_s", "label")


def holdout_identity(df: pd.DataFrame, test_mask: np.ndarray) -> str:
    """Order-independent fingerprint of the exact held-out rows + labels.

    Keep-if-better is only meaningful when the incumbent and the candidate were
    scored on the *identical* holdout; this id is what the registry records and
    asserts on. It hashes the *set* of test-row identity tuples, so it is invariant
    to row order and to changes elsewhere in the dataset (e.g. the accreting
    training history) — it changes iff the holdout rows or their labels change.
    """
    cols = [c for c in _HOLDOUT_ID_COLS if c in df.columns] or list(df.columns)
    sub = df.loc[test_mask, cols]
    keyed = sorted("|".join(str(v) for v in row) for row in sub.itertuples(index=False, name=None))
    h = hashlib.sha256()
    h.update(("\x1f".join(cols) + "\x1e").encode())
    for k in keyed:
        h.update(k.encode())
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class Evaluation:
    """Everything the registry needs to score one candidate, plus the fitted model."""

    estimator: object
    model_type: str
    features: tuple[str, ...]
    primary_metric: str
    primary_value: float | None
    metrics: dict
    training_window: tuple[str, str]
    n_train: int
    n_test: int
    holdout_id: str = ""

    @property
    def leaks(self) -> bool:
        """True iff a leakage probe tripped — such a candidate must never promote."""
        return bool((self.metrics.get("leakage_probe") or {}).get("leaks"))


def evaluate_candidate(
    spec: CandidateSpec,
    df: pd.DataFrame,
    *,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
    rng: np.random.Generator | None = None,
) -> Evaluation:
    """Fit ``spec`` on the train split and score it on the strict held-out split.

    The candidate provides only ``spec``; the split, metrics, baseline, and probe
    are fixed here. ``primary_value`` is the scalar keep-if-better compares on
    (default pooled held-out ROC-AUC). Raises ``ValueError`` if the split is too
    small or single-class to score honestly — a candidate that cannot be judged is
    never silently promoted.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    feats = list(spec.features) if spec.features else feature_columns(df)
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"candidate features absent from dataset: {missing}")

    train_mask, test_mask = time_ordered_split(df, train_frac)
    Xtr = df.loc[train_mask, feats].to_numpy(dtype="float64")
    Xte = df.loc[test_mask, feats].to_numpy(dtype="float64")
    ytr = df.loc[train_mask, "label"].to_numpy(dtype="int")
    yte = df.loc[test_mask, "label"].to_numpy(dtype="int")

    if len(ytr) < 10 or len(yte) < 5 or ytr.sum() == 0 or len(np.unique(yte)) < 2:
        raise ValueError(
            "insufficient size/class balance on time-ordered split "
            f"(n_train={len(ytr)}, n_test={len(yte)}, pos_train={int(ytr.sum())})"
        )

    estimator = _build_model(spec.model_type)
    scores = _fit_score(estimator, Xtr, ytr, Xte)

    # No-signal permutation baseline: identical model on shuffled train labels.
    perm_aucs: list[float] = []
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(ytr)
        try:
            a = _safe_auc(yte, _fit_score(_build_model(spec.model_type), Xtr, perm, Xte))
            if a is not None:
                perm_aucs.append(a)
        except Exception:
            pass
    permuted_baseline = float(np.mean(perm_aucs)) if perm_aucs else None

    metrics = {
        "roc_auc": _safe_auc(yte, scores),
        "avg_precision": _safe_ap(yte, scores),
        "roc_auc_permuted_baseline": permuted_baseline,
        "roc_auc_permuted_n": len(perm_aucs),
        "base_rate": float(yte.mean()),
        "alert_budget": [_alert_budget_metrics(yte, scores, b) for b in ALERT_BUDGETS],
        "per_horizon": _per_horizon_table(df, test_mask, scores),
        "leakage_probe": _leakage_probe(permuted_baseline, feats, Xte, yte),
    }
    if primary_metric not in metrics or not isinstance(metrics[primary_metric], (int, float)):
        raise ValueError(f"primary_metric {primary_metric!r} is not a scalar held-out metric")

    t_train = pd.to_datetime(df.loc[train_mask, "t_ref"])
    window = (t_train.min().isoformat(), t_train.max().isoformat())
    return Evaluation(
        estimator=estimator,
        model_type=spec.model_type,
        features=tuple(feats),
        primary_metric=primary_metric,
        primary_value=metrics[primary_metric],
        metrics=metrics,
        training_window=window,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        holdout_id=holdout_identity(df, test_mask),
    )


def _leakage_probe(
    permuted_baseline: float | None,
    feats: Sequence[str],
    Xte: np.ndarray,
    yte: np.ndarray,
) -> dict:
    """Two cheap leakage guards run before promotion is ever considered (Design #5).

    * **shuffle-label**: held-out AUC of the model retrained on *shuffled* train
      labels (the averaged permutation baseline) must collapse to ~0.5. If it stays
      high the eval pipeline itself is leaking (e.g. test order tracks the label).
    * **single-feature outlier**: the strongest *univariate* held-out AUC of any one
      raw feature. A value ~1.0 means a single feature essentially *is* the label —
      target leakage in the dataset, not learned signal.

    ``leaks`` is True if either guard trips, marking a candidate not to be trusted.
    """
    per_feature = {}
    worst = 0.5
    for j, name in enumerate(feats):
        col = Xte[:, j]
        if np.all(np.isnan(col)) or len(np.unique(yte)) < 2:
            continue
        a = _safe_auc(yte, np.nan_to_num(col, nan=float(np.nanmedian(col))))
        if a is None:
            continue
        per_feature[name] = a
        worst = max(worst, abs(a - 0.5) + 0.5)
    shuffle_high = permuted_baseline is not None and permuted_baseline > 0.65
    feature_leak = worst > 0.99
    return {
        "shuffled_label_auc": permuted_baseline,
        "max_single_feature_auc": worst,
        "single_feature_auc": per_feature,
        "leaks": bool(shuffle_high or feature_leak),
    }


# --- model card + registry (versioning, persistence, restart) ----------------


@dataclass
class ModelCard:
    """The persisted record of a promoted model (AC #4)."""

    version: int
    model_type: str
    features: list[str]
    primary_metric: str
    primary_value: float
    metrics: dict
    training_window: list[str]
    n_train: int
    n_test: int
    dataset_path: str
    dataset_sha256: str
    holdout_id: str = ""
    created_at: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> ModelCard:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PromotionResult:
    """Outcome of one keep-if-better decision."""

    promoted: bool
    reason: str
    candidate_value: float | None
    incumbent_value: float | None
    version: int | None
    card: ModelCard | None = None


@dataclass
class ModelRegistry:
    """Filesystem-backed registry: keep-if-better promotion + version history.

    Layout under ``root``::

        manifest.json   {"incumbent_version": N, "history": [card, ...]}
        v001.pkl        pickled estimator for version 1
        v001.json       its model card (human-readable sidecar)
        ...

    Constructing a registry on an existing ``root`` restores the incumbent and the
    full history (restart restore, AC #5). On an empty/absent ``root`` the
    incumbent is ``None`` and that is reported plainly.
    """

    root: str
    incumbent: ModelCard | None = field(default=None, init=False)
    history: list[ModelCard] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._load()

    # -- persistence ----------------------------------------------------------

    @property
    def _manifest_path(self) -> str:
        return os.path.join(self.root, MANIFEST_NAME)

    def _estimator_path(self, version: int) -> str:
        return os.path.join(self.root, f"v{version:03d}.pkl")

    def _card_path(self, version: int) -> str:
        return os.path.join(self.root, f"v{version:03d}.json")

    def _load(self) -> None:
        self.incumbent = None
        self.history = []
        if not os.path.exists(self._manifest_path):
            return
        with open(self._manifest_path) as f:
            manifest = json.load(f)
        self.history = [ModelCard.from_dict(c) for c in manifest.get("history", [])]
        inc_v = manifest.get("incumbent_version")
        if inc_v is not None:
            self.incumbent = next((c for c in self.history if c.version == inc_v), None)

    def _write_manifest(self) -> None:
        manifest = {
            "incumbent_version": self.incumbent.version if self.incumbent else None,
            "primary_metric": self.incumbent.primary_metric if self.incumbent else None,
            "history": [asdict(c) for c in self.history],
        }
        tmp = self._manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, self._manifest_path)  # atomic swap

    def has_incumbent(self) -> bool:
        return self.incumbent is not None

    def describe_incumbent(self) -> str:
        """Plain restart report (AC #5)."""
        if self.incumbent is None:
            return "no persisted incumbent in registry"
        c = self.incumbent
        return (
            f"incumbent v{c.version} ({c.model_type}): "
            f"{c.primary_metric}={c.primary_value:.4f} over {len(self.history)} version(s)"
        )

    def load_estimator(self, version: int | None = None):
        """Unpickle a promoted estimator (default: the incumbent). Proves a
        promoted model survives restart as a *usable* model, not just metadata."""
        if version is None:
            if self.incumbent is None:
                raise ValueError("no incumbent to load")
            version = self.incumbent.version
        with open(self._estimator_path(version), "rb") as f:
            return pickle.load(f)

    # -- promotion ------------------------------------------------------------

    def consider(
        self,
        ev: Evaluation,
        *,
        dataset_path: str,
        dataset_sha256: str | None = None,
        created_at: str | None = None,
    ) -> PromotionResult:
        """Keep-if-better: promote ``ev`` only if it beats the incumbent's primary value.

        A candidate is promoted only when it clears every integrity guard:

        1. **No leakage.** If a leakage probe tripped, the candidate is refused —
           even as the baseline. A leaking model would poison the incumbent and
           every comparison made against it thereafter.
        2. **Same primary metric** as the incumbent.
        3. **Same holdout.** keep-if-better is meaningful only when both models were
           scored on the *identical* held-out set; comparing AUCs across different
           holdouts is apples-to-oranges and could promote a worse model, so a
           holdout-id mismatch is refused outright.
        4. **Strictly better** primary value than the incumbent.

        The first candidate is the baseline (guards 2-4 vacuous), but still subject
        to the leakage guard. A refused candidate writes nothing: the incumbent and
        version are unchanged.
        """
        cand = ev.primary_value
        inc = self.incumbent.primary_value if self.incumbent else None
        inc_version = self.incumbent.version if self.incumbent else None
        if cand is None:
            return PromotionResult(False, "candidate has no primary metric", None, inc, inc_version)
        if ev.leaks:
            probe = ev.metrics.get("leakage_probe", {})
            return PromotionResult(
                False,
                f"leakage detected — refusing promotion "
                f"(shuffled_label_auc={probe.get('shuffled_label_auc')}, "
                f"max_single_feature_auc={probe.get('max_single_feature_auc')})",
                cand,
                inc,
                inc_version,
            )
        if self.incumbent is not None and self.incumbent.primary_metric != ev.primary_metric:
            return PromotionResult(
                False,
                f"primary metric mismatch (incumbent={self.incumbent.primary_metric}, "
                f"candidate={ev.primary_metric})",
                cand,
                inc,
                inc_version,
            )
        if self.incumbent is not None and self.incumbent.holdout_id != ev.holdout_id:
            return PromotionResult(
                False,
                f"holdout mismatch (incumbent={self.incumbent.holdout_id[:12]}, "
                f"candidate={ev.holdout_id[:12]}) — refusing cross-holdout comparison",
                cand,
                inc,
                inc_version,
            )
        if self.incumbent is not None and not (cand > inc):
            return PromotionResult(
                False, f"{cand:.4f} <= incumbent {inc:.4f}", cand, inc, inc_version
            )

        version = 1 if self.incumbent is None else self.incumbent.version + 1
        card = ModelCard(
            version=version,
            model_type=ev.model_type,
            features=list(ev.features),
            primary_metric=ev.primary_metric,
            primary_value=float(cand),
            metrics=ev.metrics,
            training_window=list(ev.training_window),
            n_train=ev.n_train,
            n_test=ev.n_test,
            dataset_path=dataset_path,
            dataset_sha256=dataset_sha256
            if dataset_sha256 is not None
            else (sha256_file(dataset_path) if os.path.exists(dataset_path) else ""),
            holdout_id=ev.holdout_id,
            created_at=created_at,
        )
        self._persist(ev.estimator, card)
        reason = (
            "first candidate (baseline)" if inc is None else f"{cand:.4f} > incumbent {inc:.4f}"
        )
        return PromotionResult(True, reason, cand, inc, version, card)

    def _persist(self, estimator, card: ModelCard) -> None:
        os.makedirs(self.root, exist_ok=True)
        with open(self._estimator_path(card.version), "wb") as f:
            pickle.dump(estimator, f)
        with open(self._card_path(card.version), "w") as f:
            json.dump(asdict(card), f, indent=2)
        self.history.append(card)
        self.incumbent = card
        self._write_manifest()


# --- one-round orchestration -------------------------------------------------


def run_round(
    spec: CandidateSpec,
    df: pd.DataFrame,
    registry: ModelRegistry,
    *,
    dataset_path: str,
    dataset_sha256: str | None = None,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
    created_at: str | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[Evaluation, PromotionResult]:
    """Evaluate one agent-authored candidate and run it through keep-if-better."""
    ev = evaluate_candidate(spec, df, train_frac=train_frac, primary_metric=primary_metric, rng=rng)
    result = registry.consider(
        ev,
        dataset_path=dataset_path,
        dataset_sha256=dataset_sha256,
        created_at=created_at,
    )
    return ev, result


def load_dataset(path: str) -> pd.DataFrame:
    """Load the labeled early-detection table (parquet or csv)."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)
