"""Operational reactive trigger: per-row risk scoring + alert-budget miss detector (bead i6k).

The offline epic (glf harness / rnh agent_loop / 8co baseline) produces a *usable*
pickled incumbent in a :class:`~gpusitter.detection.harness.ModelRegistry`. What was
missing entirely is the *operational* layer that runs that incumbent against the
live stream of per-GPU feature rows (the lys/r7j substrate):

1. **Per-row scoring.** Every streamed feature row is scored by the current
   incumbent; the row carries ``{risk_score, alert_flag, model_version}``.
2. **Alert budget.** The alert threshold is *derived from a budget* — the
   ``(1 - budget)`` quantile of the score distribution — so ~``budget`` fraction
   of rows fire by construction, never hand-tuned per demo.
3. **Miss detector.** A real Xid onset with no alert inside the horizon
   ``[t_event - H, t_event)`` is a MISS. Each miss carries the prior risk scores
   and the pre-event features, and records the horizon ``H`` it was judged at.
4. **The MISS is the retrain signal.** A miss means the incumbent is inadequate;
   :func:`react_to_misses` is the injectable seam whose default invokes the rnh
   agent loop to author a better candidate through the immutable keep-if-better
   judge — closing the self-improvement loop.

Separation of powers (Design): this module never edits the harness's split,
metrics, leakage probes, or promotion logic. It *consumes* the incumbent the judge
produced and reports honest operational metrics — recall@(budget, horizon) =
caught onsets / total onsets — surfacing weak signal plainly rather than dressing
it up.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd

from gpusitter.detection.harness import META_COLS, ModelRegistry

# 1 / 5 / 10 minutes — the operator-facing lead-time grid (AC #4).
DEFAULT_HORIZONS_S: tuple[float, ...] = (60.0, 300.0, 600.0)
# Fraction of rows we permit to fire. Mirrors harness.ALERT_BUDGETS.
DEFAULT_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.10)


# --- onsets (ground truth) ----------------------------------------------------


class Onset(NamedTuple):
    """A ground-truth Xid onset for one GPU at wall-clock time ``t``."""

    gpu: str
    t: datetime


def onsets_from_dataset(df: pd.DataFrame) -> list[Onset]:
    """Reconstruct ground-truth onsets directly from the labeled feature table.

    A positive prediction point (``label == 1``) sits exactly ``horizon_s`` seconds
    *before* its onset (early_dataset construction), so ``onset = t_ref + horizon_s``.
    The same onset is represented at every horizon, so we dedup on ``(gpu, t)``. This
    keeps the operational layer self-contained on the lys/r7j substrate (AC #6) — no
    need to re-read the raw XID CSV at monitor time.
    """
    if "label" not in df.columns or "horizon_s" not in df.columns:
        raise ValueError("dataset must carry 'label' and 'horizon_s' to recover onsets")
    pos = df[df["label"] == 1]
    seen: set[tuple[str, datetime]] = set()
    onsets: list[Onset] = []
    for gpu, t_ref, h in zip(pos["gpu"], pos["t_ref"], pos["horizon_s"], strict=False):
        t_event = _as_dt(t_ref) + timedelta(seconds=float(h))
        key = (str(gpu), t_event)
        if key not in seen:
            seen.add(key)
            onsets.append(Onset(str(gpu), t_event))
    onsets.sort(key=lambda o: (o.t, o.gpu))
    return onsets


# --- per-row scoring ----------------------------------------------------------


@dataclass
class RowScorer:
    """Adapter over the *usable* incumbent: a fitted estimator + its feature list.

    The estimator is the very object the harness pickled on promotion — not a re-fit.
    NaN missingness is consumed exactly as that model expects (the logreg pipeline
    imputes + scales; hgb consumes NaN natively), so operational scores match the
    held-out scores the judge produced.
    """

    estimator: object
    features: tuple[str, ...]
    model_version: int

    @classmethod
    def from_registry(cls, registry: ModelRegistry, version: int | None = None) -> RowScorer:
        """Load a usable incumbent (default) or a specific version from the registry."""
        if registry.incumbent is None:
            raise ValueError("registry has no persisted incumbent to score with")
        card = registry.incumbent if version is None else _card_for(registry, version)
        estimator = registry.load_estimator(card.version)
        return cls(estimator, tuple(card.features), card.version)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Risk score (P[onset within horizon]) for every row, in ``df`` order."""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"rows missing scorer features: {missing}")
        X = df.loc[:, list(self.features)].to_numpy(dtype="float64")
        return self.estimator.predict_proba(X)[:, 1]


def _card_for(registry: ModelRegistry, version: int):
    card = next((c for c in registry.history if c.version == version), None)
    if card is None:
        raise ValueError(f"no model card for version {version}")
    return card


def calibrate_threshold(scores: np.ndarray, budget: float) -> float:
    """Budget-derived alert threshold: the ``(1 - budget)`` quantile of ``scores``.

    Alerting on ``score >= threshold`` then fires on ~``budget`` fraction of rows by
    construction (AC #2) — the threshold is read off the score distribution, never
    hand-tuned. ``budget`` is clamped to ``(0, 1]``; an empty score array yields
    ``+inf`` (nothing fires).
    """
    s = np.asarray(scores, dtype="float64")
    if s.size == 0:
        return float("inf")
    b = min(max(float(budget), 1e-9), 1.0)
    return float(np.quantile(s, 1.0 - b))


@dataclass
class ScoredRow:
    """One scored streamed feature row (AC #1)."""

    gpu: str
    t_ref: datetime
    horizon_s: float
    label: int
    risk_score: float
    alert_flag: bool
    model_version: int
    features: dict

    def as_dict(self) -> dict:
        d = asdict(self)
        d["t_ref"] = self.t_ref.isoformat()
        return d


@dataclass
class ScoringResult:
    """All scored rows for one budget, plus the threshold that produced the alerts."""

    rows: list[ScoredRow]
    threshold: float
    budget: float
    model_version: int

    @property
    def alert_rate(self) -> float:
        return sum(r.alert_flag for r in self.rows) / len(self.rows) if self.rows else 0.0


def score_rows(df: pd.DataFrame, scorer: RowScorer, *, budget: float) -> ScoringResult:
    """Score every row, derive the budget threshold, and stamp ``alert_flag`` per row.

    The streamed substrate is time-ordered per GPU (early_dataset sorts by
    ``t_ref``); scoring is order-independent, but rows are returned in time order so
    a consumer can replay the stream. Each row carries the incumbent's
    ``model_version`` (AC #1, #5).
    """
    scores = scorer.score(df)
    threshold = calibrate_threshold(scores, budget)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    rows: list[ScoredRow] = []
    for i, (_, r) in enumerate(df.iterrows()):
        score = float(scores[i])
        rows.append(
            ScoredRow(
                gpu=str(r["gpu"]),
                t_ref=_as_dt(r["t_ref"]),
                horizon_s=float(r.get("horizon_s", float("nan"))),
                label=int(r.get("label", 0)),
                risk_score=score,
                alert_flag=bool(score >= threshold),
                model_version=scorer.model_version,
                features={c: _scalar(r[c]) for c in feat_cols},
            )
        )
    rows.sort(key=lambda r: (r.t_ref, r.gpu))
    return ScoringResult(rows, threshold, float(budget), scorer.model_version)


# --- miss detection -----------------------------------------------------------


@dataclass
class MissEvent:
    """A real onset that fired no alert within horizon ``H`` before it (AC #3)."""

    gpu: str
    onset_t: datetime
    horizon_s: float
    model_version: int
    budget: float
    caught: bool
    n_alerts_in_window: int
    prior_scores: list[tuple[str, float, bool]]  # (t_ref_iso, risk_score, alert_flag)
    pre_event_features: dict

    def as_dict(self) -> dict:
        d = asdict(self)
        d["onset_t"] = self.onset_t.isoformat()
        return d


def _rows_by_gpu(rows: Iterable[ScoredRow]) -> dict[str, list[ScoredRow]]:
    by: dict[str, list[ScoredRow]] = {}
    for r in rows:
        by.setdefault(r.gpu, []).append(r)
    for rs in by.values():
        rs.sort(key=lambda r: r.t_ref)
    return by


def detect_misses(
    scored_rows: Sequence[ScoredRow],
    onsets: Sequence[Onset],
    *,
    horizon_s: float,
    model_version: int,
    budget: float,
) -> list[MissEvent]:
    """Window-based operational miss detection at a single horizon ``H``.

    For each onset ``(gpu, t_event)`` the alert window is ``[t_event - H, t_event)``
    — strictly *before* the onset, so a catch means a genuine early warning. The
    onset is **caught** iff any of that GPU's scored rows in the window has
    ``alert_flag``; otherwise it is a **MISS**. Every event (caught or miss) carries
    the prior risk scores in the window and the pre-event features of the latest row
    before the onset, plus the horizon it was judged at.
    """
    by_gpu = _rows_by_gpu(scored_rows)
    events: list[MissEvent] = []
    for onset in onsets:
        lo = onset.t - timedelta(seconds=horizon_s)
        window = [r for r in by_gpu.get(onset.gpu, ()) if lo <= r.t_ref < onset.t]
        alerts = [r for r in window if r.alert_flag]
        pre_event = window[-1].features if window else {}
        events.append(
            MissEvent(
                gpu=onset.gpu,
                onset_t=onset.t,
                horizon_s=float(horizon_s),
                model_version=model_version,
                budget=float(budget),
                caught=bool(alerts),
                n_alerts_in_window=len(alerts),
                prior_scores=[(r.t_ref.isoformat(), r.risk_score, r.alert_flag) for r in window],
                pre_event_features=pre_event,
            )
        )
    return events


def horizon_grid(
    scored_rows: Sequence[ScoredRow],
    onsets: Sequence[Onset],
    *,
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
    model_version: int,
    budget: float,
) -> dict:
    """Run miss detection across the horizon grid; record ``H`` on each miss (AC #4).

    Returns, per horizon, the recall (caught / total onsets), the miss count, and the
    miss events themselves. Recall is monotonic non-decreasing in ``H`` for a fixed
    alert set: a wider window can only admit more alerts, never fewer.
    """
    alert_rate = sum(r.alert_flag for r in scored_rows) / len(scored_rows) if scored_rows else 0.0
    grid: dict[str, dict] = {}
    for h in horizons_s:
        events = detect_misses(
            scored_rows, onsets, horizon_s=h, model_version=model_version, budget=budget
        )
        caught = sum(e.caught for e in events)
        misses = [e for e in events if not e.caught]
        grid[str(int(h))] = {
            "horizon_s": float(h),
            "n_onsets": len(events),
            "caught": caught,
            "missed": len(misses),
            "recall": caught / len(events) if events else float("nan"),
            "misses": [e.as_dict() for e in misses],
        }
    return {
        "model_version": model_version,
        "budget": float(budget),
        "alert_rate": alert_rate,
        "by_horizon": grid,
    }


# --- dashboard report ---------------------------------------------------------


def monitor_report(
    df: pd.DataFrame,
    scorer: RowScorer,
    *,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
    max_rows: int = 500,
) -> dict:
    """JSON-serializable per-row scores + alert/miss status for the dashboard (AC #5).

    One scoring pass per budget (the threshold is budget-specific), each fed through
    the horizon grid. A bounded ``max_rows`` sample of the scored stream is included
    so the dashboard can render the per-row risk timeline without shipping the whole
    table.
    """
    onsets = onsets_from_dataset(df)
    budget_reports = []
    sample_rows: list[dict] = []
    for budget in budgets:
        result = score_rows(df, scorer, budget=budget)
        grid = horizon_grid(
            result.rows,
            onsets,
            horizons_s=horizons_s,
            model_version=scorer.model_version,
            budget=budget,
        )
        budget_reports.append(
            {
                "budget": float(budget),
                "threshold": result.threshold,
                "alert_rate": result.alert_rate,
                "grid": grid,
            }
        )
        if not sample_rows:  # one representative sample (lowest budget)
            sample_rows = [r.as_dict() for r in result.rows[:max_rows]]
    return {
        "available": True,
        "model_version": scorer.model_version,
        "features": list(scorer.features),
        "n_rows": int(len(df)),
        "n_onsets": len(onsets),
        "budgets": budget_reports,
        "rows": sample_rows,
    }


# --- miss -> retrain trigger (closes the self-improvement loop) ----------------


@dataclass
class RetrainTrigger:
    """Outcome of reacting to misses: whether retraining was fired and why."""

    triggered: bool
    reason: str
    n_misses: int
    result: object = None  # whatever the trainer returned (e.g. a LoopResult)


# A trainer takes the labeled dataset + registry and authors/evaluates a candidate.
Trainer = Callable[[pd.DataFrame, ModelRegistry], object]


def _default_trainer(df: pd.DataFrame, registry: ModelRegistry, *, dataset_path: str) -> object:
    """Default retrain seam: drive the rnh agent loop through the immutable judge."""
    from gpusitter.detection.agent_loop import run_loop

    return run_loop(df, registry, dataset_path=dataset_path)


def react_to_misses(
    df: pd.DataFrame,
    registry: ModelRegistry,
    misses: Sequence[MissEvent],
    *,
    dataset_path: str = "",
    min_misses: int = 1,
    trainer: Trainer | None = None,
) -> RetrainTrigger:
    """A MISS is the operational signal to retrain (AC: 'this miss triggers the agent').

    When at least ``min_misses`` real onsets slipped through, invoke the (injectable)
    ``trainer`` — by default the rnh agent loop — on the accreting dataset so a better
    candidate is authored and run through keep-if-better. The seam is dependency-
    injected so the backend and tests can verify the wiring without heavy training.
    """
    n = len(misses)
    if n < min_misses:
        return RetrainTrigger(False, f"{n} miss(es) < threshold {min_misses}", n)
    train = trainer or (lambda d, r: _default_trainer(d, r, dataset_path=dataset_path))
    result = train(df, registry)
    return RetrainTrigger(True, f"{n} miss(es) -> retrain fired", n, result)


# --- helpers ------------------------------------------------------------------


def _as_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _scalar(v):
    """JSON-friendly scalar (numpy -> python, NaN -> None)."""
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if np.isnan(f) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    return v
