"""Baseline v0 — the no-skill floor the self-improvement curve climbs from (bead 8co).

The learning curve needs an honest zero point. That is a model with *no learned
signal*: a :class:`~sklearn.dummy.DummyClassifier` that predicts the training base
rate for every held-out row. Its held-out ROC-AUC is 0.5 by construction (a
constant score ranks no positive above any negative), so the margin any promoted
candidate earns above this floor *is*, exactly, its learned skill.

Critically the baseline is scored on the **same strict time-ordered split** the
harness uses (:func:`gpusitter.detection.harness.time_ordered_split`), so v0 and
the agent's v1..vN are comparable points on one curve, not numbers from different
holdouts. This module trains nothing learnable and edits none of the harness — it
is a reference point, deliberately dumb.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from gpusitter.detection.harness import DEFAULT_TRAIN_FRAC, time_ordered_split

BASELINE_NAME = "no_skill_prior"


@dataclass
class BaselineEvaluation:
    """The v0 floor: a no-skill prediction scored on the harness held-out split."""

    name: str
    roc_auc: float | None
    avg_precision: float | None
    base_rate: float
    n_train: int
    n_test: int

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_baseline(
    df: pd.DataFrame, *, train_frac: float = DEFAULT_TRAIN_FRAC
) -> BaselineEvaluation:
    """Score the no-skill baseline on the strict time-ordered held-out split.

    Predicts the *train* positive rate for every test row (``strategy="prior"``).
    Held-out ROC-AUC is 0.5 by construction; held-out average precision collapses to
    the test base rate. Raises ``ValueError`` if the split cannot be scored for both
    classes — the same honesty bar the harness applies, so an un-scorable floor is
    never reported as 0.5.
    """
    train_mask, test_mask = time_ordered_split(df, train_frac)
    ytr = df.loc[train_mask, "label"].to_numpy(dtype="int")
    yte = df.loc[test_mask, "label"].to_numpy(dtype="int")
    if len(ytr) < 1 or len(yte) < 1 or len(np.unique(yte)) < 2:
        raise ValueError(
            f"baseline split not scorable (n_train={len(ytr)}, n_test={len(yte)}, "
            f"test_classes={len(np.unique(yte))})"
        )

    clf = DummyClassifier(strategy="prior")
    zeros_tr = np.zeros((len(ytr), 1))
    clf.fit(zeros_tr, ytr)
    scores = clf.predict_proba(np.zeros((len(yte), 1)))[:, 1]

    return BaselineEvaluation(
        name=BASELINE_NAME,
        roc_auc=float(roc_auc_score(yte, scores)),
        avg_precision=float(average_precision_score(yte, scores)),
        base_rate=float(yte.mean()),
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
    )
