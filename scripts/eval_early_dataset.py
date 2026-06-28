#!/usr/bin/env python3
"""Evaluate the labeled early-detection dataset (bead lys).

Reads the compact feature table written by ``build_early_dataset.py`` and
answers the only question that matters for the go/no-go verdict: *does kalos
telemetry carry usable pre-Xid warning signal, measured honestly on a
time-ordered held-out split?*

Honesty guards baked in:

* **Strict time-ordered split.** Rows are cut at a single ``t_ref`` threshold:
  the earliest ~70% by time become train, the rest test, with
  ``max(train.t_ref) < min(test.t_ref)``. Every training point is strictly in the
  past of every test point, so the model can never see future telemetry — the
  no-future-leakage guarantee the design mandates as primary. (An earlier
  GPU-bucketed variant ordered GPUs by their *earliest* t_ref but filled train by
  row count, which let a train GPU's later rows post-date test rows — not actually
  time-ordered; replaced.) Same-event pairs stay safe: a straddle puts the
  earlier (pre-event) point in train and the later in test — real-world ordering,
  not leakage.
* **No-signal permutation baseline.** The same model is retrained on *shuffled*
  train labels and scored on the real held-out labels. If the real model's
  held-out AUC is not clearly above this permuted control, there is no signal —
  it is the difference, not the absolute number, that proves lift.
* **Per-horizon = lead-time table.** Each horizon H is evaluated separately; a
  positive sits exactly H seconds before its onset, so recall at horizon H *is*
  the fraction of real faults catchable with H-seconds lead at that alert
  budget.

Missingness is explicit in the features (``present``/``coverage`` plus NaN
stats); we never zero-fill. LogisticRegression gets median-imputed+scaled
inputs; HistGradientBoosting consumes NaN natively.

Usage::

    python scripts/eval_early_dataset.py --data data/early_detection.csv \
        --out-json data/early_detection_eval.json \
        --out-md docs/early-detection-eval.md
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META_COLS = ("gpu", "node", "gpu_idx", "t_ref", "event_source", "horizon_s", "lookback_s", "label")
TRAIN_FRAC = 0.70
ALERT_BUDGETS = (0.01, 0.05, 0.10)  # fraction of held-out points we may alert on
N_PERMUTATIONS = 8  # label-shuffles averaged for the no-signal baseline


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def time_ordered_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Strict time-ordered split on ``t_ref``: earliest ~``TRAIN_FRAC`` -> train.

    The split is a single ``t_ref`` threshold ``t_cut`` (the t_ref of the first
    test row in time order): **train = rows strictly before ``t_cut``, test =
    rows at/after it.** So ``max(train.t_ref) < t_cut <= min(test.t_ref)`` — every
    training point is strictly in the past of every test point and the model can
    never see future telemetry. This is the no-future-leakage guarantee the
    earlier GPU-bucketed split silently violated (a train GPU's later rows could
    post-date a test GPU's earlier rows).

    Same-event pairs are still safe: a positive at ``t_event - H`` and its
    same-GPU pre-event negative at ``t_event - neg_offset`` differ in time, so a
    straddle puts the *earlier* point in train and the *later* in test — the
    real-world ordering, not leakage. Returns positional boolean masks aligned to
    ``df`` row order.
    """
    t = pd.to_datetime(df["t_ref"]).values
    order = np.argsort(t, kind="mergesort")  # stable, chronological
    n = len(df)
    k = max(1, int(round(TRAIN_FRAC * n)))
    t_cut = t[order[k]] if k < n else t[order[-1]]
    train_mask = t < t_cut
    return train_mask, ~train_mask


def _alert_budget_metrics(
    y_true: np.ndarray, scores: np.ndarray, budget: float
) -> dict[str, float]:
    """Precision/recall when we alert on the top ``budget`` fraction of scores."""
    n = len(scores)
    k = max(1, int(round(budget * n)))
    order = np.argsort(scores)[::-1]
    flagged = order[:k]
    tp = int(y_true[flagged].sum())
    total_pos = int(y_true.sum())
    return {
        "budget": budget,
        "k": k,
        "precision": tp / k if k else float("nan"),
        "recall": tp / total_pos if total_pos else float("nan"),
        "flagged_positive": tp,
    }


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def _safe_ap(y: np.ndarray, s: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, s))


def _models():
    return {
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "hgb": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0),
    }


def _fit_score(model, Xtr, ytr, Xte) -> np.ndarray:
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def evaluate_horizon(df_h: pd.DataFrame, feats: list[str], rng: np.random.Generator) -> dict:
    train_mask, test_mask = time_ordered_split(df_h)
    Xtr = df_h.loc[train_mask, feats].to_numpy(dtype="float64")
    Xte = df_h.loc[test_mask, feats].to_numpy(dtype="float64")
    ytr = df_h.loc[train_mask, "label"].to_numpy(dtype="int")
    yte = df_h.loc[test_mask, "label"].to_numpy(dtype="int")

    out: dict = {
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "pos_train": int(ytr.sum()),
        "pos_test": int(yte.sum()),
        "models": {},
    }
    if out["n_train"] < 10 or out["n_test"] < 5 or ytr.sum() == 0 or len(np.unique(yte)) < 2:
        out["skipped"] = "insufficient class balance / size on time-ordered split"
        return out

    for name, model in _models().items():
        scores = _fit_score(model, Xtr, ytr, Xte)
        # No-signal control: the identical model retrained on shuffled train
        # labels. A single shuffle is high-variance on a small held-out set
        # (it swung 0.39-0.58 here), so average several so the control is a
        # stable ~0.5 reference and the reported lift is the gap above it.
        perm_aucs = []
        for _ in range(N_PERMUTATIONS):
            perm = rng.permutation(ytr)
            try:
                perm_scores = _fit_score(_models()[name], Xtr, perm, Xte)
                a = _safe_auc(yte, perm_scores)
                if a is not None:
                    perm_aucs.append(a)
            except Exception:
                pass
        perm_auc = float(np.mean(perm_aucs)) if perm_aucs else None
        out["models"][name] = {
            "roc_auc": _safe_auc(yte, scores),
            "avg_precision": _safe_ap(yte, scores),
            "roc_auc_permuted_baseline": perm_auc,
            "roc_auc_permuted_n": len(perm_aucs),
            "base_rate": float(yte.mean()),
            "alert_budget": [_alert_budget_metrics(yte, scores, b) for b in ALERT_BUDGETS],
        }
    return out


def stratified_random_smoke(df: pd.DataFrame, feats: list[str], rng: np.random.Generator) -> dict:
    """Secondary, debug-only stratified random split (NOT a performance claim)."""
    idx = rng.permutation(len(df))
    cut = int(TRAIN_FRAC * len(df))
    tr, te = idx[:cut], idx[cut:]
    X = df[feats].to_numpy(dtype="float64")
    y = df["label"].to_numpy(dtype="int")
    if y[tr].sum() == 0 or len(np.unique(y[te])) < 2:
        return {"skipped": "insufficient balance"}
    res = {}
    for name, model in _models().items():
        s = _fit_score(model, X[tr], y[tr], X[te])
        res[name] = _safe_auc(y[te], s)
    return res


def run(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    feats = feature_columns(df)
    rng = np.random.default_rng(0)
    horizons = sorted(df["horizon_s"].unique())
    report: dict = {
        "data_path": data_path,
        "n_rows": int(len(df)),
        "n_features": len(feats),
        "n_positive": int(df["label"].sum()),
        "n_gpus": int(df["gpu"].nunique()),
        "horizons_s": [float(h) for h in horizons],
        "split": "strict time-ordered (single t_ref cut; max(train.t_ref) < min(test.t_ref))",
        "per_horizon": {},
        "stratified_random_smoke": stratified_random_smoke(df, feats, rng),
    }
    for h in horizons:
        report["per_horizon"][str(int(h))] = evaluate_horizon(
            df[df["horizon_s"] == h].copy(), feats, rng
        )
    return report


# --- reporting ---------------------------------------------------------------


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def to_markdown(rep: dict) -> str:
    L: list[str] = []
    L.append("# Early-detection dataset — evaluation report")
    L.append("")
    L.append(f"- Dataset: `{rep['data_path']}`")
    L.append(
        f"- Rows: {rep['n_rows']}  ·  positives: {rep['n_positive']}  ·  "
        f"GPUs: {rep['n_gpus']}  ·  features: {rep['n_features']}"
    )
    L.append(f"- Horizons (s): {rep['horizons_s']}")
    L.append(f"- Split: {rep['split']}")
    L.append("")
    L.append("## Held-out metrics by horizon (lead time = horizon)")
    L.append("")
    L.append(
        "| Horizon | model | n_test | pos_test | ROC-AUC | AP | "
        "permuted-AUC | recall@5% | prec@5% |"
    )
    L.append("|--:|:--|--:|--:|--:|--:|--:|--:|--:|")
    for h, hr in rep["per_horizon"].items():
        if "skipped" in hr:
            L.append(
                f"| {h}s | — | {hr['n_test']} | {hr['pos_test']} | "
                f"skipped: {hr['skipped']} | | | | |"
            )
            continue
        for name, m in hr["models"].items():
            ab5 = next(a for a in m["alert_budget"] if a["budget"] == 0.05)
            L.append(
                f"| {h}s | {name} | {hr['n_test']} | {hr['pos_test']} | "
                f"{_fmt(m['roc_auc'])} | {_fmt(m['avg_precision'])} | "
                f"{_fmt(m['roc_auc_permuted_baseline'])} | "
                f"{_fmt(ab5['recall'])} | {_fmt(ab5['precision'])} |"
            )
    L.append("")
    L.append(
        f"- Stratified random smoke (debug only, not a perf claim): "
        f"{rep['stratified_random_smoke']}"
    )
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, help="early_detection.csv/.parquet")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-md", default=None)
    args = p.parse_args(argv)

    rep = run(args.data)
    md = to_markdown(rep)
    print(md)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(rep, f, indent=2)
    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
