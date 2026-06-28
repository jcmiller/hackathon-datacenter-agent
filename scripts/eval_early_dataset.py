#!/usr/bin/env python3
"""Evaluate the labeled early-detection dataset (bead lys).

Reads the compact feature table written by ``build_early_dataset.py`` and
answers the only question that matters for the go/no-go verdict: *does kalos
telemetry carry usable pre-Xid warning signal, measured honestly on a
time-ordered held-out split?*

Honesty guards baked in:

* **Time-ordered, GPU-grouped split.** GPUs are ordered by their earliest
  ``t_ref`` and the earliest 70% (by row count) become train, the rest test.
  Grouping by GPU means a positive at ``t_event - H`` and its paired same-GPU
  pre-event negative can never straddle the split — the classic leak that
  inflates AUC. It is also genuinely time-ordered (train is the past, test the
  future), the split the design mandates as primary.
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META_COLS = ("gpu", "node", "gpu_idx", "t_ref", "event_source",
             "horizon_s", "lookback_s", "label")
TRAIN_FRAC = 0.70
ALERT_BUDGETS = (0.01, 0.05, 0.10)  # fraction of held-out points we may alert on


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


def gpu_grouped_time_split(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Earliest-70%-of-GPUs (by t_ref) -> train, latest 30% -> test.

    GPU-atomic (no GPU appears on both sides) and time-ordered (train precedes
    test). Returns boolean masks (train, test) aligned to ``df.index``.
    """
    t = pd.to_datetime(df["t_ref"])
    order = (pd.DataFrame({"gpu": df["gpu"].values, "t": t.values})
             .groupby("gpu")["t"].min().sort_values())
    counts = df.groupby("gpu").size()
    cum, cutoff, total = 0, [], len(df)
    train_gpus = set()
    for gpu in order.index:
        if cum < TRAIN_FRAC * total:
            train_gpus.add(gpu)
            cum += int(counts[gpu])
    train_mask = df["gpu"].isin(train_gpus).values
    return train_mask, ~train_mask


def _alert_budget_metrics(y_true: np.ndarray, scores: np.ndarray,
                          budget: float) -> Dict[str, float]:
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


def _safe_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def _safe_ap(y: np.ndarray, s: np.ndarray) -> Optional[float]:
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
        "hgb": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, random_state=0),
    }


def _fit_score(model, Xtr, ytr, Xte) -> np.ndarray:
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def evaluate_horizon(df_h: pd.DataFrame, feats: List[str],
                     rng: np.random.Generator) -> Dict:
    train_mask, test_mask = gpu_grouped_time_split(df_h)
    Xtr = df_h.loc[train_mask, feats].to_numpy(dtype="float64")
    Xte = df_h.loc[test_mask, feats].to_numpy(dtype="float64")
    ytr = df_h.loc[train_mask, "label"].to_numpy(dtype="int")
    yte = df_h.loc[test_mask, "label"].to_numpy(dtype="int")

    out: Dict = {
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        "pos_train": int(ytr.sum()), "pos_test": int(yte.sum()),
        "models": {},
    }
    if out["n_train"] < 10 or out["n_test"] < 5 or ytr.sum() == 0 \
            or len(np.unique(yte)) < 2:
        out["skipped"] = "insufficient class balance / size on time-ordered split"
        return out

    for name, model in _models().items():
        scores = _fit_score(model, Xtr, ytr, Xte)
        # no-signal control: identical model, shuffled train labels.
        perm = rng.permutation(ytr)
        try:
            perm_scores = _fit_score(_models()[name], Xtr, perm, Xte)
            perm_auc = _safe_auc(yte, perm_scores)
        except Exception:
            perm_auc = None
        out["models"][name] = {
            "roc_auc": _safe_auc(yte, scores),
            "avg_precision": _safe_ap(yte, scores),
            "roc_auc_permuted_baseline": perm_auc,
            "base_rate": float(yte.mean()),
            "alert_budget": [_alert_budget_metrics(yte, scores, b)
                             for b in ALERT_BUDGETS],
        }
    return out


def stratified_random_smoke(df: pd.DataFrame, feats: List[str],
                            rng: np.random.Generator) -> Dict:
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


def run(data_path: str) -> Dict:
    df = pd.read_csv(data_path)
    feats = feature_columns(df)
    rng = np.random.default_rng(0)
    horizons = sorted(df["horizon_s"].unique())
    report: Dict = {
        "data_path": data_path,
        "n_rows": int(len(df)),
        "n_features": len(feats),
        "n_positive": int(df["label"].sum()),
        "n_gpus": int(df["gpu"].nunique()),
        "horizons_s": [float(h) for h in horizons],
        "split": "time-ordered, GPU-grouped (earliest 70% of GPUs by t_ref -> train)",
        "per_horizon": {},
        "stratified_random_smoke": stratified_random_smoke(df, feats, rng),
    }
    for h in horizons:
        report["per_horizon"][str(int(h))] = evaluate_horizon(
            df[df["horizon_s"] == h].copy(), feats, rng)
    return report


# --- reporting ---------------------------------------------------------------


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def to_markdown(rep: Dict) -> str:
    L: List[str] = []
    L.append("# Early-detection dataset — evaluation report")
    L.append("")
    L.append(f"- Dataset: `{rep['data_path']}`")
    L.append(f"- Rows: {rep['n_rows']}  ·  positives: {rep['n_positive']}  ·  "
             f"GPUs: {rep['n_gpus']}  ·  features: {rep['n_features']}")
    L.append(f"- Horizons (s): {rep['horizons_s']}")
    L.append(f"- Split: {rep['split']}")
    L.append("")
    L.append("## Held-out metrics by horizon (lead time = horizon)")
    L.append("")
    L.append("| Horizon | model | n_test | pos_test | ROC-AUC | AP | "
             "permuted-AUC | recall@5% | prec@5% |")
    L.append("|--:|:--|--:|--:|--:|--:|--:|--:|--:|")
    for h, hr in rep["per_horizon"].items():
        if "skipped" in hr:
            L.append(f"| {h}s | — | {hr['n_test']} | {hr['pos_test']} | "
                     f"skipped: {hr['skipped']} | | | | |")
            continue
        for name, m in hr["models"].items():
            ab5 = next(a for a in m["alert_budget"] if a["budget"] == 0.05)
            L.append(f"| {h}s | {name} | {hr['n_test']} | {hr['pos_test']} | "
                     f"{_fmt(m['roc_auc'])} | {_fmt(m['avg_precision'])} | "
                     f"{_fmt(m['roc_auc_permuted_baseline'])} | "
                     f"{_fmt(ab5['recall'])} | {_fmt(ab5['precision'])} |")
    L.append("")
    L.append(f"- Stratified random smoke (debug only, not a perf claim): "
             f"{rep['stratified_random_smoke']}")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
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
