#!/usr/bin/env python3
"""Learning-curve demo: the self-improving classifier harness, end to end (bead 8co).

Produces the headline artifact of the 5fq epic — a curve that starts at an honest
no-skill floor (v0) and climbs as the agent's keep-if-better promotions (v1..vN)
beat it on the *same* strict time-ordered held-out split. Each promotion carries
the agent's HYPOTHESIS and REFLECTION (the version diffs / reasoning), so the curve
reads as a story, not just numbers.

Honesty (per the lys NO-GO verdict): real Kalos telemetry carries only a *weak*
pre-Xid signal (held-out logreg ROC-AUC ~0.65, hgb ~0.58, no-signal baseline
~0.50). Those real numbers are read from ``docs/early_detection_eval.json`` and
embedded as the reference block — not fabricated. With ``--data`` the demo runs on
the real dataset; without it, on a deterministic *weak-signal synthetic* table
(effect sizes tuned to mirror Kalos: ROC-AUC ~0.6-0.66 at base rate ~0.18, NOT a
saturated 0.95) so the mechanism is reproducible offline without overselling. The
loop/mechanism is the deliverable.

Usage::

    # reproducible synthetic mechanism demo -> writes docs/learning_curve.json
    python scripts/learning_curve_demo.py

    # on the real dataset (droplet / wherever data/early_detection.csv exists)
    python scripts/learning_curve_demo.py --data data/early_detection.csv
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from gpusitter.detection.agent_loop import ReflectiveProposer, run_loop
from gpusitter.detection.baseline import evaluate_baseline
from gpusitter.detection.harness import (
    DEFAULT_PRIMARY_METRIC,
    DEFAULT_TRAIN_FRAC,
    ModelRegistry,
    feature_columns,
    load_dataset,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ARTIFACT = os.path.join(_ROOT, "docs", "learning_curve.json")
DEFAULT_EVAL_REF = os.path.join(_ROOT, "docs", "early_detection_eval.json")
TZ = timezone(timedelta(hours=8))  # Kalos fixed +08:00
_SYNTH_BASE = datetime(2023, 8, 15, 0, 0, 0, tzinfo=TZ)


# --- deterministic weak-signal synthetic table (mirrors Kalos, honestly weak) -


def synthetic_weak_signal(n: int = 2400, base_rate: float = 0.18, seed: int = 0) -> pd.DataFrame:
    """A reproducible early-detection table with deliberately *weak* signal.

    Mirrors the real Kalos regime so the demo numbers are honest: base rate ~0.18,
    a leading ``temp_mean`` with a small effect size (Cohen's d ~0.55 -> held-out
    ROC-AUC ~0.65), a weaker ``power_mean``, and pure-noise ``util_mean`` /
    ``mem_last``. A boosted model has no edge over the linear signal here (so the
    loop's keep-if-better correctly rejects it), exactly as on real data. Fixed
    seed -> the committed artifact is regenerable byte-for-similar.
    """
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < base_rate).astype(int)
    temp = rng.normal(0.0, 1.0, n) + labels * 0.55  # weak but real
    power = rng.normal(0.0, 1.0, n) + labels * 0.25  # weaker
    util = rng.normal(0.0, 1.0, n)  # noise
    mem = rng.normal(0.0, 1.0, n) + labels * 0.08  # near-noise
    return pd.DataFrame(
        {
            "gpu": [f"node{i % 30}#{i % 8}" for i in range(n)],
            "t_ref": [(_SYNTH_BASE + timedelta(seconds=30 * i)).isoformat() for i in range(n)],
            "horizon_s": np.tile([60.0, 300.0], n // 2)[:n],
            "label": labels,
            "temp_mean": temp,
            "power_mean": power,
            "util_mean": util,
            "mem_last": mem,
        }
    )


# --- real-data honesty reference (read, never fabricated) ---------------------


def real_data_reference(eval_path: str = DEFAULT_EVAL_REF) -> dict | None:
    """Best real held-out numbers from the lys eval — the honest reality anchor.

    Scans every (horizon, model) cell of ``early_detection_eval.json`` and reports
    the strongest real ROC-AUC, its no-signal permuted baseline, and the best
    boosted-model number for contrast. Returns ``None`` if the eval file is absent.
    """
    if not os.path.exists(eval_path):
        return None
    with open(eval_path) as f:
        ev = json.load(f)
    best = {"model": None, "horizon": None, "roc_auc": -1.0, "permuted_baseline": None}
    best_hgb = -1.0
    for horizon, blk in (ev.get("per_horizon") or {}).items():
        for model, m in (blk.get("models") or {}).items():
            auc = m.get("roc_auc")
            if auc is None:
                continue
            if model == "hgb":
                best_hgb = max(best_hgb, auc)
            if auc > best["roc_auc"]:
                best = {
                    "model": model,
                    "horizon": f"{horizon}s",
                    "roc_auc": float(auc),
                    "permuted_baseline": m.get("roc_auc_permuted_baseline"),
                }
    return {
        "source": os.path.relpath(eval_path, _ROOT),
        "dataset": {
            "data_path": ev.get("data_path"),
            "n_rows": ev.get("n_rows"),
            "n_features": ev.get("n_features"),
            "n_positive": ev.get("n_positive"),
        },
        "best_real": best,
        "best_hgb_roc_auc": best_hgb if best_hgb >= 0 else None,
        "verdict": (
            "Weak but real, leakage-free signal (held-out ROC-AUC ~0.65). Not a "
            "standalone reliable predictor (NO-GO); the dataset + self-improving "
            "harness are the reusable deliverable (GO)."
        ),
    }


# --- curve assembly ----------------------------------------------------------


def _round_point(attempt) -> dict:
    r = attempt.reflection
    return {
        "round": attempt.round + 1,
        "model_type": attempt.spec.model_type,
        "n_features": len(attempt.spec.features) if attempt.spec.features else None,
        "features": list(attempt.spec.features),
        "roc_auc": r.roc_auc,
        "signal_gap": r.signal_gap,
        "leaks": r.leaks,
        "promoted": bool(attempt.promotion and attempt.promotion.promoted),
        "version": (attempt.promotion.version if attempt.promotion else None),
        "hypothesis": attempt.hypothesis,
        "reflection": r.notes,
    }


def build_artifact(
    df: pd.DataFrame,
    *,
    source: str,
    synthetic: bool,
    train_frac: float,
    primary_metric: str,
    eval_ref_path: str,
    proposer=None,
) -> dict:
    """Run v0 + the agent loop and assemble the full learning-curve artifact.

    ``proposer`` defaults to the full typed search (hgb + logreg); callers may pass
    a narrower one (e.g. logreg-only) for a faster run.
    """
    baseline = evaluate_baseline(df, train_frac=train_frac)

    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(os.path.join(tmp, "registry"))
        proposer = proposer or ReflectiveProposer(tuple(feature_columns(df)))
        # The dataset is in-memory; record its logical source on the cards.
        data_path = source if os.path.exists(source) else os.path.join(tmp, "dataset.csv")
        if not os.path.exists(data_path):
            df.to_csv(data_path, index=False)
        result = run_loop(
            df,
            registry,
            dataset_path=data_path,
            proposer=proposer,
            train_frac=train_frac,
            primary_metric=primary_metric,
        )

    # The curve: v0 floor, then each keep-if-better promotion (the climbing line).
    promoted = [a for a in result.history if a.promotion and a.promotion.promoted]
    curve = [
        {
            "version": "v0",
            "label": "no-skill baseline",
            "model_type": baseline.name,
            "n_features": 0,
            "roc_auc": baseline.roc_auc,
            "signal_gap": 0.0,
            "hypothesis": "Predict the training base rate for every GPU — the zero-skill floor.",
            "reflection": (
                f"no learned signal; held-out ROC-AUC {baseline.roc_auc:.3f} by construction "
                f"(base rate {baseline.base_rate:.3f}). Everything above this line is real skill."
            ),
        }
    ] + [
        {
            "version": f"v{a.promotion.version}",
            "label": f"{a.spec.model_type} / {len(a.spec.features)} features",
            "model_type": a.spec.model_type,
            "n_features": len(a.spec.features),
            "roc_auc": a.reflection.roc_auc,
            "signal_gap": a.reflection.signal_gap,
            "hypothesis": a.hypothesis,
            "reflection": a.reflection.notes,
        }
        for a in promoted
    ]

    inc = result.incumbent
    return {
        "primary_metric": primary_metric,
        "dataset": {
            "source": source,
            "synthetic": synthetic,
            "n_rows": int(len(df)),
            "n_features": len(feature_columns(df)),
            "n_positive": int(df["label"].sum()),
            "base_rate": float(df["label"].mean()),
        },
        "baseline_v0": baseline.as_dict(),
        "curve": curve,
        "rounds": [_round_point(a) for a in result.history],
        "n_promotions": result.n_promotions,
        "final_incumbent": (
            None
            if inc is None
            else {
                "version": inc.version,
                "model_type": inc.model_type,
                "roc_auc": inc.primary_value,
                "n_features": len(inc.features),
                "features": list(inc.features),
            }
        ),
        "honest_note": (
            "Weak signal is expected on real Kalos telemetry (lys NO-GO verdict). The "
            "self-improving LOOP/mechanism is the deliverable, not a headline AUC. The "
            "synthetic table here is deliberately weak (~0.6-0.66) to mirror reality."
            if synthetic
            else "Run on the real dataset; numbers are the honest held-out reality."
        ),
        "real_data_reference": real_data_reference(eval_ref_path),
    }


def _print_summary(art: dict) -> None:
    print("=== learning curve: no-skill floor -> keep-if-better promotions ===\n")
    for p in art["curve"]:
        auc = p["roc_auc"]
        print(f"  {p['version']:>3}  ROC-AUC {auc:.3f}  {p['label']}")
        print(f"        reflect: {p['reflection']}")
    print("\n=== rounds (incl. rejected candidates) ===")
    for r in art["rounds"]:
        mark = "PROMOTED" if r["promoted"] else "rejected"
        auc = f"{r['roc_auc']:.3f}" if r["roc_auc"] is not None else "n/a"
        print(f"  round {r['round']}: {r['model_type']} ROC-AUC {auc} [{mark}]")
    ref = art.get("real_data_reference")
    if ref:
        b = ref["best_real"]
        print(
            f"\nreal-data anchor: best held-out {b['model']}@{b['horizon']} "
            f"ROC-AUC {b['roc_auc']:.3f} (no-signal {b['permuted_baseline']:.3f}). {ref['verdict']}"
        )
    print(f"\nHONEST: {art['honest_note']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", help="real early_detection.csv/.parquet (default: synthetic demo)")
    p.add_argument("--out", default=DEFAULT_ARTIFACT, help="artifact JSON path")
    p.add_argument("--eval-ref", default=DEFAULT_EVAL_REF, help="real eval json for the anchor")
    p.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    p.add_argument("--primary-metric", default=DEFAULT_PRIMARY_METRIC)
    p.add_argument("--seed", type=int, default=0, help="synthetic generator seed")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.data:
        df = load_dataset(args.data)
        source, synthetic = args.data, False
    else:
        df = synthetic_weak_signal(seed=args.seed)
        source, synthetic = "synthetic-weak-signal", True

    art = build_artifact(
        df,
        source=source,
        synthetic=synthetic,
        train_frac=args.train_frac,
        primary_metric=args.primary_metric,
        eval_ref_path=args.eval_ref,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(art, f, indent=2)
    if not args.quiet:
        _print_summary(art)
        print(f"\nwrote {os.path.relpath(args.out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
