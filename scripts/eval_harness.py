#!/usr/bin/env python3
"""Run one agent-authored candidate through the eval harness (bead glf).

Trains a candidate (``--model`` over ``--features`` or all features) on the
strict time-ordered train split of the labeled early-detection dataset, scores it
on the held-out split, and runs it through keep-if-better against the persisted
registry under ``--registry``. The candidate is promoted (a new version, pickled
with its model card) only if it beats the incumbent on the primary metric.

This is the loop a self-improving agent (bead rnh) drives: it authors candidates;
this command — which it does not edit — decides whether each one is kept.

Usage::

    python scripts/eval_harness.py --data data/early_detection.csv \
        --registry models/early_detection --model hgb

    # restart / inspect: report the persisted incumbent without training
    python scripts/eval_harness.py --registry models/early_detection --show
"""

from __future__ import annotations

import argparse
import json

from gpusitter.detection.harness import (
    DEFAULT_PRIMARY_METRIC,
    DEFAULT_TRAIN_FRAC,
    CandidateSpec,
    ModelRegistry,
    load_dataset,
    run_round,
    sha256_file,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--registry", required=True, help="directory holding the model registry")
    p.add_argument("--data", help="labeled early_detection.csv/.parquet (omit with --show)")
    p.add_argument("--model", default="hgb", choices=("logreg", "hgb"))
    p.add_argument("--features", nargs="*", default=None, help="feature columns (default: all)")
    p.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    p.add_argument("--primary-metric", default=DEFAULT_PRIMARY_METRIC)
    p.add_argument("--show", action="store_true", help="report incumbent and exit")
    args = p.parse_args(argv)

    registry = ModelRegistry(args.registry)
    if args.show or not args.data:
        print(registry.describe_incumbent())
        return 0

    df = load_dataset(args.data)
    spec = CandidateSpec(args.model, tuple(args.features) if args.features else ())
    ev, result = run_round(
        spec,
        df,
        registry,
        dataset_path=args.data,
        dataset_sha256=sha256_file(args.data),
        train_frac=args.train_frac,
        primary_metric=args.primary_metric,
    )

    print(
        json.dumps(
            {
                "candidate": {"model": spec.model_type, "n_features": len(ev.features)},
                "primary_metric": ev.primary_metric,
                "candidate_value": ev.primary_value,
                "roc_auc": ev.metrics["roc_auc"],
                "avg_precision": ev.metrics["avg_precision"],
                "permuted_baseline": ev.metrics["roc_auc_permuted_baseline"],
                "leakage_probe": ev.metrics["leakage_probe"],
                "promoted": result.promoted,
                "reason": result.reason,
                "version": result.version,
                "incumbent": registry.describe_incumbent(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
