#!/usr/bin/env python3
"""Run the agent-authored classifier loop and print its learning curve (bead rnh).

This is the hero-loop demo for 8co. The agent proposes a typed classifier
(model_type + feature subset), the glf harness scores it on the strict
time-ordered held-out split, the loop reflects on the metrics, and the agent
revises — every promotion through keep-if-better, so the incumbent only improves.

Usage::

    python scripts/agent_loop.py --data data/early_detection.csv \
        --registry models/early_detection_loop --max-rounds 8

    # restart / inspect: report the persisted incumbent without running
    python scripts/agent_loop.py --registry models/early_detection_loop --show
"""

from __future__ import annotations

import argparse
import json

from gpusitter.detection.agent_loop import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_MODEL_TYPES,
    ReflectiveProposer,
    run_loop,
)
from gpusitter.detection.harness import (
    DEFAULT_PRIMARY_METRIC,
    DEFAULT_TRAIN_FRAC,
    ModelRegistry,
    feature_columns,
    load_dataset,
    sha256_file,
)


def _print_transcript(result, primary_metric: str) -> None:
    print("=== agent-authored classifier loop: write -> eval -> reflect -> revise ===\n")
    for a in result.history:
        feats = a.spec.features
        n_feat = len(feats) if feats else "all"
        print(f"--- round {a.round + 1}: {a.spec.model_type} on {n_feat} features ---")
        print(f"  HYPOTHESIS: {a.hypothesis}")
        if a.evaluation is not None:
            r = a.reflection
            gap = f"{r.signal_gap:+.3f}" if r.signal_gap is not None else "n/a"
            auc = f"{r.roc_auc:.3f}" if r.roc_auc is not None else "n/a"
            print(f"  held-out {primary_metric}={auc}  signal_gap={gap}  leaks={r.leaks}")
            mark = "PROMOTED" if a.promotion.promoted else "kept incumbent"
            print(f"  decision: {mark} ({a.promotion.reason})")
        print(f"  REFLECT: {a.reflection.notes}\n")

    print("=== learning curve (keep-if-better promotions) ===")
    if result.learning_curve:
        for version, value in result.learning_curve:
            print(f"  v{version}: {primary_metric}={value:.4f}")
    else:
        print("  (no promotions)")
    print(f"\n{_incumbent_line(result)}")


def _incumbent_line(result) -> str:
    if result.incumbent is None:
        return "final incumbent: none"
    c = result.incumbent
    return (
        f"final incumbent: v{c.version} ({c.model_type}) "
        f"{c.primary_metric}={c.primary_value:.4f} over {len(c.features)} features"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--registry", required=True, help="directory holding the model registry")
    p.add_argument("--data", help="labeled early_detection.csv/.parquet (omit with --show)")
    p.add_argument(
        "--model-types",
        nargs="*",
        default=list(DEFAULT_MODEL_TYPES),
        choices=("logreg", "hgb"),
        help="typed search space of model forms",
    )
    p.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    p.add_argument("--primary-metric", default=DEFAULT_PRIMARY_METRIC)
    p.add_argument("--json", action="store_true", help="emit machine-readable summary")
    p.add_argument("--show", action="store_true", help="report incumbent and exit")
    args = p.parse_args(argv)

    registry = ModelRegistry(args.registry)
    if args.show or not args.data:
        print(registry.describe_incumbent())
        return 0

    df = load_dataset(args.data)
    proposer = ReflectiveProposer(tuple(feature_columns(df)), model_types=tuple(args.model_types))
    result = run_loop(
        df,
        registry,
        dataset_path=args.data,
        dataset_sha256=sha256_file(args.data),
        proposer=proposer,
        max_rounds=args.max_rounds,
        train_frac=args.train_frac,
        primary_metric=args.primary_metric,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "rounds": len(result.history),
                    "promotions": result.n_promotions,
                    "learning_curve": result.learning_curve,
                    "incumbent": _incumbent_line(result),
                    "transcript": [
                        {
                            "round": a.round + 1,
                            "model_type": a.spec.model_type,
                            "n_features": len(a.spec.features) if a.spec.features else None,
                            "hypothesis": a.hypothesis,
                            "roc_auc": a.reflection.roc_auc,
                            "signal_gap": a.reflection.signal_gap,
                            "leaks": a.reflection.leaks,
                            "promoted": bool(a.promotion and a.promotion.promoted),
                            "reflection": a.reflection.notes,
                        }
                        for a in result.history
                    ],
                },
                indent=2,
            )
        )
    else:
        _print_transcript(result, args.primary_metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
