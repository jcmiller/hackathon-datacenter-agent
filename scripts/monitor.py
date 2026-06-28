#!/usr/bin/env python3
"""Score the live feature stream + detect alert-budget misses (bead i6k).

Loads the labeled per-GPU feature table (the lys/r7j substrate — NOT jobs.csv) and
the usable pickled incumbent from a model registry, scores every row, derives the
alert-budget thresholds, and runs the horizon-grid miss detector. Prints the
operational report — per-budget thresholds, per-horizon recall (caught onsets /
total), and the miss events that would trigger the agent loop to retrain.

This is the OPERATIONAL counterpart to ``eval_harness.py``: that command decides
whether a candidate is kept; this one runs the kept incumbent against the stream
and surfaces where it would have *missed* a real Xid onset.

Usage::

    python scripts/monitor.py --data data/early_detection.parquet \
        --registry models/early_detection

    # narrow the grid and write a dashboard artifact
    python scripts/monitor.py --data data/early_detection.parquet \
        --registry models/early_detection --budget 0.05 --out docs/monitor_report.json
"""

from __future__ import annotations

import argparse
import json

from gpusitter.detection.harness import ModelRegistry, load_dataset
from gpusitter.detection.monitor import (
    DEFAULT_BUDGETS,
    DEFAULT_HORIZONS_S,
    RowScorer,
    monitor_report,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, help="labeled feature table (.parquet/.csv)")
    p.add_argument("--registry", required=True, help="model registry holding the incumbent")
    p.add_argument("--budget", type=float, action="append", help="alert budget (repeatable)")
    p.add_argument("--horizon", type=float, action="append", help="horizon seconds (repeatable)")
    p.add_argument("--max-rows", type=int, default=500, help="per-row sample cap in the report")
    p.add_argument("--out", help="write the full JSON report here (default: stdout summary)")
    args = p.parse_args(argv)

    registry = ModelRegistry(args.registry)
    if registry.incumbent is None:
        print(json.dumps({"available": False, "reason": "no persisted incumbent in registry"}))
        return 1

    df = load_dataset(args.data)
    scorer = RowScorer.from_registry(registry)
    report = monitor_report(
        df,
        scorer,
        budgets=tuple(args.budget) if args.budget else DEFAULT_BUDGETS,
        horizons_s=tuple(args.horizon) if args.horizon else DEFAULT_HORIZONS_S,
        max_rows=args.max_rows,
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")

    # Compact operator summary (the full per-row detail lives in --out).
    print(
        json.dumps(
            {
                "incumbent": registry.describe_incumbent(),
                "model_version": report["model_version"],
                "n_rows": report["n_rows"],
                "n_onsets": report["n_onsets"],
                "budgets": [
                    {
                        "budget": b["budget"],
                        "threshold": round(b["threshold"], 4),
                        "alert_rate": round(b["alert_rate"], 4),
                        "recall_by_horizon": {
                            h: round(cell["recall"], 4)
                            if cell["recall"] == cell["recall"]
                            else None
                            for h, cell in b["grid"]["by_horizon"].items()
                        },
                        "missed_by_horizon": {
                            h: cell["missed"] for h, cell in b["grid"]["by_horizon"].items()
                        },
                    }
                    for b in report["budgets"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
