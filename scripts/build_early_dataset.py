#!/usr/bin/env python3
"""Build the labeled early-detection dataset from raw Kalos telemetry.

Offline, run-once on the box where the ~80 GB acme-util repo lives (the droplet).
It resolves each metric CSV *cache-safe* — preferring a materialized working-tree
file, else the Git LFS cache object (even when the working path was deleted) —
then streams it without ever materializing the full wide frame, and writes a
compact reusable feature table the model experiments read instead of rescanning
the raw telemetry.

Droplet example (working tree absent, only the LFS cache present)::

    python scripts/build_early_dataset.py \
        --repo-dir data/acme-util \
        --metrics GPU_TEMP POWER_USAGE GPU_UTIL MEMORY_TEMP \
        --horizons 60 300 600 --lookback 600 \
        --out data/early_detection.parquet

See docs/TEAM_GUIDE.md ("Cache-safe early-detection dataset").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a bare script (python scripts/build_early_dataset.py): make
# both the repo root (for `scripts`) and src/ (for `gpusitter`) importable.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.lfs_helper import kalos_metric_paths, resolve_data_path  # noqa: E402

from gpusitter.detection.early_dataset import (  # noqa: E402
    DEFAULT_FEATURE_METRICS,
    XID_METRIC,
    build_dataset,
    write_dataset,
)


def _resolve_sources(repo_dir: str, metrics: list[str]) -> dict[str, str]:
    """Resolve cache-safe readable paths for XID_ERRORS + each feature metric."""
    names = [XID_METRIC, *metrics]
    rels = kalos_metric_paths(names)
    return {name: resolve_data_path(repo_dir, rel) for name, rel in zip(names, rels, strict=False)}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo-dir", required=True, help="acme-util repo root (e.g. data/acme-util)")
    p.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_FEATURE_METRICS),
        help="feature metric names under data/utilization/kalos/",
    )
    p.add_argument(
        "--horizons",
        nargs="+",
        type=float,
        default=[60, 300, 600],
        help="prediction horizons in seconds",
    )
    p.add_argument(
        "--lookback",
        type=float,
        default=600,
        help="feature lookback window in seconds before t_ref",
    )
    p.add_argument(
        "--neg-offset",
        type=float,
        default=3600,
        help="same-GPU pre-event negative offset in seconds",
    )
    p.add_argument(
        "--sample-period",
        type=float,
        default=15,
        help="nominal telemetry sample period (s) for coverage",
    )
    p.add_argument(
        "--control-gpus",
        nargs="*",
        default=None,
        help="canonical ids ('node#idx') for time-matched controls",
    )
    p.add_argument("--out", required=True, help="output path (.parquet or .csv)")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    sources = _resolve_sources(args.repo_dir, args.metrics)
    rows = build_dataset(
        sources,
        horizons_s=args.horizons,
        lookback_s=args.lookback,
        neg_offset_s=args.neg_offset,
        control_gpus=args.control_gpus,
        sample_period_s=args.sample_period,
    )
    written = write_dataset(rows, args.out)
    pos = sum(int(r["label"]) for r in rows)
    print(f"wrote {written}: {len(rows)} rows ({pos} positive, {len(rows) - pos} negative)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
