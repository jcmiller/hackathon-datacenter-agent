#!/usr/bin/env python
"""Build the committed REAL dashboard data substrate from Kalos Xid onsets (t7p).

Resolves the canonical kalos CSVs (XID_ERRORS / GPU_TEMP / POWER_USAGE /
GPU_UTIL) through ``gpusitter.telemetry.sources`` — so it reads the single-owner
prepared dataset, never stale ``data/util`` paths — derives edge-detected Xid
onsets, and writes ``src/gpusitter/app/fixtures/dashboard_substrate/``
(meta/fleet/incidents/telemetry + manifest). Deterministic.

Run on the droplet where the raw trace is materialized:

    PYTHONPATH=src python scripts/build_dashboard_substrate.py

Off-droplet the LFS objects are not fetched and this exits with a clear
"raw data not materialized" error — by design; the substrate it produces is the
committed portable artifact the dashboard then serves without the raw trace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gpusitter.app.dashboard_substrate import (  # noqa: E402
    SUBSTRATE_DIR,
    build_substrate,
    write_substrate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(SUBSTRATE_DIR), help="output substrate dir")
    parser.add_argument(
        "--repo-dir",
        default=None,
        help="repo root holding data/ (defaults to the package's resolved REPO_DIR)",
    )
    args = parser.parse_args()

    try:
        substrate = build_substrate(repo_dir=args.repo_dir)
    except FileNotFoundError as exc:
        print(f"raw kalos data not materialized: {exc}", file=sys.stderr)
        return 2
    summary = write_substrate(substrate, args.out)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
