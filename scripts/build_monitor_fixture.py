#!/usr/bin/env python
"""Regenerate the committed /api/monitor demo fixture + prebuilt registry (bead jds).

Writes ``src/gpusitter/app/fixtures/early_detection/{features.csv,registry/}`` —
a small honest synthetic table and a logreg incumbent promoted through the
unmodified harness keep-if-better judge. Deterministic; this is the provenance of
the committed pickled estimator.

    PYTHONPATH=src python scripts/build_monitor_fixture.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gpusitter.app.monitor_fixture import FIXTURE_DIR, build_fixture  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(FIXTURE_DIR), help="output fixture dir")
    parser.add_argument("--seed", type=int, default=0, help="rng seed (determinism)")
    args = parser.parse_args()

    summary = build_fixture(args.out, seed=args.seed)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
