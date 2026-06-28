#!/usr/bin/env python3
"""Validate the job<->telemetry RCA join on the real kalos trace (droplet).

Streams Xid onsets from the full XID CSV (memory-bounded), loads FAILED jobs
from trace_kalos, and reports the temporal coincidence rate: what fraction of
in-window FAILED jobs have a diagnosable Xid onset within +/- W minutes of
fail_time. Cross-timezone (job UTC vs telemetry +08:00) is handled by tz-aware
parsing.

Usage (on the droplet, repo root):
    python scripts/validate_rca.py \
        --xid data/acme-util/data/utilization/kalos/XID_ERRORS.csv \
        --trace data/acme-util/data/job_trace/trace_kalos.csv \
        --window-min 5
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gpusitter.rca.job_join import (  # noqa: E402
    coincidence,
    load_failed_jobs,
    stream_xid_onsets,
    xid_sample_span,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xid", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--window-min", type=float, default=5.0)
    args = ap.parse_args()

    if os.path.getsize(args.xid) < 1024:
        print(f"{args.xid} looks like an LFS pointer — re-hydrate first", file=sys.stderr)
        return 1

    # Telemetry SAMPLE coverage — the denominator window — is independent of
    # (and wider than) the onset span. Jobs in this window but outside the onset
    # span are real in-window jobs that simply have no nearby onset (unmatched).
    span = xid_sample_span(args.xid)
    print(f"  telemetry sample span: {span[0]} .. {span[1]}")

    print("streaming Xid onsets (full file, memory-bounded)...")
    onsets = stream_xid_onsets(args.xid)
    print(f"  total Xid onset events: {len(onsets)}")
    if onsets:
        print(f"  onset span: {onsets[0][0]} .. {onsets[-1][0]}")
    else:
        print("no onsets found")
        return 0

    jobs = load_failed_jobs(args.trace)
    print(f"FAILED jobs in trace: {len(jobs)}")

    results, rate = coincidence(onsets, jobs, args.window_min, telemetry_span=span)
    in_window = len(results)
    matched = sum(1 for r in results if r["matched"])
    print(f"FAILED jobs within telemetry span: {in_window}")
    print(
        f"coincidence rate (+/-{args.window_min:g} min): {matched}/{in_window} = "
        f"{rate:.1%} of in-window FAILED jobs co-time a diagnosable Xid onset"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
