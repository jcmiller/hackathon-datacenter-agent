#!/usr/bin/env python3
"""Validate the job<->telemetry RCA join on the real kalos trace (droplet).

Streams Xid onsets from the full XID CSV (memory-bounded), loads FAILED jobs
from trace_kalos, and reports the temporal coincidence rate: what fraction of
in-window FAILED jobs have a diagnosable Xid onset within +/- W minutes of
fail_time. Cross-timezone (job UTC vs telemetry +08:00) is handled by tz-aware
parsing.

Inputs are resolved *cache-safe*: on the cache-only droplet the working tree holds
only Git LFS pointers, so ``--repo-dir`` resolves both the XID frame and the job
trace through the LFS object cache. Explicit ``--xid``/``--trace`` paths still work
for already-materialized files (fail-loud if an XID path is still a pointer).

Usage (on the droplet, repo root) — cache-safe::

    python scripts/validate_rca.py --repo-dir data/acme-util --window-min 5

Or against explicit, already-materialized files::

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
from gpusitter.telemetry.sources import (  # noqa: E402
    resolve_data_path,
    resolve_metric_csv,
    validate_timeseries_csv,
)

# The job trace is not a canonical metric, so it is resolved by repo-relative path.
TRACE_RELPATH = "data/job_trace/trace_kalos.csv"


def resolve_rca_paths(
    *, repo_dir: str | None = None, xid: str | None = None, trace: str | None = None
) -> tuple[str, str]:
    """Return readable ``(xid_csv, trace_csv)`` paths, cache-safe.

    ``repo_dir`` (the acme-util repo root) resolves both through the Git LFS cache
    when the working tree holds only pointers — the droplet cache-only state.
    Otherwise ``xid``/``trace`` are explicit, already-materialized paths; the XID
    path is validated fail-loud so an un-materialized pointer is rejected with a
    pointer at ``--repo-dir`` rather than silently producing zero onsets.
    """
    if repo_dir is not None:
        xid_path = resolve_metric_csv("XID_ERRORS", repo_dir=repo_dir)
        trace_path = resolve_data_path(repo_dir, TRACE_RELPATH)
        return xid_path, trace_path
    if not xid or not trace:
        raise ValueError("provide --repo-dir, or both --xid and --trace")
    try:
        validate_timeseries_csv(xid)
    except ValueError as exc:
        raise ValueError(
            f"{exc}\nThis host appears cache-only; pass --repo-dir <acme-util root> "
            "to resolve XID_ERRORS through the Git LFS cache."
        ) from exc
    return xid, trace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-dir",
        help="acme-util repo root; resolves XID + trace cache-safe via the Git LFS cache",
    )
    ap.add_argument("--xid", help="explicit, already-materialized XID_ERRORS.csv")
    ap.add_argument("--trace", help="explicit, already-materialized job trace CSV")
    ap.add_argument("--window-min", type=float, default=5.0)
    args = ap.parse_args()

    try:
        xid_path, trace_path = resolve_rca_paths(
            repo_dir=args.repo_dir, xid=args.xid, trace=args.trace
        )
    except (ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1

    # Telemetry SAMPLE coverage — the denominator window — is independent of
    # (and wider than) the onset span. Jobs in this window but outside the onset
    # span are real in-window jobs that simply have no nearby onset (unmatched).
    span = xid_sample_span(xid_path)
    print(f"  telemetry sample span: {span[0]} .. {span[1]}")

    print("streaming Xid onsets (full file, memory-bounded)...")
    onsets = stream_xid_onsets(xid_path)
    print(f"  total Xid onset events: {len(onsets)}")
    if onsets:
        print(f"  onset span: {onsets[0][0]} .. {onsets[-1][0]}")
    else:
        print("no onsets found")
        return 0

    jobs = load_failed_jobs(trace_path)
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
