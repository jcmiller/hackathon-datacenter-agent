#!/usr/bin/env python3
"""Validate the telemetry ingest against the real kalos DCGM subset.

The kalos CSVs are ~1 GB each and, on the cache-only droplet, the working tree
holds only Git LFS pointers — the real frames live in the LFS object cache. This
script resolves each metric *cache-safe* (materialized working-tree file, else the
LFS cache object), loads a *bounded time window* across all available metrics —
never the full frame — and prints shape stats plus a sample point lookup and
snapshot, proving the store query surface works on real data with real GPU-id
namespaces.

Usage (on the droplet, from repo root) — cache-safe, resolves the LFS cache::

    python scripts/validate_kalos.py \
        --repo-dir data/acme-util \
        --start "2023-08-15 15:30:15+08:00" --end "2023-08-15 15:45:00+08:00"

Or against a loose directory of already-materialized CSVs::

    python scripts/validate_kalos.py \
        --data-dir data/acme-util/data/utilization/kalos \
        --start "2023-08-15 15:30:15+08:00" --end "2023-08-15 15:45:00+08:00"
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Make ``src`` importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gpusitter.telemetry.ingest import iter_long_records  # noqa: E402
from gpusitter.telemetry.sources import (  # noqa: E402
    resolve_metric_csv,
    validate_timeseries_csv,
)
from gpusitter.telemetry.store import TelemetryStore  # noqa: E402

# The canonical kalos metrics. On the droplet most are LFS pointers in the
# working tree; the cache-safe resolver reads them from the LFS object cache.
KALOS_METRICS = [
    "XID_ERRORS",
    "GPU_TEMP",
    "GPU_UTIL",
    "POWER_USAGE",
    "MEMORY_TEMP",
    "MEM_CLOCK",
    "SM_ACTIVE",
]


def resolve_sources(*, repo_dir: str | None = None, data_dir: str | None = None) -> dict[str, str]:
    """Resolve readable CSV paths for the kalos metrics, cache-safe.

    ``repo_dir`` (the acme-util repo root) resolves each canonical metric through
    the Git LFS cache when the working tree holds only a pointer — the droplet
    cache-only state. ``data_dir`` reads a loose directory of *materialized* CSVs.
    Exactly one must be given. Metrics that are absent, not fetched on this host
    (LFS object missing), or not a wide telemetry frame are skipped — never read
    via a brittle byte-size heuristic.
    """
    if (repo_dir is None) == (data_dir is None):
        raise ValueError("provide exactly one of repo_dir / data_dir")

    sources: dict[str, str] = {}
    for metric in KALOS_METRICS:
        if repo_dir is not None:
            try:
                sources[metric] = resolve_metric_csv(metric, repo_dir=repo_dir)
            except FileNotFoundError:
                continue  # LFS object not fetched on this host
            except ValueError as exc:
                print(f"skipping {metric}: {exc}", file=sys.stderr)
            continue
        path = os.path.join(data_dir, f"{metric}.csv")
        if not os.path.exists(path):
            continue
        try:
            sources[metric] = validate_timeseries_csv(path)
        except ValueError as exc:
            print(f"skipping {metric}: {exc}", file=sys.stderr)
    return sources


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--repo-dir",
        help="acme-util repo root; resolves metrics cache-safe via the Git LFS cache",
    )
    src.add_argument("--data-dir", help="directory of already-materialized kalos CSVs")
    ap.add_argument("--start", required=True, help="inclusive ISO timestamp")
    ap.add_argument("--end", required=True, help="inclusive ISO timestamp")
    ap.add_argument("--downsample", type=int, default=1)
    args = ap.parse_args()

    sources = resolve_sources(repo_dir=args.repo_dir, data_dir=args.data_dir)
    if not sources:
        where = args.repo_dir or args.data_dir
        print(f"no usable metric CSVs resolved from {where}", file=sys.stderr)
        return 1
    print(f"metrics present: {sorted(sources)}")

    t0 = time.time()
    store = TelemetryStore.load(
        sources,
        time_range=(args.start, args.end),
        downsample=args.downsample if args.downsample > 1 else None,
    )
    dt = time.time() - t0

    gpu_ids = store.gpus()
    print(f"loaded window [{args.start} .. {args.end}] in {dt:.1f}s")
    print(f"  GPUs observed:   {len(gpu_ids)}")
    print(f"  metrics in store: {store.metrics()}")

    # Namespace breakdown: IP-named vs pod-named canonical ids.
    ip = [g for g in gpu_ids if g[0].isdigit()]
    pod = [g for g in gpu_ids if g.startswith("lingjun")]
    print(f"  IP-named GPUs:   {len(ip)}   pod-named GPUs: {len(pod)}")

    # Sample point lookup + snapshot on the first IP-named GPU with data.
    for g in ip or gpu_ids:
        ts = store.timestamps(g)
        if not ts:
            continue
        t = ts[0]
        snap = store.snapshot(g, t)
        print(f"\nsample GPU {g} @ {t}")
        print(f"  snapshot (metrics x value): {snap}")
        for metric in snap:
            print(f"  value({metric}) = {store.value(metric, g, t)}")
        break

    # Streaming-melt sanity: count non-empty XID cells in the window without
    # building a store (proves the sparse streaming path on real data).
    if "XID_ERRORS" in sources:
        n_xid = sum(
            1
            for _ in iter_long_records(
                sources["XID_ERRORS"],
                "XID_ERRORS",
                time_range=(args.start, args.end),
            )
        )
        print(f"\nstreamed XID_ERRORS non-empty cells in window: {n_xid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
