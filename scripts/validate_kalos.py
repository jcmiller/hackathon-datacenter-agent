#!/usr/bin/env python3
"""Validate the telemetry ingest against the real kalos DCGM subset.

The kalos CSVs live only on the droplet (gitignored, ~1 GB each). This script
loads a *bounded time window* across all available metrics — never the full
frame — and prints shape stats plus a sample point lookup and snapshot, proving
the store query surface works on real data with real GPU-id namespaces.

Usage (on the droplet, from repo root):
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
from gpusitter.telemetry.store import TelemetryStore  # noqa: E402

# The 7 metrics checked out on the droplet (others are LFS pointers / tiny stubs).
KALOS_METRICS = [
    "XID_ERRORS",
    "GPU_TEMP",
    "GPU_UTIL",
    "POWER_USAGE",
    "MEMORY_TEMP",
    "MEM_CLOCK",
    "SM_ACTIVE",
]


def _present_sources(data_dir: str) -> dict:
    sources = {}
    for metric in KALOS_METRICS:
        path = os.path.join(data_dir, f"{metric}.csv")
        # Skip absent files and tiny LFS-pointer stubs (a few hundred bytes).
        if os.path.exists(path) and os.path.getsize(path) > 1024:
            sources[metric] = path
    return sources


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--start", required=True, help="inclusive ISO timestamp")
    ap.add_argument("--end", required=True, help="inclusive ISO timestamp")
    ap.add_argument("--downsample", type=int, default=1)
    args = ap.parse_args()

    sources = _present_sources(args.data_dir)
    if not sources:
        print(f"no usable metric CSVs under {args.data_dir}", file=sys.stderr)
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
