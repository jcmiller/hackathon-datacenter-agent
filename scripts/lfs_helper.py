#!/usr/bin/env python3
"""CLI for resolving large Git LFS data files without checking them out.

The resolution logic now lives in :mod:`gpusitter.telemetry.sources` (the package
source of truth — the runtime tools must not import ``scripts/``). This module
re-exports those helpers and keeps the ``resolve`` / ``status`` / ``kalos-status``
CLI, so existing callers and ``tests/test_lfs_helper.py`` are unchanged.

The Acme raw telemetry repository is too large to keep both checked-out files and
the Git LFS cache on the droplet. The helpers support the three states we see:

- a real working-tree file is checked out;
- a small Git LFS pointer file is checked out;
- the working-tree path was deleted, but the pointer is still available in Git.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from gpusitter.telemetry.sources import (  # re-exported for the CLI + existing callers
    CANONICAL_METRICS,
    LfsPointer,
    RawDataStatus,
    get_lfs_cache_path,
    kalos_metric_paths,
    lfs_cache_path,
    parse_lfs_pointer,
    raw_data_status,
    read_lfs_pointer,
    resolve_data_path,
)

__all__ = [
    "LfsPointer",
    "RawDataStatus",
    "parse_lfs_pointer",
    "read_lfs_pointer",
    "lfs_cache_path",
    "get_lfs_cache_path",
    "resolve_data_path",
    "raw_data_status",
    "kalos_metric_paths",
]


def _print_status(statuses: list[RawDataStatus], as_json: bool) -> None:
    if as_json:
        print(json.dumps([asdict(status) for status in statuses], indent=2, sort_keys=True))
        return
    for status in statuses:
        size = "-" if status.size is None else str(status.size)
        disk = "-" if status.bytes_on_disk is None else str(status.bytes_on_disk)
        print(
            f"{status.file_path}\tworktree={status.working_state}"
            f"\tcache={status.cache_state}\tsize={size}\tdisk={disk}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    resolve = sub.add_parser("resolve", help="print a readable path for one file")
    resolve.add_argument("repo_dir")
    resolve.add_argument("file_path")
    resolve.add_argument("--cache-only", action="store_true", help="require the LFS cache object")

    status = sub.add_parser("status", help="show worktree/cache state for files")
    status.add_argument("repo_dir")
    status.add_argument("file_path", nargs="+")
    status.add_argument("--json", action="store_true")

    kalos = sub.add_parser("kalos-status", help="status for common kalos raw metrics")
    kalos.add_argument("repo_dir")
    kalos.add_argument("--json", action="store_true")
    kalos.add_argument("--metrics", nargs="+", default=list(CANONICAL_METRICS))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "resolve":
        path = (
            get_lfs_cache_path(args.repo_dir, args.file_path)
            if args.cache_only
            else resolve_data_path(args.repo_dir, args.file_path)
        )
        print(path)
        return 0
    if args.command == "status":
        _print_status([raw_data_status(args.repo_dir, path) for path in args.file_path], args.json)
        return 0
    if args.command == "kalos-status":
        statuses = [
            raw_data_status(args.repo_dir, path) for path in kalos_metric_paths(args.metrics)
        ]
        _print_status(statuses, args.json)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
