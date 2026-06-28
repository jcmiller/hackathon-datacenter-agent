#!/usr/bin/env python3
"""Resolve large Git LFS data files without checking them out.

The Acme raw telemetry repository is too large to keep both checked-out files
and the Git LFS cache on the droplet. These helpers support the three states we
actually see there:

- a real working-tree file is checked out;
- a small Git LFS pointer file is checked out;
- the working-tree path was deleted, but the pointer is still available in Git.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

_OID_RE = re.compile(r"^oid sha256:([a-f0-9]{64})$", re.MULTILINE)
_SIZE_RE = re.compile(r"^size ([0-9]+)$", re.MULTILINE)


@dataclass(frozen=True)
class LfsPointer:
    """Git LFS pointer metadata."""

    oid: str
    size: int | None


@dataclass(frozen=True)
class RawDataStatus:
    """Resolved status for one repository-relative data path."""

    file_path: str
    working_state: str
    cache_state: str
    resolved_path: str | None
    oid: str | None
    size: int | None
    bytes_on_disk: int | None


def _run_git_show(repo_dir: Path, file_path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "show", f"HEAD:{file_path}"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise FileNotFoundError(
            f"Pointer file not found in working tree or HEAD: {file_path}"
        ) from exc


def parse_lfs_pointer(content: str) -> LfsPointer | None:
    """Return pointer metadata if *content* is a Git LFS pointer."""

    oid_match = _OID_RE.search(content)
    if not oid_match:
        return None
    size_match = _SIZE_RE.search(content)
    size = int(size_match.group(1)) if size_match else None
    return LfsPointer(oid=oid_match.group(1), size=size)


def _read_small_text(path: Path, limit: int = 4096) -> str | None:
    """Read a small candidate pointer file as text, or return None."""

    if not path.exists() or not path.is_file() or path.stat().st_size > limit:
        return None
    try:
        return path.read_text()
    except UnicodeDecodeError:
        return None


def read_lfs_pointer(repo_dir: str | os.PathLike[str], file_path: str) -> LfsPointer:
    """Read pointer metadata from the worktree or from ``git show HEAD:path``."""

    repo = Path(repo_dir)
    worktree_path = repo / file_path
    content = _read_small_text(worktree_path)
    pointer = parse_lfs_pointer(content) if content is not None else None
    if pointer is not None:
        return pointer

    pointer = parse_lfs_pointer(_run_git_show(repo, file_path))
    if pointer is None:
        raise ValueError(f"File is not a Git LFS pointer in HEAD: {file_path}")
    return pointer


def lfs_cache_path(repo_dir: str | os.PathLike[str], pointer: LfsPointer) -> Path:
    """Return the expected cache path for a pointer."""

    repo = Path(repo_dir)
    return repo / ".git" / "lfs" / "objects" / pointer.oid[:2] / pointer.oid[2:4] / pointer.oid


def get_lfs_cache_path(repo_dir: str, file_path: str) -> str:
    """Resolve the cached Git LFS object for ``file_path``.

    This preserves the original public helper API. It no longer requires the
    pointer file to exist in the worktree; deleted working paths are resolved
    through ``git show HEAD:path``.
    """

    pointer = read_lfs_pointer(repo_dir, file_path)
    cache_path = lfs_cache_path(repo_dir, pointer)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"LFS object cache not found for {file_path}: {cache_path}. "
            "Run 'git lfs fetch' to download it."
        )
    return str(cache_path)


def resolve_data_path(repo_dir: str, file_path: str) -> str:
    """Return a readable path for data, preferring a materialized worktree file."""

    worktree_path = Path(repo_dir) / file_path
    content = _read_small_text(worktree_path)
    if worktree_path.exists() and parse_lfs_pointer(content or "") is None:
        return str(worktree_path)
    return get_lfs_cache_path(repo_dir, file_path)


def raw_data_status(repo_dir: str, file_path: str) -> RawDataStatus:
    """Return management status for a raw data file."""

    repo = Path(repo_dir)
    worktree_path = repo / file_path
    content = _read_small_text(worktree_path)
    pointer = parse_lfs_pointer(content or "") if content is not None else None

    if worktree_path.exists() and pointer is None:
        return RawDataStatus(
            file_path=file_path,
            working_state="materialized",
            cache_state="unknown",
            resolved_path=str(worktree_path),
            oid=None,
            size=None,
            bytes_on_disk=worktree_path.stat().st_size,
        )

    if pointer is None:
        pointer = read_lfs_pointer(repo, file_path)

    cache_path = lfs_cache_path(repo, pointer)
    cache_state = "present" if cache_path.exists() else "missing"
    working_state = "pointer" if worktree_path.exists() else "missing"
    return RawDataStatus(
        file_path=file_path,
        working_state=working_state,
        cache_state=cache_state,
        resolved_path=str(cache_path) if cache_path.exists() else None,
        oid=pointer.oid,
        size=pointer.size,
        bytes_on_disk=cache_path.stat().st_size if cache_path.exists() else None,
    )


def kalos_metric_paths(metrics: Iterable[str]) -> list[str]:
    """Return Acme kalos repository-relative CSV paths for metric names."""

    return [f"data/utilization/kalos/{metric}.csv" for metric in metrics]


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
    kalos.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "XID_ERRORS",
            "GPU_TEMP",
            "POWER_USAGE",
            "GPU_UTIL",
            "MEMORY_TEMP",
            "MEM_CLOCK",
            "SM_ACTIVE",
        ],
    )
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
        statuses = [raw_data_status(args.repo_dir, path) for path in kalos_metric_paths(args.metrics)]
        _print_status(statuses, args.json)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
