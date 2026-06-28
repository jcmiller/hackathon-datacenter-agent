"""Canonical Kalos/Xid telemetry source resolution — the package source of truth.

The runtime agent tools must read the *canonical* prepared dataset (the wide
``data/utilization/kalos/{METRIC}.csv`` frames), resolved through the Git LFS
cache when the working tree only holds pointers, NOT stale ``data/util/*`` paths
or the ``util_pkl`` CDF artifacts. This module owns:

- the DCGM-field -> canonical-metric map and the canonical metric set;
- the pure path/cache resolution helpers (ported here from ``scripts/lfs_helper``
  so the package never imports ``scripts/``; the CLI re-imports these names);
- ``validate_timeseries_csv`` — fail-loud rejection of anything that is not a
  wide ``Time`` + GPU-columns telemetry frame (a ``.pkl`` CDF artifact, an
  un-materialized LFS pointer, or a header without ``Time`` + >=1 GPU column);
- ``resolve_metric_csv`` — metric name -> validated, readable CSV path, raising
  ``FileNotFoundError`` when the LFS object has not been fetched on this host.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_OID_RE = re.compile(r"^oid sha256:([a-f0-9]{64})$", re.MULTILINE)
_SIZE_RE = re.compile(r"^size ([0-9]+)$", re.MULTILINE)

# Probe size for pointer detection / header read. A Git LFS pointer is ~130 B; a
# real kalos header line (~2344 GPU columns) is far larger, but we only need its
# first field ("Time") and proof of a second column, both well within this slice.
_PROBE_BYTES = 4096


# --- Canonical metric vocabulary -------------------------------------------------

# Real dcgm-exporter field names a production fleet emits -> the canonical kalos
# metric CSV basename. The runtime tools iterate this map so their output stays
# keyed by the DCGM field names while reading the canonical prepared sources.
DCGM_FIELD_TO_METRIC = {
    "DCGM_FI_DEV_POWER_USAGE": "POWER_USAGE",
    "DCGM_FI_DEV_GPU_TEMP": "GPU_TEMP",
}

# All kalos metrics present in the single-owner dataset layout.
CANONICAL_METRICS = (
    "XID_ERRORS",
    "GPU_TEMP",
    "POWER_USAGE",
    "GPU_UTIL",
    "MEMORY_TEMP",
    "MEM_CLOCK",
    "SM_ACTIVE",
)


def _default_repo_dir() -> Path:
    """Repo root: ``$GPUSITTER_REPO_DIR`` or inferred from this file's location.

    The env override is the seam for an installed (non-editable) package and for
    tests, which point it at a tmp repo. From ``src/gpusitter/telemetry`` the
    repo root is three parents up.
    """
    env = os.environ.get("GPUSITTER_REPO_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


# Module-level so tests can ``monkeypatch.setattr(sources, "REPO_DIR", tmp)``;
# resolve_metric_csv reads this at call time (not as a bound default).
REPO_DIR = _default_repo_dir()


# --- Git LFS pointer / cache resolution (ported from scripts/lfs_helper) ---------


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

    Does not require the pointer file to exist in the worktree; deleted working
    paths are resolved through ``git show HEAD:path``. Raises ``FileNotFoundError``
    when the LFS object has not been fetched on this host.
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


def metric_csv_relpath(metric: str) -> str:
    """Repository-relative canonical CSV path for one metric name."""

    return kalos_metric_paths([metric])[0]


# --- Time-series validation + metric resolution ---------------------------------


def validate_timeseries_csv(path: str | os.PathLike[str]) -> str:
    """Return *path* iff it is a wide ``Time`` + GPU-columns telemetry CSV.

    Fails loud (``ValueError``) — never silently returns ``samples:0`` from a
    wrong source — when the resolved artifact is:
    - a ``.pkl`` file (a CDF/distribution artifact, not a time-series frame);
    - an un-materialized Git LFS pointer (text, not the real CSV content);
    - a CSV whose header is not ``Time`` followed by >=1 GPU column.
    """
    p = str(path)
    if p.endswith(".pkl"):
        raise ValueError(
            f"not a time-series CSV: {p} is a pickled CDF/distribution artifact, "
            "not a wide Time+GPU telemetry frame"
        )
    with open(p, newline="") as fh:
        head = fh.read(_PROBE_BYTES)
    if parse_lfs_pointer(head) is not None:
        raise ValueError(
            f"un-materialized Git LFS pointer, not telemetry data: {p} "
            "(fetch the LFS object before reading)"
        )
    first_line = head.splitlines()[0] if head.strip() else ""
    header = next(csv.reader([first_line])) if first_line else []
    if not header or header[0] != "Time" or len(header) < 2:
        raise ValueError(f"not a wide time-series CSV (need 'Time' + >=1 GPU column): {p}")
    return p


def resolve_metric_csv(metric: str, *, repo_dir: str | os.PathLike[str] | None = None) -> str:
    """Resolve a canonical kalos metric name to a validated, readable CSV path.

    Prefers a materialized worktree file, else the Git LFS cache object. Raises
    ``FileNotFoundError`` when the LFS object is not fetched on this host (the
    "raw data not materialized" operational state the agent reports explicitly),
    and ``ValueError`` when the resolved artifact is not a time-series CSV.
    """
    if repo_dir is None:
        repo_dir = REPO_DIR
    rel = metric_csv_relpath(metric)
    path = resolve_data_path(str(repo_dir), rel)
    return validate_timeseries_csv(path)
