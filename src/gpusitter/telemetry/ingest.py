"""Streaming wide->long melt for kalos DCGM CSVs.

Source files are wide frames (row = timestamp, column = GPU, cell = value) up to
~3000 columns and ~80k rows / 1 GB each. We never densify them: the reader
streams one row at a time and emits a :class:`LongRecord` only for *non-empty*
cells. Empty cells mark an idle/unallocated GPU and are skipped — never
zero-filled, since 0 is a meaningful value (e.g. XID 0 = healthy, util 0 = idle
but allocated).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Mapping
from typing import NamedTuple

from .normalize import GpuId, parse_gpu_id


class LongRecord(NamedTuple):
    """One (timestamp, GPU, metric) -> value observation."""

    t: str
    gpu: GpuId
    metric: str
    value: float


def iter_long_records(
    path: str,
    metric: str,
    *,
    time_range: tuple[str, str] | None = None,
    gpus: Iterable[str] | None = None,
    alias: Mapping[str, str] | None = None,
) -> Iterator[LongRecord]:
    """Stream long records from one wide metric CSV.

    Parameters
    ----------
    path:
        Wide CSV; first column header is ``Time``, the rest are GPU columns.
    metric:
        Metric name attached to every emitted record (e.g. ``"GPU_TEMP"``).
    time_range:
        Optional inclusive ``(start, end)`` on the timestamp string. Kalos
        timestamps are ISO with a fixed ``+08:00`` offset, so lexical
        comparison matches chronological order.
    gpus:
        Optional set of canonical GPU id strings to keep; others are skipped.
    alias:
        Optional ``foreign-node -> canonical-node`` map applied while parsing
        column headers, so a foreign namespace folds into the canonical one.
    """
    keep = set(gpus) if gpus is not None else None
    lo, hi = time_range if time_range is not None else (None, None)

    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return
        # Pre-parse column headers once; column 0 is Time.
        col_gpus = [parse_gpu_id(name, alias=alias) for name in header[1:]]
        # Resolve the kept GPU columns ONCE, intersecting the keep-set against
        # parsed headers, instead of re-scanning all ~3000 columns (and recomputing
        # gpu.canonical / the set-membership test) for every data cell. Per-row
        # work then drops from O(all_cols) to O(kept_cols) — the win when a gpus=
        # filter keeps only a slice of the fleet. Index is offset by 1 because
        # column 0 is Time, so kept[i] = (row index, GpuId).
        kept: list[tuple[int, GpuId]] = [
            (i + 1, gpu) for i, gpu in enumerate(col_gpus) if keep is None or gpu.canonical in keep
        ]

        for row in reader:
            if not row:
                continue
            t = row[0]
            if lo is not None and t < lo:
                continue
            if hi is not None and t > hi:
                # Kalos rows are time-ordered (verified on the droplet), so once
                # past the window's end no later row can match — stop scanning
                # instead of reading the rest of a ~1 GB file. Lets a consumer
                # replay a short incident window without a full-file scan.
                break
            width = len(row)
            for idx, gpu in kept:
                # Ragged line: a row shorter than the header has no cell for this
                # column (mirrors the old zip(strict=False) short-stop).
                if idx >= width:
                    continue
                cell = row[idx]
                if cell == "":
                    continue
                yield LongRecord(t=t, gpu=gpu, metric=metric, value=float(cell))
