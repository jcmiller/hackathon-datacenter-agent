"""Bounded, queryable telemetry store — the contract w28 (env) and eku (Xid) read.

Built from streamed long records (:mod:`src.telemetry.ingest`), so the full
~80k x 3000 wide frame is never materialized. ``load`` ingests a *bounded*
slice — a time window, a GPU subset, and/or a downsample stride — appropriate
for replaying an incident or characterizing precursors, not the whole trace.

Query surface (keep stable — two downstream beads depend on it):
    value(metric, gpu, t)  -> float | None        # metric[gpu][t]
    snapshot(gpu, t)       -> {metric: value}     # per-(t,gpu) across metrics
    series(metric, gpu)    -> [(t, value), ...]    # sorted time-series
    window(gpu, t0, t1)    -> {t: {metric: value}} # all metrics over a window
    gpus() / metrics() / timestamps(gpu)
GPU args accept either a canonical id string or a :class:`GpuId`.
"""

from __future__ import annotations

from collections.abc import Mapping

from .ingest import iter_long_records
from .normalize import GpuId

GpuKey = str | GpuId


def _canon(gpu: GpuKey) -> str:
    return gpu.canonical if isinstance(gpu, GpuId) else gpu


class TelemetryStore:
    """In-memory index of ``by_gpu[gpu][t][metric] = value`` over a bounded slice."""

    def __init__(self) -> None:
        # canonical gpu -> timestamp -> metric -> value
        self._by_gpu: dict[str, dict[str, dict[str, float]]] = {}
        self._metrics: set[str] = set()

    # ---- construction -------------------------------------------------------

    @classmethod
    def load(
        cls,
        sources: Mapping[str, str],
        *,
        time_range: tuple[str, str] | None = None,
        gpus: list[str] | None = None,
        downsample: int | None = None,
        alias: Mapping[str, str] | None = None,
    ) -> TelemetryStore:
        """Build a store from ``{metric: csv_path}``.

        ``downsample=N`` keeps every Nth *source row* per metric (stride over the
        timeline) to cap memory while preserving shape. ``time_range``/``gpus``
        filter at read time; ``alias`` normalizes foreign GPU namespaces.
        """
        store = cls()
        for metric, path in sources.items():
            store._metrics.add(metric)
            store._ingest_metric(
                metric,
                path,
                time_range=time_range,
                gpus=gpus,
                downsample=downsample,
                alias=alias,
            )
        return store

    def _ingest_metric(
        self,
        metric: str,
        path: str,
        *,
        time_range: tuple[str, str] | None,
        gpus: list[str] | None,
        downsample: int | None,
        alias: Mapping[str, str] | None,
    ) -> None:
        # Downsample by source timestamp (not by record), so all GPUs at a kept
        # timestamp are kept together. None/<=1 keeps every row.
        stride = downsample if downsample and downsample > 1 else 1
        seen_ts: dict[str, int] = {}
        records = iter_long_records(path, metric, time_range=time_range, gpus=gpus, alias=alias)
        for rec in records:
            if stride > 1:
                rank = seen_ts.get(rec.t)
                if rank is None:
                    rank = len(seen_ts)
                    seen_ts[rec.t] = rank
                if rank % stride != 0:
                    continue
            self._by_gpu.setdefault(rec.gpu.canonical, {}).setdefault(rec.t, {})[metric] = rec.value

    # ---- query surface (stable contract) ------------------------------------

    def value(self, metric: str, gpu: GpuKey, t: str) -> float | None:
        """metric[gpu][t]; ``None`` if that GPU was idle/unobserved then."""
        return self._by_gpu.get(_canon(gpu), {}).get(t, {}).get(metric)

    def snapshot(self, gpu: GpuKey, t: str) -> dict[str, float]:
        """All metrics observed for one GPU at one instant ({} if none)."""
        return dict(self._by_gpu.get(_canon(gpu), {}).get(t, {}))

    def series(self, metric: str, gpu: GpuKey) -> list[tuple[str, float]]:
        """Sorted ``[(t, value)]`` for one metric on one GPU."""
        per_t = self._by_gpu.get(_canon(gpu), {})
        out = [(t, m[metric]) for t, m in per_t.items() if metric in m]
        out.sort(key=lambda kv: kv[0])
        return out

    def window(self, gpu: GpuKey, t0: str, t1: str) -> dict[str, dict[str, float]]:
        """All metrics for one GPU across inclusive ``[t0, t1]``, keyed by time."""
        per_t = self._by_gpu.get(_canon(gpu), {})
        return {t: dict(metrics) for t, metrics in sorted(per_t.items()) if t0 <= t <= t1}

    def gpus(self) -> list[str]:
        """Canonical ids of every GPU with at least one observation."""
        return sorted(self._by_gpu)

    def metrics(self) -> list[str]:
        return sorted(self._metrics)

    def timestamps(self, gpu: GpuKey) -> list[str]:
        """Sorted timestamps at which this GPU has any observation."""
        return sorted(self._by_gpu.get(_canon(gpu), {}))
