"""GPU-id normalization.

Kalos DCGM metrics name the same physical fleet two ways: 6 of 7 metrics use an
IP-based scheme (``172.31.15.112-6`` = ``<node-ip>-<gpu-index>``), while
SM_ACTIVE uses a pod-based scheme (``lingjun-pod9-0016-3`` =
``<pod-node>-<gpu-index>``). Both end in ``-<index>`` where index is the local
GPU 0-7.

A :class:`GpuId` is the canonical handle. The trace carries no deterministic
IP<->pod map (different monitoring exporters, non-identical GPU sets), so a
cross-namespace join is only possible when an external *alias map*
(``foreign-node -> canonical-node``) is supplied. Within a single namespace
(the IP metrics) ids already line up and need no alias.
"""

from __future__ import annotations

from typing import Mapping, Optional

# GpuId is the canonical domain identity; re-exported here so telemetry's own
# modules can keep importing it from .normalize.
from ..domain.models import GpuId

__all__ = ["GpuId", "parse_gpu_id"]


def parse_gpu_id(raw: str, alias: Optional[Mapping[str, str]] = None) -> GpuId:
    """Parse a raw wide-CSV column name into a canonical :class:`GpuId`.

    Splits on the *last* ``-`` (the GPU index); everything before is the node.
    When ``alias`` is given, the node is rewritten through it first, folding a
    foreign namespace (e.g. pod names) into the canonical one so the same
    physical GPU resolves to one :class:`GpuId`.
    """
    node, _, idx = raw.rpartition("-")
    if not node or not idx:
        raise ValueError(f"unrecognized GPU id (expected '<node>-<index>'): {raw!r}")
    try:
        index = int(idx)
    except ValueError as exc:
        raise ValueError(f"GPU index is not an integer in {raw!r}") from exc
    if alias is not None:
        node = alias.get(node, node)
    return GpuId(node=node, index=index)
