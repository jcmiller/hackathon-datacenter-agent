"""Domain value objects — pure, no IO.

The center of the hexagon: identities and records the rest of the system speaks
in. Thin first pass — GPU identity plus the two records that cross the telemetry
/ RCA / agent-memory boundaries. Fault/GpuState models are deferred with the
parked remediation env.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class GpuId:
    """A single physical GPU: a node identifier plus its local index (0-7)."""

    node: str
    index: int

    @property
    def canonical(self) -> str:
        # '#' separates node from index so it never collides with the interior
        # dashes in pod node names (lingjun-pod9-0016).
        return f"{self.node}#{self.index}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.canonical


@dataclass(frozen=True)
class Incident:
    """A job/node failure drawn from the cluster trace."""

    job_id: str
    fail_time: datetime  # tz-aware
    state: str  # FAILED | NODE_FAIL | ...
    type: str | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "fail_time": self.fail_time,
            "state": self.state,
            "type": self.type,
        }


@dataclass(frozen=True)
class SopEntry:
    """A resolved incident in the SOP memory (incident -> resolution)."""

    type: str
    summary: str
    disposition: str
    resolution: str

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "summary": self.summary,
            "disposition": self.disposition,
            "resolution": self.resolution,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SopEntry:
        return cls(
            type=d.get("type", ""),
            summary=d.get("summary", ""),
            disposition=d.get("disposition", ""),
            resolution=d.get("resolution", ""),
        )
