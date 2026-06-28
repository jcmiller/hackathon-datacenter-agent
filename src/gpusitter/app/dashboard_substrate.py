"""Reproducible REAL dashboard data substrate from Kalos Xid onsets (bead t7p).

The dashboard (incident feed, fleet heatmap, per-incident telemetry) needs a
small, portable artifact it can serve without scanning the ~80 GB raw kalos
telemetry per click. This module BUILDS that artifact from the canonical sources,
deterministically and with provenance, so the demo serves *derived real data*
rather than hand-authored or stale fixtures.

Honesty / methodology (the load-bearing correction):
- Incidents are **edge-detected Xid onsets** — a non-fault -> fault transition
  streamed through :func:`gpusitter.rca.job_join.stream_xid_onset_records`, the
  same empty-aware detector detection/eku use. They are NOT latched cumulative
  Xid-gauge snapshots. This is what stops the debunked "Aug-29 13:57 / 882-GPU
  simultaneous cascade" (a latched-state + window-edge artifact) from ever being
  reconstructed: at any late timestamp almost every cumulatively-faulted GPU
  reads nonzero, but only a true non-fault->fault transition counts as an onset.
- The fleet snapshot shows real per-GPU telemetry at the hero burst center, but a
  cell is rendered "fault" ONLY if it is an edge-detected member of the hero
  burst cohort. Every other GPU's status is derived from UTILIZATION, never from
  its latched Xid value, so the heatmap cannot resurrect the cumulative cascade.

The build resolves canonical kalos CSVs through
:mod:`gpusitter.telemetry.sources` (NOT stale ``data/util`` paths), runs on the
droplet where the raw trace is materialized, and emits a compact substrate plus a
``manifest.json`` recording source, generation command, input paths/OIDs, the
event window, and that the telemetry is real. Pure helpers take CSV paths so the
edge-detection / latched-state semantics are unit-tested off-droplet against tiny
synthetic frames.
"""

from __future__ import annotations

import csv
import json
import subprocess
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from ..rca.job_join import stream_xid_onset_records
from ..telemetry import sources

# Committed artifact lives OUTSIDE the gitignored data/ tree, beside the jds
# monitor fixture, so the dashboard ships it without the raw trace.
SUBSTRATE_DIR = Path(__file__).parent / "fixtures" / "dashboard_substrate"

# Kalos DCGM sample grid (verified 15s).
SAMPLE_SECONDS = 15.0

# A burst is >= this many edge-detected onsets coincident within COHORT_WINDOW_S.
# (Aug-17 06:00 real burst ~116; isolated faults are 1-3.)
BURST_MIN_ONSETS = 40
# Half-width of the coincidence window used both to call bursts and to compute an
# incident's correlated cohort. 45s spans the two adjacent 15s burst samples
# (e.g. 06:00:15 + 06:00:30) with zero-baseline neighbours on each side.
COHORT_WINDOW_S = 45.0

# Per-incident telemetry window (real series), center -/+ these seconds.
TELEMETRY_PRE_S = 180.0
TELEMETRY_POST_S = 180.0

# Selection caps — keep the artifact small and the feed legible.
MAX_CASCADE_INCIDENTS = 6
MAX_ISOLATED_INCIDENTS = 4
MAX_CORRELATED_LISTED = 8

# Canonical NVIDIA/DCGM Xid meanings + dashboard severity. Ported from eku's
# scripts/characterize_xid table (the package must not import scripts/). UI
# severity collapses the fault taxonomy to the two badges the feed renders.
XID_LABELS: dict[int, str] = {
    43: "GPU stopped processing",
    31: "GPU memory page fault",
    45: "Preemptive cleanup (robust channel)",
    94: "Contained ECC error",
    95: "Uncontained ECC error",
    48: "Double-bit ECC error",
    79: "GPU has fallen off the bus",
}
_CRIT_CODES = frozenset({31, 48, 95, 79})  # memory-fault / uncontained / off-bus


def xid_label(code: int) -> str:
    return XID_LABELS.get(code, f"Xid {code}")


def xid_severity(code: int) -> str:
    return "crit" if code in _CRIT_CODES else "warn"


# --- id helpers: canonical 'node#idx' (domain) <-> 'node-idx' (dashboard) -------


def _node_of(canonical: str) -> str:
    return canonical.split("#", 1)[0]


def _idx_of(canonical: str) -> int:
    return int(canonical.rsplit("#", 1)[1])


def _dash_id(canonical: str) -> str:
    """'172.31.15.220#4' -> '172.31.15.220-4' (the dashboard correlated-id form)."""
    node, _, idx = canonical.rpartition("#")
    return f"{node}-{idx}"


# --- onset records + burst / cohort detection (pure) ----------------------------


@dataclass(frozen=True)
class Onset:
    """One edge-detected Xid onset: time, canonical GPU id, fault code."""

    t: datetime
    gpu: str
    code: int


def load_onsets(xid_csv: str) -> list[Onset]:
    """Edge-detected onsets from a wide XID CSV, time-sorted (canonical detector)."""
    return [
        Onset(t=t, gpu=gpu, code=int(code)) for (t, gpu, code) in stream_xid_onset_records(xid_csv)
    ]


def _sample_counts(onsets: Sequence[Onset]) -> dict[datetime, int]:
    return dict(Counter(o.t for o in onsets))


def cohort(onsets: Sequence[Onset], center: datetime, window_s: float) -> list[Onset]:
    """All onsets within +/- ``window_s`` of ``center`` (the correlated cohort).

    ``onsets`` must be time-sorted; located with two binary searches so this is
    cheap to call once per candidate incident.
    """
    times = [o.t for o in onsets]
    lo = bisect_left(times, center - timedelta(seconds=window_s))
    hi = bisect_right(times, center + timedelta(seconds=window_s))
    return list(onsets[lo:hi])


@dataclass
class Burst:
    """A correlated onset burst: center sample + union-deduped membership."""

    center: datetime
    n_onsets: int
    members: list[Onset]
    gpus: set[str] = field(default_factory=set)
    nodes: set[str] = field(default_factory=set)
    codes: Counter = field(default_factory=Counter)


def detect_bursts(
    onsets: Sequence[Onset],
    *,
    min_onsets: int | None = None,
    window_s: float = COHORT_WINDOW_S,
) -> list[Burst]:
    """Correlated bursts: each per-sample peak whose cohort >= ``min_onsets``.

    A peak is a 15s sample that is a local maximum of cohort size (so the two
    adjacent samples of one physical burst collapse to a single Burst centered on
    the denser sample). Latched faults never appear here — only edge-detected
    onsets feed ``onsets`` — so a cumulative snapshot of N faulted GPUs at a late
    timestamp yields NO burst.

    ``min_onsets`` defaults to the module ``BURST_MIN_ONSETS`` read at call time
    (so it stays overridable/patchable), not bound at definition.
    """
    if min_onsets is None:
        min_onsets = BURST_MIN_ONSETS
    counts = _sample_counts(onsets)
    samples = sorted(counts)
    cohorts: dict[datetime, list[Onset]] = {s: cohort(onsets, s, window_s) for s in samples}
    bursts: list[Burst] = []
    claimed: set[datetime] = set()
    # Greedy by descending cohort size so the densest sample wins its neighbours.
    for s in sorted(samples, key=lambda x: (-len(cohorts[x]), x)):
        if s in claimed or len(cohorts[s]) < min_onsets:
            continue
        members = cohorts[s]
        for m in members:
            claimed.add(m.t)
        bursts.append(
            Burst(
                center=s,
                n_onsets=len(members),
                members=members,
                gpus={m.gpu for m in members},
                nodes={_node_of(m.gpu) for m in members},
                codes=Counter(m.code for m in members),
            )
        )
    bursts.sort(key=lambda b: (-b.n_onsets, b.center))
    return bursts


# --- metric CSV readers (pure; streaming, bounded) ------------------------------


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def snapshot_row(metric_csv: str, t_target: datetime) -> dict[str, float]:
    """Per-GPU value at the sample == ``t_target`` (canonical id -> float).

    Streams to the matching row and stops; empty cells are omitted. Used for the
    fleet heatmap snapshot (one instant).
    """
    out: dict[str, float] = {}
    with open(metric_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        gpus = [_canon_header(h) for h in header[1:]]
        for row in reader:
            if not row:
                continue
            if _parse_ts(row[0]) != t_target:
                continue
            for i, cell in enumerate(row[1:]):
                if cell != "" and i < len(gpus):
                    out[gpus[i]] = float(cell)
            break
    return out


def read_incident_windows(
    metric_csv: str, targets: Mapping[str, tuple[datetime, datetime]]
) -> dict[str, list[tuple[datetime, float]]]:
    """For each target GPU, its ``[(t, value)]`` series inside its own window.

    One streaming pass extracts every requested GPU's bounded window
    simultaneously (targets may center on different days). Empty cells skipped.
    """
    series: dict[str, list[tuple[datetime, float]]] = {g: [] for g in targets}
    if not targets:
        return series
    lo = min(w[0] for w in targets.values())
    hi = max(w[1] for w in targets.values())
    with open(metric_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {_canon_header(h): i for i, h in enumerate(header[1:])}
        cols = {g: col[g] for g in targets if g in col}
        for row in reader:
            if not row:
                continue
            t = _parse_ts(row[0])
            if t < lo or t > hi:
                continue
            for g, ci in cols.items():
                g_lo, g_hi = targets[g]
                if not (g_lo <= t <= g_hi):
                    continue
                cell = row[ci + 1] if ci + 1 < len(row) else ""
                if cell != "":
                    series[g].append((t, float(cell)))
    return series


def _canon_header(raw: str) -> str:
    """Wide-CSV column name -> canonical 'node#idx' (mirrors parse_gpu_id)."""
    node, _, idx = raw.rpartition("-")
    if not node or not idx:
        raise ValueError(f"unrecognized GPU column id (need '<node>-<idx>'): {raw!r}")
    return f"{node}#{int(idx)}"


# --- substrate assembly ---------------------------------------------------------


def _incident_record(
    inc_id: str,
    onset: Onset,
    *,
    onsets: Sequence[Onset],
    hero: bool,
) -> dict:
    """One dashboard incident from an onset + its real correlated cohort."""
    coh = cohort(onsets, onset.t, COHORT_WINDOW_S)
    coh_gpus = [o.gpu for o in coh]
    node = _node_of(onset.gpu)
    node_cofaults = sum(1 for g in coh_gpus if _node_of(g) == node) - 1
    correlated = [_dash_id(g) for g in sorted(set(coh_gpus))][:MAX_CORRELATED_LISTED]
    # ensure the incident's own GPU is listed first
    own = _dash_id(onset.gpu)
    if own in correlated:
        correlated.remove(own)
    correlated = [own] + correlated[: MAX_CORRELATED_LISTED - 1]
    return {
        "id": inc_id,
        "ts": onset.t.isoformat(),
        "gpu": {"node": node, "idx": _idx_of(onset.gpu)},
        "xid": onset.code,
        "xidLabel": xid_label(onset.code),
        "severity": xid_severity(onset.code),
        "nodeCofaults": max(0, node_cofaults),
        "correlatedCount": len(coh),
        "correlated": correlated,
        "hero": hero,
        "state": "triaging",
    }


def select_incidents(onsets: Sequence[Onset], bursts: Sequence[Burst]) -> list[dict]:
    """Deterministic incident set: hero-cascade members + isolated faults.

    No RNG; every tiebreak is a stable sort, so the same onsets yield the same
    incident list byte-for-byte.
    """
    if not onsets:
        return []
    incidents: list[dict] = []
    used: set[str] = set()
    seq = 1

    burst_members: set[str] = {m.gpu for b in bursts for m in b.members}

    hero_gpu: str | None = None
    if bursts:
        hero = bursts[0]
        # cascade members sorted by (node, idx); first is THE hero card
        members = sorted(hero.members, key=lambda o: (_node_of(o.gpu), _idx_of(o.gpu)))
        hero_gpu = members[0].gpu if members else None
        for o in members[:MAX_CASCADE_INCIDENTS]:
            incidents.append(
                _incident_record(f"INC-{seq:03d}", o, onsets=onsets, hero=(o.gpu == hero_gpu))
            )
            used.add(o.gpu)
            seq += 1

    # isolated onsets: those NOT part of any correlated burst, one per distinct
    # code, earliest first (so the feed spreads across fault types and days).
    isolated = [o for o in onsets if o.gpu not in burst_members]
    by_code: dict[int, Onset] = {}
    for o in sorted(isolated, key=lambda o: o.t):
        if o.gpu in used:
            continue
        by_code.setdefault(o.code, o)
    for o in sorted(by_code.values(), key=lambda o: o.t)[:MAX_ISOLATED_INCIDENTS]:
        incidents.append(_incident_record(f"INC-{seq:03d}", o, onsets=onsets, hero=False))
        used.add(o.gpu)
        seq += 1
    return incidents


def build_fleet_snapshot(
    *,
    temp_at: Mapping[str, float],
    util_at: Mapping[str, float],
    hero: Burst | None,
    center: datetime,
) -> dict:
    """Real per-GPU snapshot at ``center``; fault cells = hero onset members only.

    A GPU's ``status`` derives from utilization, NOT its latched Xid value: only
    an edge-detected hero-burst member is "fault". This is the latched-state guard
    at the heatmap layer.
    """
    onset_code: dict[str, int] = {}
    if hero is not None:
        for m in hero.members:
            onset_code[m.gpu] = m.code
    gpus = sorted(set(temp_at) | set(util_at) | set(onset_code))
    cells = []
    for g in gpus:
        temp = temp_at.get(g)
        util = util_at.get(g)
        if g in onset_code:
            status, xid = "fault", onset_code[g]
        elif util is None or util <= 0.0:
            status, xid = "idle", 0
        else:
            status, xid = "active", 0
        cells.append(
            {
                "node": _node_of(g),
                "idx": _idx_of(g),
                "temp": temp if temp is not None else 0.0,
                "util": util if util is not None else 0.0,
                "xid": xid,
                "status": status,
            }
        )
    faulted = len(onset_code)
    nodes = len({_node_of(g) for g in onset_code})
    return {
        "ts": center.isoformat(),
        "nodes": nodes,
        "faulted": faulted,
        "cells": cells,
    }


def _telemetry_records(
    incidents: Sequence[dict],
    onset_by_id: Mapping[str, Onset],
    temp_csv: str,
    power_csv: str,
    util_csv: str,
) -> dict[str, dict]:
    """Per-incident real ``temp/power/util`` window series (one pass per metric)."""
    targets = {
        onset_by_id[inc["id"]].gpu: (
            onset_by_id[inc["id"]].t - timedelta(seconds=TELEMETRY_PRE_S),
            onset_by_id[inc["id"]].t + timedelta(seconds=TELEMETRY_POST_S),
        )
        for inc in incidents
    }
    temp = read_incident_windows(temp_csv, targets)
    power = read_incident_windows(power_csv, targets)
    util = read_incident_windows(util_csv, targets)

    def _ser(rows: list[tuple[datetime, float]]) -> list[list]:
        return [[t.isoformat(), v] for t, v in rows]

    out: dict[str, dict] = {}
    for inc in incidents:
        o = onset_by_id[inc["id"]]
        out[inc["id"]] = {
            "gpu": _dash_id(o.gpu),
            "centerTs": o.t.isoformat(),
            "series": {
                "temp": _ser(temp.get(o.gpu, [])),
                "power": _ser(power.get(o.gpu, [])),
                "util": _ser(util.get(o.gpu, [])),
            },
        }
    return out


@dataclass
class Substrate:
    """The complete dashboard substrate plus its provenance manifest."""

    meta: dict
    fleet: dict
    incidents: list[dict]
    telemetry: dict[str, dict]
    manifest: dict


def assemble_substrate(
    *,
    xid_csv: str,
    temp_csv: str,
    power_csv: str,
    util_csv: str,
    total_gpus: int,
    source: str,
    generation_command: str,
    input_meta: list[dict] | None = None,
    git_rev: str | None = None,
) -> Substrate:
    """Build the substrate from resolved canonical CSV paths (pure orchestration).

    Separated from :func:`build_substrate` (which resolves the paths via
    :mod:`gpusitter.telemetry.sources`) so the whole pipeline is unit-tested with
    tiny synthetic CSVs and no LFS/droplet dependency.
    """
    onsets = load_onsets(xid_csv)
    bursts = detect_bursts(onsets)
    hero = bursts[0] if bursts else None
    incidents = select_incidents(onsets, bursts)

    onset_by_gpu = {o.gpu: o for o in onsets}
    onset_by_id = {
        inc["id"]: onset_by_gpu[f"{inc['gpu']['node']}#{inc['gpu']['idx']}"] for inc in incidents
    }

    center = hero.center if hero is not None else (onsets[0].t if onsets else None)
    if center is not None:
        temp_at = snapshot_row(temp_csv, center)
        util_at = snapshot_row(util_csv, center)
    else:
        temp_at, util_at = {}, {}
    fleet = (
        build_fleet_snapshot(temp_at=temp_at, util_at=util_at, hero=hero, center=center)
        if center is not None
        else {"ts": None, "nodes": 0, "faulted": 0, "cells": []}
    )

    telemetry = _telemetry_records(incidents, onset_by_id, temp_csv, power_csv, util_csv)

    window = (
        [
            (center - timedelta(minutes=20)).isoformat(),
            (center + timedelta(minutes=20)).isoformat(),
        ]
        if center is not None
        else None
    )
    meta = {
        "window": window,
        "cascadeTs": center.isoformat() if center is not None else None,
        "totalGpus": total_gpus,
        "faulted": hero.n_onsets if hero is not None else 0,
        "nodesAffected": len(hero.nodes) if hero is not None else 0,
        "source": source,
    }
    manifest = {
        "schema": "gpusitter.dashboard_substrate/v1",
        "kind": "real",
        "telemetryKind": "real",
        "source": source,
        "generationCommand": generation_command,
        "generatedFromGitRev": git_rev,
        "inputs": input_meta or [],
        "samplePeriodSeconds": SAMPLE_SECONDS,
        "onsetTotal": len(onsets),
        "burstCount": len(bursts),
        "hero": (
            {
                "center": hero.center.isoformat(),
                "onsets": hero.n_onsets,
                "gpus": len(hero.gpus),
                "nodes": len(hero.nodes),
                "codes": dict(sorted(hero.codes.items(), key=lambda kv: -kv[1])),
            }
            if hero is not None
            else None
        ),
        "incidentCount": len(incidents),
        "eventWindow": window,
        "note": (
            "Derived REAL substrate: incidents are edge-detected Xid onsets "
            "(empty-aware), NOT latched cumulative Xid-gauge snapshots. Fleet "
            "fault cells are hero-burst onset members only; other GPUs' status is "
            "from utilization, never latched Xid. Raw kalos data is unmodified."
        ),
    }
    return Substrate(
        meta=meta,
        fleet=fleet,
        incidents=incidents,
        telemetry=telemetry,
        manifest=manifest,
    )


def substrate_available(substrate_dir: str | Path = SUBSTRATE_DIR) -> bool:
    """True when a complete committed substrate artifact exists at ``substrate_dir``.

    Requires all four top-level documents; the telemetry directory is optional
    (an incident-less substrate is still a valid, if empty, real artifact).
    """
    d = Path(substrate_dir)
    return all(
        (d / n).exists() for n in ("manifest.json", "meta.json", "fleet.json", "incidents.json")
    )


def load_substrate(substrate_dir: str | Path = SUBSTRATE_DIR) -> Substrate:
    """Read a committed substrate artifact back into a :class:`Substrate`.

    Inverse of :func:`write_substrate`: loads meta/fleet/incidents/manifest plus
    every ``telemetry/INC-*.json`` window. Pure file IO — no LFS, no droplet, no
    re-derivation — so the dashboard serves the real artifact off-droplet.
    """
    d = Path(substrate_dir)

    def _read(name: str) -> object:
        return json.loads((d / name).read_text())

    telemetry: dict[str, dict] = {}
    tel_dir = d / "telemetry"
    if tel_dir.exists():
        for f in sorted(tel_dir.glob("INC-*.json")):
            telemetry[f.stem] = json.loads(f.read_text())
    return Substrate(
        meta=_read("meta.json"),
        fleet=_read("fleet.json"),
        incidents=_read("incidents.json"),
        telemetry=telemetry,
        manifest=_read("manifest.json"),
    )


def _git_rev(repo_dir: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _input_meta(repo_dir: str, metrics: Iterable[str]) -> list[dict]:
    """Provenance for each canonical input: rel path + LFS status/OID when known."""
    out = []
    for m in metrics:
        rel = sources.metric_csv_relpath(m)
        entry: dict = {"metric": m, "path": rel}
        try:
            st = sources.raw_data_status(repo_dir, rel)
            entry.update(
                {
                    "workingState": st.working_state,
                    "cacheState": st.cache_state,
                    "oid": st.oid,
                    "size": st.size,
                }
            )
        except Exception as exc:  # noqa: BLE001 — provenance is best-effort
            entry["statusError"] = str(exc)
        out.append(entry)
    return out


def build_substrate(*, repo_dir: str | Path | None = None) -> Substrate:
    """Resolve canonical kalos sources and build the real substrate (droplet path).

    Reads XID_ERRORS / GPU_TEMP / POWER_USAGE / GPU_UTIL through
    :func:`gpusitter.telemetry.sources.resolve_metric_csv` — the single-owner
    canonical resolver — so it never touches stale ``data/util`` paths. Raises
    ``FileNotFoundError`` when the LFS objects are not materialized on this host
    (the honest "raw data not available off-droplet" state).
    """
    rd = str(repo_dir) if repo_dir is not None else str(sources.REPO_DIR)
    metrics = ("XID_ERRORS", "GPU_TEMP", "POWER_USAGE", "GPU_UTIL")
    xid_csv = sources.resolve_metric_csv("XID_ERRORS", repo_dir=rd)
    temp_csv = sources.resolve_metric_csv("GPU_TEMP", repo_dir=rd)
    power_csv = sources.resolve_metric_csv("POWER_USAGE", repo_dir=rd)
    util_csv = sources.resolve_metric_csv("GPU_UTIL", repo_dir=rd)
    total_gpus = _count_header_gpus(xid_csv)
    return assemble_substrate(
        xid_csv=xid_csv,
        temp_csv=temp_csv,
        power_csv=power_csv,
        util_csv=util_csv,
        total_gpus=total_gpus,
        source="AcmeTrace Kalos (Shanghai AI Lab), Aug 2023 — edge-detected Xid onsets",
        generation_command="python scripts/build_dashboard_substrate.py",
        input_meta=_input_meta(rd, metrics),
        git_rev=_git_rev(rd),
    )


def _count_header_gpus(metric_csv: str) -> int:
    with open(metric_csv, newline="") as fh:
        header = next(csv.reader(fh))
    return max(0, len(header) - 1)


def write_substrate(substrate: Substrate, out_dir: str | Path) -> dict:
    """Write the substrate to ``out_dir`` (meta/fleet/incidents/telemetry/manifest).

    Deterministic, sorted-key JSON so re-running on identical inputs produces a
    byte-identical tree. Returns a small summary.
    """
    out = Path(out_dir)
    (out / "telemetry").mkdir(parents=True, exist_ok=True)

    def _dump(obj, path: Path) -> None:
        path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

    _dump(substrate.meta, out / "meta.json")
    _dump(substrate.fleet, out / "fleet.json")
    _dump(substrate.incidents, out / "incidents.json")
    _dump(substrate.manifest, out / "manifest.json")
    # prune stale telemetry files so the tree always matches the incident set
    tel_dir = out / "telemetry"
    keep = set(substrate.telemetry)
    for f in tel_dir.glob("INC-*.json"):
        if f.stem not in keep:
            f.unlink()
    for inc_id, rec in substrate.telemetry.items():
        _dump(rec, tel_dir / f"{inc_id}.json")
    return {
        "out_dir": str(out),
        "incidents": len(substrate.incidents),
        "fleet_cells": len(substrate.fleet.get("cells", [])),
        "telemetry_files": len(substrate.telemetry),
        "hero": substrate.manifest.get("hero"),
    }
