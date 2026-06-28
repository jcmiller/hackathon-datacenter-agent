"""Dashboard fixtures pin the corrected Aug-17 06:00 hero snapshot (bead 69c).

These fixtures are *derived* demo data: raw Kalos is untouched, but the dashboard
must present the verified Aug-17 06:00 correlated burst (116 GPUs / 74 nodes,
Xid 43-dominant) — NOT the debunked Aug-29/882 latched-state artifact. Two
tracked copies must stay byte-identical: the Vite source (`dashboard/public`)
and the copy the FastAPI backend serves (`src/gpusitter/app/dashboard`).
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "dashboard" / "public" / "fixtures"
APP = ROOT / "src" / "gpusitter" / "app" / "dashboard" / "fixtures"
COPIES = (PUB, APP)

FAULTED = 116
NODES = 74
HERO_TS = "2023-08-17 06:00:30+08:00"


def _load(base: Path, name: str):
    return json.loads((base / name).read_text())


@pytest.mark.parametrize("base", COPIES, ids=lambda p: p.parent.name)
def test_meta_is_aug17_hero(base):
    meta = _load(base, "meta.json")
    assert meta["cascadeTs"] == HERO_TS
    assert meta["window"][0].startswith("2023-08-17")
    assert meta["faulted"] == FAULTED
    assert meta["nodesAffected"] == NODES
    assert "Aug-17" in meta["source"]


@pytest.mark.parametrize("base", COPIES, ids=lambda p: p.parent.name)
def test_fleet_snapshot_matches_headline(base):
    fleet = _load(base, "fleet.json")
    assert fleet["ts"] == HERO_TS
    fault_cells = [c for c in fleet["cells"] if c["status"] == "fault"]
    # The declared totals must equal what the cells actually contain — guards
    # against a relabeled header over a still-882 cell array.
    assert fleet["faulted"] == FAULTED == len(fault_cells)
    assert fleet["nodes"] == NODES == len({c["node"] for c in fault_cells})
    # 43-dominant, scattered topology (no whole-node failures, <=4 of 8).
    by_node: dict[str, int] = {}
    for c in fault_cells:
        by_node[c["node"]] = by_node.get(c["node"], 0) + 1
    assert max(by_node.values()) <= 4
    xid43 = sum(c["xid"] == 43 for c in fault_cells)
    assert xid43 / len(fault_cells) >= 0.85


@pytest.mark.parametrize("base", COPIES, ids=lambda p: p.parent.name)
def test_incidents_are_aug17_correlated(base):
    incidents = _load(base, "incidents.json")
    assert len(incidents) == 7
    assert sum(i["hero"] for i in incidents) == 1
    for inc in incidents:
        assert inc["ts"] == HERO_TS
        assert inc["correlatedCount"] == 115
        assert "2023-08-29" not in inc["ts"]


@pytest.mark.parametrize("base", COPIES, ids=lambda p: p.parent.name)
def test_every_incident_has_matching_telemetry(base):
    incidents = _load(base, "incidents.json")
    for inc in incidents:
        tw = _load(base, f"telemetry/{inc['id']}.json")
        assert tw["gpu"] == f"{inc['gpu']['node']}-{inc['gpu']['idx']}"
        assert tw["centerTs"] == HERO_TS
        assert tw["series"]["temp"][0][0].startswith("2023-08-17")


def test_both_copies_byte_identical():
    names = ["meta.json", "fleet.json", "incidents.json"]
    names += [f"telemetry/{p.name}" for p in sorted((PUB / "telemetry").glob("INC-*.json"))]
    for name in names:
        assert (PUB / name).read_bytes() == (APP / name).read_bytes(), name


@pytest.mark.parametrize("base", COPIES, ids=lambda p: p.parent.name)
def test_no_debunked_aug29_artifact_remains(base):
    # Regression guard: the Aug-29 13:57 / 882 / 881 framing must not reappear
    # in any fixture file under either copy.
    for path in base.rglob("*.json"):
        text = path.read_text()
        assert "2023-08-29" not in text, path
        assert "882" not in text, path
        assert "881" not in text, path
