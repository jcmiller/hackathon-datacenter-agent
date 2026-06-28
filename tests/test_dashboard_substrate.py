"""Tests for the reproducible REAL dashboard substrate builder (bead t7p).

Load-bearing properties, all proven off-droplet against tiny synthetic wide-CSV
frames (no LFS / 80 GB trace):

1.  Incidents are edge-detected Xid ONSETS, not latched gauge snapshots.
2.  THE latched-state regression: a late timestamp where N GPUs read nonzero
    (cumulative latched faults) must NOT produce an N-GPU simultaneous cascade —
    the exact Aug-29/882 artifact the substrate exists to prevent.
3.  Deterministic selection + byte-identical re-serialization.
4.  Fleet fault cells are hero-burst onset members only; other GPUs' status comes
    from utilization, never their latched Xid value.
5.  Telemetry windows carry the real per-incident series; the manifest carries
    provenance and is flagged real.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta

import pytest

from gpusitter.app import dashboard_substrate as ds

TZ = "+08:00"
T0 = datetime.fromisoformat(f"2023-08-17 05:00:00{TZ}")


def _ts(i: int) -> datetime:
    """The i-th 15s sample timestamp."""
    return T0 + timedelta(seconds=15 * i)


def _write_csv(path, gpus, rows):
    """rows: list[(sample_index, {gpu_dash_id: cell_str})]. Missing -> '' (idle)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Time", *gpus])
        for idx, vals in rows:
            w.writerow([_ts(idx).isoformat(), *[vals.get(g, "") for g in gpus]])


# --- id helpers -----------------------------------------------------------------


def test_id_helpers_bridge_canonical_and_dashboard_forms():
    assert ds._node_of("172.31.15.220#4") == "172.31.15.220"
    assert ds._idx_of("172.31.15.220#4") == 4
    assert ds._dash_id("172.31.15.220#4") == "172.31.15.220-4"
    assert ds._canon_header("172.31.15.220-4") == "172.31.15.220#4"


def test_xid_label_and_severity():
    assert ds.xid_label(43) == "GPU stopped processing"
    assert ds.xid_severity(43) == "warn"  # hang -> warn badge
    assert ds.xid_severity(31) == "crit"  # memory fault -> crit badge
    assert ds.xid_label(9999).startswith("Xid 9999")


# --- onset edge detection -------------------------------------------------------


def test_load_onsets_is_edge_detection_not_latched_samples(tmp_path):
    gpus = ["10.0.0.1-0", "10.0.0.1-1"]
    xid = tmp_path / "XID_ERRORS.csv"
    # gpu0: healthy then onset->latched for 3 samples (ONE onset, not three).
    # gpu1: stays healthy.
    _write_csv(
        xid,
        gpus,
        [
            (0, {"10.0.0.1-0": "0", "10.0.0.1-1": "0"}),
            (1, {"10.0.0.1-0": "43", "10.0.0.1-1": "0"}),
            (2, {"10.0.0.1-0": "43", "10.0.0.1-1": "0"}),
            (3, {"10.0.0.1-0": "43", "10.0.0.1-1": "0"}),
        ],
    )
    onsets = ds.load_onsets(str(xid))
    assert len(onsets) == 1
    assert onsets[0].gpu == "10.0.0.1#0"
    assert onsets[0].code == 43
    assert onsets[0].t == _ts(1)


# --- THE latched-state regression (AC6) -----------------------------------------


def _latched_cascade_csv(tmp_path):
    """N GPUs already faulted at the first observation (left-censored latched) +
    2 real onsets at a late sample. The raw XID snapshot at the late sample reads
    N+2 nonzero, but only 2 are true onsets."""
    latched = [f"10.0.{n}.1-0" for n in range(12)]
    real = ["10.9.9.1-0", "10.9.9.2-0"]  # real1 onsets @ s10 (43), real2 @ s14 (31)
    gpus = latched + real
    rows = []
    # row 0: latched GPUs ALREADY nonzero (first observation == fault -> excluded);
    # real GPUs healthy.
    rows.append((0, {**dict.fromkeys(latched, "43"), "10.9.9.1-0": "0", "10.9.9.2-0": "0"}))
    for i in range(1, 10):
        rows.append((i, {**dict.fromkeys(latched, "43"), "10.9.9.1-0": "0", "10.9.9.2-0": "0"}))
    # s10: real1 onsets (43); s14: real2 onsets (31). Distinct time + code so they
    # are two separate isolated incidents, not one coincident cohort.
    for i in range(10, 14):
        rows.append((i, {**dict.fromkeys(latched, "43"), "10.9.9.1-0": "43", "10.9.9.2-0": "0"}))
    for i in range(14, 16):
        rows.append((i, {**dict.fromkeys(latched, "43"), "10.9.9.1-0": "43", "10.9.9.2-0": "31"}))
    xid = tmp_path / "XID_ERRORS.csv"
    _write_csv(xid, gpus, rows)
    return xid, latched, real


def test_latched_cumulative_snapshot_is_not_a_simultaneous_cascade(tmp_path):
    xid, latched, real = _latched_cascade_csv(tmp_path)

    onsets = ds.load_onsets(str(xid))
    # Only the 2 real non-fault->fault transitions are onsets; the 12 latched GPUs
    # (first observation already nonzero) are excluded.
    assert sorted(o.gpu for o in onsets) == ["10.9.9.1#0", "10.9.9.2#0"]

    # No burst: 2 onsets is far below a correlated-cascade threshold. Critically,
    # the 14-nonzero late snapshot does NOT become a 14-GPU burst.
    bursts = ds.detect_bursts(onsets, min_onsets=10)
    assert bursts == []

    # A naive latched-snapshot reader WOULD see 14 nonzero at the late sample —
    # prove the trap exists, then prove the substrate avoids it.
    raw_nonzero = ds.snapshot_row(str(xid), _ts(14))
    assert len(raw_nonzero) == len(latched) + len(real)  # 14, the tempting count

    incidents = ds.select_incidents(onsets, bursts)
    assert len(incidents) == 2
    assert all(inc["correlatedCount"] <= 2 for inc in incidents)
    assert max(inc["correlatedCount"] for inc in incidents) != len(raw_nonzero)


def test_fleet_snapshot_does_not_mark_latched_gpus_as_fault(tmp_path):
    xid, latched, real = _latched_cascade_csv(tmp_path)
    # latched GPUs are busy (util>0) at the snapshot; real onsets too.
    util_at = {ds._canon_header(g): 99.0 for g in latched + real}
    temp_at = {ds._canon_header(g): 55.0 for g in latched + real}
    fleet = ds.build_fleet_snapshot(temp_at=temp_at, util_at=util_at, hero=None, center=_ts(10))
    statuses = {(c["node"], c["idx"]): c["status"] for c in fleet["cells"]}
    # No hero burst -> ZERO fault cells even though 14 GPUs hold a latched Xid.
    assert fleet["faulted"] == 0
    assert all(s != "fault" for s in statuses.values())
    # latched GPUs are rendered by utilization (active), never by their Xid value.
    assert statuses[("10.0.0.1", 0)] == "active"


# --- burst detection / hero selection -------------------------------------------


def _burst_csv(tmp_path, *, small=20, big=45):
    """Two correlated bursts at different samples; the bigger is the hero."""
    big_gpus = [f"10.1.{n // 8}.{n % 8 + 10}-{n % 8}" for n in range(big)]
    small_gpus = [f"10.2.{n // 8}.{n % 8 + 10}-{n % 8}" for n in range(small)]
    gpus = big_gpus + small_gpus
    rows = [(0, dict.fromkeys(gpus, "0"))]
    # small burst at sample 2; big burst at sample 10 (>45s apart so the two
    # cohorts never overlap and both are detected independently).
    rows.append((1, dict.fromkeys(gpus, "0")))
    rows.append((2, {**dict.fromkeys(gpus, "0"), **dict.fromkeys(small_gpus, "31")}))
    for i in range(3, 10):
        rows.append((i, {**dict.fromkeys(big_gpus, "0"), **dict.fromkeys(small_gpus, "31")}))
    rows.append((10, {**dict.fromkeys(small_gpus, "31"), **dict.fromkeys(big_gpus, "43")}))
    xid = tmp_path / "XID_ERRORS.csv"
    _write_csv(xid, gpus, rows)
    return xid, big_gpus, small_gpus


def test_detect_bursts_picks_largest_as_hero(tmp_path):
    xid, big_gpus, small_gpus = _burst_csv(tmp_path)
    onsets = ds.load_onsets(str(xid))
    bursts = ds.detect_bursts(onsets, min_onsets=10)
    assert len(bursts) == 2
    hero = bursts[0]
    assert hero.n_onsets == len(big_gpus)
    assert hero.center == _ts(10)
    assert hero.codes == {43: len(big_gpus)}
    assert len(hero.nodes) == len({ds._node_of(ds._canon_header(g)) for g in big_gpus})


def test_cohort_counts_coincident_onsets(tmp_path):
    xid, big_gpus, _ = _burst_csv(tmp_path)
    onsets = ds.load_onsets(str(xid))
    center = _ts(10)
    coh = ds.cohort(onsets, center, ds.COHORT_WINDOW_S)
    assert len(coh) == len(big_gpus)


# --- telemetry windows ----------------------------------------------------------


def test_read_incident_windows_slices_per_gpu(tmp_path):
    gpus = ["10.0.0.1-0", "10.0.0.2-0"]
    temp = tmp_path / "GPU_TEMP.csv"
    _write_csv(
        temp,
        gpus,
        [(i, {"10.0.0.1-0": str(50 + i), "10.0.0.2-0": str(60 + i)}) for i in range(20)],
    )
    targets = {
        "10.0.0.1#0": (_ts(5), _ts(8)),
        "10.0.0.2#0": (_ts(10), _ts(12)),
    }
    series = ds.read_incident_windows(str(temp), targets)
    assert [t for t, _ in series["10.0.0.1#0"]] == [_ts(5), _ts(6), _ts(7), _ts(8)]
    assert [v for _, v in series["10.0.0.1#0"]] == [55.0, 56.0, 57.0, 58.0]
    assert [t for t, _ in series["10.0.0.2#0"]] == [_ts(10), _ts(11), _ts(12)]


# --- end-to-end assembly + provenance + determinism -----------------------------


def _full_csvs(tmp_path):
    xid, big_gpus, small_gpus = _burst_csv(tmp_path)
    gpus = big_gpus + small_gpus
    # metric frames covering the same samples for telemetry + snapshot.
    temp = tmp_path / "GPU_TEMP.csv"
    power = tmp_path / "POWER_USAGE.csv"
    util = tmp_path / "GPU_UTIL.csv"
    rows_t = [(i, {g: str(50 + (i % 5)) for g in gpus}) for i in range(25)]
    rows_p = [(i, {g: str(300 + (i % 5) * 10) for g in gpus}) for i in range(25)]
    rows_u = [(i, dict.fromkeys(gpus, "99")) for i in range(25)]
    _write_csv(temp, gpus, rows_t)
    _write_csv(power, gpus, rows_p)
    _write_csv(util, gpus, rows_u)
    return xid, temp, power, util, big_gpus, small_gpus


def _assemble(tmp_path, monkeypatch):
    # shrink the burst threshold so the synthetic frames exercise the real path.
    monkeypatch.setattr(ds, "BURST_MIN_ONSETS", 10)
    xid, temp, power, util, big_gpus, small_gpus = _full_csvs(tmp_path)
    sub = ds.assemble_substrate(
        xid_csv=str(xid),
        temp_csv=str(temp),
        power_csv=str(power),
        util_csv=str(util),
        total_gpus=len(big_gpus) + len(small_gpus),
        source="synthetic test frame",
        generation_command="pytest",
        input_meta=[{"metric": "XID_ERRORS", "path": "data/.../XID_ERRORS.csv"}],
        git_rev="deadbeef",
    )
    return sub, big_gpus, small_gpus


def test_assemble_substrate_shapes_and_provenance(tmp_path, monkeypatch):
    sub, big_gpus, small_gpus = _assemble(tmp_path, monkeypatch)

    # meta: cascade count is the edge-detected hero ONSET count, not cumulative.
    assert sub.meta["faulted"] == len(big_gpus)
    assert sub.meta["totalGpus"] == len(big_gpus) + len(small_gpus)
    assert sub.meta["cascadeTs"] == _ts(10).isoformat()

    # incidents: a hero + cascade members; exactly one hero card.
    assert sum(1 for i in sub.incidents if i["hero"]) == 1
    hero = next(i for i in sub.incidents if i["hero"])
    assert hero["correlatedCount"] == len(big_gpus)
    assert hero["xid"] == 43

    # fleet: faulted == hero onsets; all fault cells are hero members.
    assert sub.fleet["faulted"] == len(big_gpus)
    fault_nodes = {(c["node"], c["idx"]) for c in sub.fleet["cells"] if c["status"] == "fault"}
    assert len(fault_nodes) == len(big_gpus)

    # telemetry: real series per incident, centered on the onset.
    for inc in sub.incidents:
        rec = sub.telemetry[inc["id"]]
        assert rec["centerTs"] == inc["ts"]
        assert len(rec["series"]["temp"]) > 0
        assert len(rec["series"]["power"]) > 0

    # manifest provenance (AC4).
    m = sub.manifest
    assert m["kind"] == "real" and m["telemetryKind"] == "real"
    assert m["generatedFromGitRev"] == "deadbeef"
    assert m["generationCommand"] == "pytest"
    assert m["onsetTotal"] == len(big_gpus) + len(small_gpus)
    assert m["hero"]["onsets"] == len(big_gpus)
    assert m["inputs"] and m["inputs"][0]["metric"] == "XID_ERRORS"


def test_write_substrate_is_deterministic_and_prunes(tmp_path, monkeypatch):
    sub, _, _ = _assemble(tmp_path, monkeypatch)
    out = tmp_path / "out"

    # seed a stale telemetry file that must be pruned.
    (out / "telemetry").mkdir(parents=True)
    (out / "telemetry" / "INC-099.json").write_text("{}")

    ds.write_substrate(sub, out)
    assert not (out / "telemetry" / "INC-099.json").exists()
    first = {p.name: p.read_bytes() for p in sorted(out.rglob("*.json"))}

    # re-run on identical inputs -> byte-identical tree.
    ds.write_substrate(sub, out)
    second = {p.name: p.read_bytes() for p in sorted(out.rglob("*.json"))}
    assert first == second

    # manifest + the four top-level files exist.
    for name in ("meta.json", "fleet.json", "incidents.json", "manifest.json"):
        assert (out / name).exists()


def test_load_substrate_round_trips_written_artifact(tmp_path, monkeypatch):
    # write -> load must reconstruct meta/fleet/incidents/manifest + every
    # telemetry window, so the backend serves the real artifact off-droplet.
    sub, _, _ = _assemble(tmp_path, monkeypatch)
    out = tmp_path / "out"
    ds.write_substrate(sub, out)

    assert ds.substrate_available(out) is True
    loaded = ds.load_substrate(out)
    assert loaded.meta == sub.meta
    assert loaded.fleet == sub.fleet
    assert loaded.incidents == sub.incidents
    # The loaded artifact must equal the on-disk JSON: re-encode the in-memory
    # manifest the same way (JSON coerces the int Xid-code keys to strings).
    assert loaded.manifest == json.loads(json.dumps(sub.manifest))
    assert set(loaded.telemetry) == set(sub.telemetry)
    for inc_id, rec in sub.telemetry.items():
        assert loaded.telemetry[inc_id] == json.loads(json.dumps(rec))


def test_substrate_available_false_when_incomplete(tmp_path):
    # A directory missing any of the four top-level documents is NOT available
    # (so the backend resolver degrades to the fixture/unavailable branch).
    assert ds.substrate_available(tmp_path) is False
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "meta.json").write_text("{}")
    (tmp_path / "fleet.json").write_text("{}")
    assert ds.substrate_available(tmp_path) is False  # incidents.json still absent
    (tmp_path / "incidents.json").write_text("[]")
    assert ds.substrate_available(tmp_path) is True


def test_build_substrate_off_droplet_fails_loud(tmp_path):
    # No data/ tree under repo_dir -> the canonical resolver raises (honest
    # "raw data not materialized" state), never a silent empty substrate.
    with pytest.raises((FileNotFoundError, ValueError)):
        ds.build_substrate(repo_dir=str(tmp_path))


# --- the COMMITTED real artifact (generated on the droplet) ---------------------


def _committed():
    d = ds.SUBSTRATE_DIR

    def load(name):
        return json.loads((d / name).read_text())

    return (
        d,
        load("meta.json"),
        load("fleet.json"),
        load("incidents.json"),
        load("manifest.json"),
    )


def test_committed_substrate_ships_and_is_real():
    d, meta, fleet, incidents, manifest = _committed()
    assert (d / "README.md").exists()
    assert manifest["kind"] == "real" and manifest["telemetryKind"] == "real"
    # provenance is populated from the real droplet run.
    assert manifest["onsetTotal"] > 0
    assert manifest["generatedFromGitRev"]
    assert all(i.get("path") for i in manifest["inputs"])


def test_committed_substrate_is_edge_detected_not_aug29_cascade():
    _, meta, fleet, incidents, manifest = _committed()
    # the debunked latched artifact must appear NOWHERE.
    assert "2023-08-29" not in meta["cascadeTs"]
    assert all("2023-08-29" not in i["ts"] for i in incidents)
    # cascade size is the edge-detected onset count, never the cumulative 882.
    assert meta["faulted"] == manifest["hero"]["onsets"]
    assert meta["faulted"] != 882


def test_committed_fleet_fault_cells_are_hero_members_only():
    _, meta, fleet, incidents, manifest = _committed()
    fault = [c for c in fleet["cells"] if c["status"] == "fault"]
    # exactly the hero onset members are faults; nobody else is latched-into-fault.
    assert len(fault) == manifest["hero"]["onsets"]
    assert fleet["faulted"] == manifest["hero"]["onsets"]
    assert all(c["xid"] == 0 for c in fleet["cells"] if c["status"] != "fault")


def test_committed_substrate_internally_consistent():
    d, meta, fleet, incidents, manifest = _committed()
    assert manifest["incidentCount"] == len(incidents)
    assert sum(1 for i in incidents if i["hero"]) == 1
    # every incident has a telemetry file with the matching center timestamp.
    for inc in incidents:
        rec = json.loads((d / "telemetry" / f"{inc['id']}.json").read_text())
        assert rec["centerTs"] == inc["ts"]
        assert rec["gpu"] == f"{inc['gpu']['node']}-{inc['gpu']['idx']}"
