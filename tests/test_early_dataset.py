"""Tests for the labeled early-detection dataset builder (bead lys/r7j).

Covers the genuinely-new layer this bead adds on top of the existing cache-safe
resolver (scripts.lfs_helper) and streaming reader (gpusitter.telemetry):
Xid-onset labeling, pre-reference windowed features (no zero-fill, no future
leakage), prediction-point sampling with a negative-leakage guard, and the
compact output schema. The cache-safe seam is exercised end-to-end against a
real fake LFS repo so the "works when the working-tree CSV is deleted" criterion
is evidenced for the builder, not just the helper.
"""

import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from gpusitter.detection.early_dataset import (
    XID_METRIC,
    build_dataset,
    window_features,
    write_dataset,
    xid_onsets,
)

TZ = timezone(timedelta(hours=8))  # Kalos fixed +08:00


def _t(i, *, base=datetime(2023, 8, 15, 0, 0, 0, tzinfo=TZ), step=15):
    return base + timedelta(seconds=i * step)


def _iso(dt):
    # Match the real kalos textual scheme: space separator, +08:00 offset.
    return dt.isoformat(sep=" ")


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# --- Xid onset detection -----------------------------------------------------


def _xid_csv(tmp_path):
    # cols: n1-0 (A), n1-1 (B), n2-0 (C), n2-1 (D); empty cell = idle (skipped).
    rows = [
        "Time,n1-0,n1-1,n2-0,n2-1",
        f"{_iso(_t(0))},0,43,,0",  # A=0  B=43(first->censored) C=idle  D=0(first)
        f"{_iso(_t(1))},0,43,0,43",  # A=0  B=43                 C=0(first) D=43 -> onset
        f"{_iso(_t(2))},43,0,0,0",  # A=43 -> onset             B=0        D=0
        f"{_iso(_t(3))},43,,94,43",  # C=94 -> onset             D=43 -> onset
    ]
    return _write(tmp_path, "XID_ERRORS.csv", "\n".join(rows) + "\n")


def test_xid_onsets_detects_idle_to_fault(tmp_path):
    # Navigator repro (bug 1): a GPU idle (empty) in one row then faulting in the
    # next is a real onset — it must NOT be dropped as left-censored. The
    # empty-skipping long-record reader misses this; the empty-aware detector
    # catches it.
    csv = (
        "\n".join(
            [
                "Time,n1-0",
                f"{_iso(_t(0))},",  # idle (empty cell)
                f"{_iso(_t(1))},43",  # idle -> fault  => one onset
            ]
        )
        + "\n"
    )
    onsets = xid_onsets(_write(tmp_path, "XID_ERRORS.csv", csv))
    assert [(o.gpu.canonical, o.t) for o in onsets] == [("n1#0", _t(1))]


def test_xid_onsets_detects_transitions_and_excludes_left_censored(tmp_path):
    onsets = xid_onsets(_xid_csv(tmp_path))
    got = {(o.gpu.canonical, o.t) for o in onsets}
    assert got == {
        ("n1#0", _t(2)),  # A: 0->43
        ("n2#0", _t(3)),  # C: first obs 0, later ->94
        ("n2#1", _t(1)),  # D: 0->43 (re-arm)
        ("n2#1", _t(3)),  # D: 0->43 again after clearing
    }
    # B (n1#1) starts at 43 on its first observation -> left-censored, never an onset.
    assert all(o.gpu.canonical != "n1#1" for o in onsets)


# --- Windowed features (pre-reference, no zero-fill, no future leakage) -------


def _series(values, *, step=15):
    return [(_t(i, step=step), float(v)) for i, v in enumerate(values)]


def test_window_features_aggregates_only_lookback_and_past():
    # values at i=0..5; a spike of 999 sits at i=5 (AFTER t_ref) and must be excluded.
    series = _series([10, 20, 30, 40, 50, 999])
    f = window_features(series, t_ref=_t(4), lookback_s=45, sample_period_s=15)
    assert f["count"] == 4  # i = 1,2,3,4 (i=0 before lookback, i=5 future)
    assert f["present"] == 1
    assert f["min"] == 20.0 and f["max"] == 50.0  # 999 future spike excluded
    assert f["mean"] == 35.0
    assert f["last"] == 50.0  # chronological last in window
    assert f["delta"] == 30.0  # 50 - 20
    assert math.isclose(f["slope"], 2.0 / 3.0, rel_tol=1e-9)  # +10 per 15s
    assert math.isclose(f["coverage"], 1.0)  # 4 / (45/15 + 1)


def test_window_features_empty_is_nan_never_zero_filled():
    series = _series([10, 20, 30])
    f = window_features(series, t_ref=_t(-10), lookback_s=45, sample_period_s=15)
    assert f["count"] == 0 and f["present"] == 0 and f["coverage"] == 0.0
    for stat in ("mean", "std", "min", "max", "last", "delta", "slope"):
        assert math.isnan(f[stat]), f"{stat} should be NaN, not zero-filled"


# --- End-to-end build: schema, labels, leakage guard, cache-safe read --------


def _telemetry_csvs(tmp_path):
    # A=n1-0 faults at i=6; B=n1-1 never faults (control). Rising temp on A,
    # with a spike at i=6 that must NOT leak into a t_ref<=i=4 window.
    xid = (
        "\n".join(
            [
                "Time,n1-0,n1-1",
                f"{_iso(_t(0))},0,0",
                f"{_iso(_t(1))},0,0",
                f"{_iso(_t(2))},0,0",
                f"{_iso(_t(3))},0,0",
                f"{_iso(_t(4))},0,0",
                f"{_iso(_t(5))},0,0",
                f"{_iso(_t(6))},43,0",  # A onset at i=6 (t=90s)
                f"{_iso(_t(7))},43,0",
            ]
        )
        + "\n"
    )
    temp = (
        "\n".join(
            [
                "Time,n1-0,n1-1",
                f"{_iso(_t(0))},40,40",
                f"{_iso(_t(1))},42,40",
                f"{_iso(_t(2))},44,40",
                f"{_iso(_t(3))},46,40",
                f"{_iso(_t(4))},48,40",
                f"{_iso(_t(5))},80,40",
                f"{_iso(_t(6))},95,40",
                f"{_iso(_t(7))},95,40",
            ]
        )
        + "\n"
    )
    return {
        XID_METRIC: _write(tmp_path, "XID_ERRORS.csv", xid),
        "GPU_TEMP": _write(tmp_path, "GPU_TEMP.csv", temp),
    }


def test_build_dataset_schema_labels_and_no_future_leakage(tmp_path):
    sources = _telemetry_csvs(tmp_path)
    rows = build_dataset(
        sources,
        horizons_s=[30],  # onset at i=6 (90s) -> positive t_ref at i=4 (60s)
        lookback_s=45,
        neg_offset_s=60,  # same-gpu negative at i=2 (30s)
        control_gpus=["n1#1"],
        sample_period_s=15,
    )
    df = pd.DataFrame(rows)

    required = {
        "gpu",
        "node",
        "gpu_idx",
        "t_ref",
        "event_source",
        "horizon_s",
        "lookback_s",
        "label",
        "GPU_TEMP_mean",
        "GPU_TEMP_coverage",
        "GPU_TEMP_present",
    }
    assert required <= set(df.columns)
    assert (df["horizon_s"] == 30).all()
    assert (df["lookback_s"] == 45).all()
    assert (df["event_source"] == XID_METRIC).all()

    # Both classes present.
    assert set(df["label"]) == {0, 1}

    # The positive row for A: window i=1..4 (temp 42,44,46,48); the i=5 (80) and
    # i=6 (95) values are at/after the onset and must not leak into features.
    pos = df[(df["gpu"] == "n1#0") & (df["label"] == 1)].iloc[0]
    assert pos["GPU_TEMP_max"] == 48.0
    assert pos["GPU_TEMP_last"] == 48.0

    # Leakage-guard invariant: every label is consistent with the onset truth.
    onset_t = _t(6)
    for _, r in df.iterrows():
        t_ref = datetime.fromisoformat(r["t_ref"])
        within = (r["gpu"] == "n1#0") and (t_ref < onset_t <= t_ref + timedelta(seconds=30))
        assert int(r["label"]) == int(within)


def test_build_dataset_reads_through_lfs_cache_when_worktree_deleted(tmp_path):
    """The builder must work when raw CSVs exist only as LFS cache objects."""
    repo = tmp_path / "acme-util"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@e.com"], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "t"], check=True, capture_output=True
    )
    rel_dir = repo / "data" / "utilization" / "kalos"
    rel_dir.mkdir(parents=True)

    def _commit_lfs(name, content):
        # Stage a pointer in the tree + place the object in the LFS cache, then
        # delete the working file so only the cache object remains.
        import hashlib

        oid = hashlib.sha256(content.encode()).hexdigest()
        pointer = (
            f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {len(content)}\n"
        )
        ptr_path = rel_dir / name
        ptr_path.write_text(pointer)
        cache = repo / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(content)
        subprocess.run(
            ["git", "-C", str(repo), "add", f"data/utilization/kalos/{name}"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", name], check=True, capture_output=True
        )
        ptr_path.unlink()  # worktree file gone; only the LFS cache object survives

    xid = (
        "\n".join(
            [
                "Time,n1-0,n1-1",
                f"{_iso(_t(0))},0,0",
                f"{_iso(_t(1))},0,0",
                f"{_iso(_t(2))},0,0",
                f"{_iso(_t(3))},0,0",
                f"{_iso(_t(4))},0,0",
                f"{_iso(_t(5))},0,0",
                f"{_iso(_t(6))},43,0",
                f"{_iso(_t(7))},43,0",
            ]
        )
        + "\n"
    )
    temp = (
        "\n".join(
            [
                "Time,n1-0,n1-1",
                f"{_iso(_t(0))},40,40",
                f"{_iso(_t(1))},42,40",
                f"{_iso(_t(2))},44,40",
                f"{_iso(_t(3))},46,40",
                f"{_iso(_t(4))},48,40",
                f"{_iso(_t(5))},80,40",
                f"{_iso(_t(6))},95,40",
                f"{_iso(_t(7))},95,40",
            ]
        )
        + "\n"
    )
    _commit_lfs("XID_ERRORS.csv", xid)
    _commit_lfs("GPU_TEMP.csv", temp)

    # Drive the CLI end-to-end: it must resolve the deleted paths via the cache.
    out = tmp_path / "early.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_early_dataset.py",
            "--repo-dir",
            str(repo),
            "--metrics",
            "GPU_TEMP",
            "--horizons",
            "30",
            "--lookback",
            "45",
            "--neg-offset",
            "60",
            "--sample-period",
            "15",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    df = pd.read_csv(out)
    assert set(df["label"]) == {0, 1}
    assert (df["event_source"] == XID_METRIC).all()


def test_build_dataset_drops_same_gpu_negative_inside_horizon(tmp_path):
    # Navigator repro (bug 2): with neg_offset_s < horizon_s the same-GPU
    # pre-event negative falls INSIDE the positive horizon. It must be dropped
    # (leakage guard), not emitted at all. Onset at i=6 (90s); horizon=60 ->
    # positive t_ref at i=2 (30s); neg_offset=30 -> same-GPU negative at i=4
    # (60s), which lies in the positive horizon (30s, 90s].
    sources = _telemetry_csvs(tmp_path)  # A=n1-0 faults at i=6; n1-1 never faults
    rows = build_dataset(
        sources,
        horizons_s=[60],
        lookback_s=30,
        neg_offset_s=30,
        control_gpus=["n1#1"],  # a real, surviving negative for class balance
        sample_period_s=15,
    )
    refs = {(r["gpu"], datetime.fromisoformat(r["t_ref"])) for r in rows}

    # The leaked same-GPU negative at i=4 must be absent entirely.
    assert ("n1#0", _t(4)) not in refs
    # The positive at i=2 survives with label 1.
    pos = [r for r in rows if r["gpu"] == "n1#0" and r["label"] == 1]
    assert pos and datetime.fromisoformat(pos[0]["t_ref"]) == _t(2)
    # The control GPU yields a true negative at the positive ref (both classes).
    assert any(r["gpu"] == "n1#1" and r["label"] == 0 for r in rows)
    # Invariant: every emitted label matches the onset truth.
    onset_t = _t(6)
    for r in rows:
        t_ref = datetime.fromisoformat(r["t_ref"])
        within = (r["gpu"] == "n1#0") and (t_ref < onset_t <= t_ref + timedelta(seconds=60))
        assert int(r["label"]) == int(within)


def test_write_dataset_csv_roundtrip(tmp_path):
    rows = [{"gpu": "n1#0", "label": 1, "GPU_TEMP_mean": 48.0}]
    out = tmp_path / "d.csv"
    written = write_dataset(rows, str(out))
    assert written.endswith(".csv")
    df = pd.read_csv(written)
    assert df.iloc[0]["label"] == 1
