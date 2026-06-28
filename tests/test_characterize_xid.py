"""Deterministic unit tests for scripts/characterize_xid.py.

No droplet / no real data: tiny synthetic wide CSVs with planted signals exercise
the rising-edge event extractor and the precursor windowing. Each test sets up
the *wrong*/raw state and asserts the corrected characterization, so deleting the
logic under test makes the test fail (non-vacuous).
"""

import importlib.util
import os
import sys
from datetime import datetime, timedelta

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# scripts/characterize_xid imports the consolidated telemetry surface. The
# package resolves via the editable install; importorskip guards a partial env.
pytest.importorskip("gpusitter.telemetry.ingest")

_spec = importlib.util.spec_from_file_location(
    "characterize_xid", os.path.join(REPO_ROOT, "scripts", "characterize_xid.py")
)
cx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cx)

FMT = "%Y-%m-%d %H:%M:%S%z"
T0 = datetime.strptime("2023-08-15 15:30:15+08:00", FMT)


def _ts(i):
    # isoformat(sep=" ") emits the colon offset ("+08:00") — the real kalos cell
    # format, and what extract_events emits (it parses via fromisoformat). Using
    # strftime("%z") here would give "+0800" and never match event timestamps.
    return (T0 + timedelta(seconds=15 * i)).isoformat(sep=" ")


def _write_wide(path, gpu_cols, rows):
    """rows: list of (i, {gpu: cell_str}); missing gpu -> empty cell."""
    with open(path, "w", newline="") as fh:
        fh.write("Time," + ",".join(gpu_cols) + "\n")
        for i, cells in rows:
            line = [_ts(i)] + [cells.get(g, "") for g in gpu_cols]
            fh.write(",".join(line) + "\n")


# --------------------------------------------------------------------------- #
# Event extraction (rising edge over the gauge)
# --------------------------------------------------------------------------- #
def test_held_code_collapses_to_one_event(tmp_path):
    """A faulted GPU repeats its code every sample (gauge). That is ONE event,
    not one-per-sample. Deleting the rising-edge guard would over-count."""
    gpus = ["10.0.0.1-0"]
    rows = [(i, {"10.0.0.1-0": "0"}) for i in range(3)]
    rows += [(i, {"10.0.0.1-0": "43"}) for i in range(3, 50)]  # 47 repeats
    csv = tmp_path / "XID_ERRORS.csv"
    _write_wide(csv, gpus, rows)

    events = cx.extract_events(str(csv))
    assert len(events) == 1
    assert events[0].code == 43
    assert events[0].gpu == "10.0.0.1#0"
    assert events[0].t == _ts(3)  # the rising edge, not a later repeat


def test_clear_then_refault_is_two_events(tmp_path):
    """0->43->0->43 is two distinct faults (transformation: before has 4 nonzero
    cells in one stream, after has exactly 2 events)."""
    gpus = ["10.0.0.1-0"]
    rows = [
        (0, {"10.0.0.1-0": "0"}),
        (1, {"10.0.0.1-0": "43"}),
        (2, {"10.0.0.1-0": "0"}),
        (3, {"10.0.0.1-0": "43"}),
    ]
    csv = tmp_path / "XID_ERRORS.csv"
    _write_wide(csv, gpus, rows)
    events = cx.extract_events(str(csv))
    assert [e.t for e in events] == [_ts(1), _ts(3)]
    assert all(e.code == 43 for e in events)


def test_empty_clear_same_code_refault_is_two_onsets(tmp_path):
    """THE empty-clear regression. A fault that clears to an EMPTY cell and then
    re-raises the SAME code is two onsets. The empty-skipping long-record path
    never sees the clear (last stays 43) and collapses this to one — the bug nav
    rejected. The canonical empty-aware detector treats empty as idle/clear, so
    empty->43, (empty), empty->43 yields exactly TWO onsets."""
    gpus = ["10.0.0.1-0"]
    rows = [
        (0, {}),  # idle (empty) — establishes non-fault state
        (1, {"10.0.0.1-0": "43"}),  # idle -> 43  : onset 1
        (2, {}),  # cleared back to empty/idle
        (3, {"10.0.0.1-0": "43"}),  # idle -> 43  : onset 2 (same code)
    ]
    csv = tmp_path / "XID_ERRORS.csv"
    _write_wide(csv, gpus, rows)
    events = cx.extract_events(str(csv))
    assert [(e.code, e.t) for e in events] == [(43, _ts(1)), (43, _ts(3))]


def test_latched_code_change_is_not_a_new_onset(tmp_path):
    """Canonical onset semantics: an onset is a transition INTO a fault from a
    non-fault state. A code change while already faulted with NO intervening clear
    (31 -> 45) is a latched fault, NOT a new onset — only the first 31 counts."""
    gpus = ["10.0.0.2-3"]
    rows = [
        (0, {"10.0.0.2-3": "0"}),
        (1, {"10.0.0.2-3": "31"}),  # 0 -> 31 : onset
        (2, {"10.0.0.2-3": "31"}),  # latched
        (3, {"10.0.0.2-3": "45"}),  # 31 -> 45 latched (no clear) : NOT an onset
    ]
    csv = tmp_path / "XID_ERRORS.csv"
    _write_wide(csv, gpus, rows)
    events = cx.extract_events(str(csv))
    assert [(e.code, e.t) for e in events] == [(31, _ts(1))]


def test_cleared_then_different_code_is_new_onset(tmp_path):
    """With a clear between them, two different codes are two onsets: 31 clears to
    0, then 45 is a fresh transition into a fault."""
    gpus = ["10.0.0.2-3"]
    rows = [
        (0, {"10.0.0.2-3": "0"}),
        (1, {"10.0.0.2-3": "31"}),  # 0 -> 31 : onset
        (2, {"10.0.0.2-3": "0"}),  # cleared
        (3, {"10.0.0.2-3": "45"}),  # 0 -> 45 : onset
    ]
    csv = tmp_path / "XID_ERRORS.csv"
    _write_wide(csv, gpus, rows)
    events = cx.extract_events(str(csv))
    assert [(e.code, e.t) for e in events] == [(31, _ts(1)), (45, _ts(3))]


def test_no_events_when_all_clear(tmp_path):
    gpus = ["10.0.0.3-0"]
    rows = [(i, {"10.0.0.3-0": "0"}) for i in range(10)]
    csv = tmp_path / "XID_ERRORS.csv"
    _write_wide(csv, gpus, rows)
    assert cx.extract_events(str(csv)) == []


def test_left_censored_first_sample_excluded(tmp_path):
    """A GPU already nonzero at the trace's FIRST sample (t0) is left-censored:
    its onset predates the window, so it is NOT an event. Setup is the wrong/raw
    state (a held 43 from the very first row); the corrected result is zero
    events. Deleting the t0 guard would manufacture a boundary rising edge."""
    gpus = ["10.0.0.1-0"]
    rows = [(i, {"10.0.0.1-0": "43"}) for i in range(10)]  # nonzero from row 0 (t0)
    csv = tmp_path / "XID_ERRORS.csv"
    _write_wide(csv, gpus, rows)
    assert cx.extract_events(str(csv)) == []


def test_midtrace_onset_counted_but_t0_onset_suppressed(tmp_path):
    """The discriminating case for the gauge's empty==healthy semantics.

    Empty (healthy) cells are the cleared baseline, so a GPU's FIRST nonzero
    appearing *mid-trace* is a genuine observed onset and must be counted; only a
    code already present at t0 is left-censored. GPU B is nonzero at t0 (drop);
    GPU A is empty until it first faults at t3 (keep). Suppressing every GPU's
    first observation (the rejected over-fix) would wrongly drop A and yield no
    events here — this asserts exactly one event, A's onset."""
    gpus = ["10.0.0.1-0", "10.0.0.2-0"]
    rows = [
        (0, {"10.0.0.2-0": "43"}),  # B nonzero at t0 -> left-censored (A empty)
        (1, {"10.0.0.2-0": "43"}),
        (2, {"10.0.0.2-0": "43"}),
        (3, {"10.0.0.1-0": "31", "10.0.0.2-0": "43"}),  # A's first fault -> event
    ]
    csv = tmp_path / "XID_ERRORS.csv"
    _write_wide(csv, gpus, rows)
    events = cx.extract_events(str(csv))
    assert [(e.gpu, e.code, e.t) for e in events] == [("10.0.0.1#0", 31, _ts(3))]


def test_t0_left_censored_then_clear_then_refault_is_one_event(tmp_path):
    """A t0-left-censored GPU that later clears and re-faults yields exactly ONE
    event — the observed re-fault — not two. The t0 43 is suppressed; only the
    genuine in-window onset counts."""
    gpus = ["10.0.0.1-0"]
    rows = [
        (0, {"10.0.0.1-0": "43"}),  # left-censored at t0, suppressed
        (1, {"10.0.0.1-0": "43"}),
        (2, {"10.0.0.1-0": "0"}),  # cleared
        (3, {"10.0.0.1-0": "43"}),  # genuine observed re-fault -> 1 event
    ]
    csv = tmp_path / "XID_ERRORS.csv"
    _write_wide(csv, gpus, rows)
    events = cx.extract_events(str(csv))
    assert [(e.code, e.t) for e in events] == [(43, _ts(3))]


def test_first_row_all_empty_then_onset_counted(tmp_path):
    """t0 is read from the Time column, not the first non-empty record: if the
    first row is all-empty (no faults at t0), a GPU faulting on a later row is a
    genuine onset, not suppressed. Guards the _first_timestamp boundary."""
    gpus = ["10.0.0.1-0"]
    rows = [
        (0, {}),  # all-empty first row (t0); no faults
        (1, {"10.0.0.1-0": "43"}),  # first nonzero, after t0 -> event
    ]
    csv = tmp_path / "XID_ERRORS.csv"
    _write_wide(csv, gpus, rows)
    events = cx.extract_events(str(csv))
    assert [(e.code, e.t) for e in events] == [(43, _ts(1))]


def test_distribution_counts(tmp_path):
    gpus = ["10.0.0.1-0", "10.0.0.1-1", "10.0.0.2-0"]
    rows = [
        (0, {"10.0.0.1-0": "0", "10.0.0.1-1": "0", "10.0.0.2-0": "0"}),
        (1, {"10.0.0.1-0": "43", "10.0.0.1-1": "43", "10.0.0.2-0": "31"}),
        (2, {"10.0.0.1-0": "43", "10.0.0.1-1": "43", "10.0.0.2-0": "31"}),
    ]
    csv = tmp_path / "XID_ERRORS.csv"
    _write_wide(csv, gpus, rows)
    events = cx.extract_events(str(csv))
    freq = cx.frequency_stats(events, n_gpus_total=3, span_seconds=30)
    assert freq["total_events"] == 3
    assert freq["distinct_event_gpus"] == 3
    assert freq["by_code"][43]["events"] == 2
    assert freq["by_code"][31]["events"] == 1
    assert freq["by_code"][43]["severity"] == "fatal_hang"


# --------------------------------------------------------------------------- #
# Temporal distribution / correlated-burst detection
# --------------------------------------------------------------------------- #
def test_largest_burst_isolated_from_sporadic():
    """Many GPUs across nodes faulting in one bin is a correlated burst; a lone
    later fault is sporadic. The detector must separate them (else MTBF treats a
    cluster-wide event as independent per-GPU faults)."""
    burst = [
        cx.XidEvent("10.0.0.1#0", _ts(0), 43),
        cx.XidEvent("10.0.0.1#1", _ts(1), 43),
        cx.XidEvent("10.0.0.2#0", _ts(2), 43),
        cx.XidEvent("10.0.0.3#0", _ts(3), 43),
        cx.XidEvent("10.0.0.3#1", _ts(4), 31),
    ]
    sporadic = [cx.XidEvent("10.0.0.9#0", _ts(1000), 43)]
    stats = cx.temporal_stats(burst + sporadic, burst_bin_seconds=300)
    lb = stats["largest_burst"]
    assert lb["events"] == 5
    assert lb["distinct_gpus"] == 5
    assert lb["distinct_nodes"] == 3
    assert lb["by_code"] == {43: 4, 31: 1}
    assert lb["frac_of_all_events"] == round(5 / 6, 3)
    assert stats["sporadic_events"] == 1


def test_temporal_empty():
    assert cx.temporal_stats([])["largest_burst"] is None


# --------------------------------------------------------------------------- #
# Precursor windowing
# --------------------------------------------------------------------------- #
def test_prefault_window_bounds_and_ramp(tmp_path):
    """A planted pre-fault ramp must surface as a positive delta vs baseline and
    a detectable 2-sigma deviation; samples at/after the fault are excluded."""
    gpu_col = "10.0.0.1-0"
    gpu_canon = "10.0.0.1#0"
    # 60 samples (15 min): baseline ~50 with small +/-0.5 noise for the first 40
    # (so std>0, exercising the real 2-sigma path), then a ramp in the final 20.
    # Fault at sample 60.
    rows = []
    for i in range(60):
        if i < 40:
            val = 49.5 if i % 2 else 50.5
        else:
            val = 50.0 + (i - 39) * 1.5
        rows.append((i, {gpu_col: f"{val:.1f}"}))
    csv = tmp_path / "GPU_TEMP.csv"
    _write_wide(csv, [gpu_col], rows)

    key = f"{gpu_canon}@{_ts(60)}"
    event_times = {key: (gpu_canon, _ts(60))}
    windows = cx.collect_prefault_windows(str(csv), "GPU_TEMP", event_times, baseline_seconds=3600)
    samples = windows[key]
    # All retained samples are strictly before the fault timestamp.
    assert samples, "expected pre-fault samples"
    assert all(t < _ts(60) for t, _ in samples)
    assert len(samples) == 60

    stats = cx.precursor_stats(
        windows,
        event_times,
        horizons_seconds=[300],  # 5 min = last 20 samples (the ramp)
        lead_for_baseline=300,
    )
    h = stats["by_horizon"]["300"]
    assert h["mean_delta_vs_baseline"] > 5.0  # ramp well above baseline 50
    assert h["mean_slope_per_min"] > 0.0
    assert h["frac_detectable_2sigma"] == 1.0


def test_window_excludes_samples_before_baseline_and_after_fault(tmp_path):
    gpu_col = "10.0.0.1-0"
    gpu_canon = "10.0.0.1#0"
    rows = [(i, {gpu_col: "50.0"}) for i in range(20)]  # 0..19
    rows += [(i, {gpu_col: "99.0"}) for i in range(20, 25)]  # at/after fault
    csv = tmp_path / "GPU_TEMP.csv"
    _write_wide(csv, [gpu_col], rows)
    # Fault at sample 20; baseline window only 90s (6 samples) before it.
    key = f"{gpu_canon}@{_ts(20)}"
    event_times = {key: (gpu_canon, _ts(20))}
    windows = cx.collect_prefault_windows(str(csv), "GPU_TEMP", event_times, baseline_seconds=90)
    samples = windows[key]
    assert all(t < _ts(20) for t, _ in samples)  # nothing from >= fault
    assert all(v == 50.0 for _, v in samples)  # no 99.0 leak
    assert len(samples) == 6  # samples 14..19


def test_repeated_same_gpu_events_kept_distinct(tmp_path):
    """Two faults on the SAME GPU must yield two distinct precursor windows, not
    collapse to one. Keying event_times by GPU alone (the rejected behavior)
    would drop the earlier event; keying by (gpu, t_event) keeps both. A single
    streamed sample can also feed both windows when their baseline spans overlap.
    """
    gpu_col = "10.0.0.1-0"
    gpu_canon = "10.0.0.1#0"
    # Steady 50.0 telemetry across the whole span; two faults at samples 10, 30.
    rows = [(i, {gpu_col: "50.0"}) for i in range(30)]
    csv = tmp_path / "GPU_TEMP.csv"
    _write_wide(csv, [gpu_col], rows)

    e1, e2 = _ts(10), _ts(30)
    k1, k2 = f"{gpu_canon}@{e1}", f"{gpu_canon}@{e2}"
    event_times = {k1: (gpu_canon, e1), k2: (gpu_canon, e2)}
    windows = cx.collect_prefault_windows(
        str(csv), "GPU_TEMP", event_times, baseline_seconds=10 * 60
    )
    # Both events retained as separate keys (collapse-by-gpu would lose one).
    assert set(windows) == {k1, k2}
    assert all(t < e1 for t, _ in windows[k1])  # window 1 strictly before fault 1
    assert all(t < e2 for t, _ in windows[k2])
    assert windows[k1], "first event must keep its own pre-fault samples"
    # The shared 50.0 samples in [e1-10min, e1) feed BOTH windows (overlap).
    assert windows[k1][0] in windows[k2]

    stats = cx.precursor_stats(windows, event_times, horizons_seconds=[60], lead_for_baseline=60)
    # Two events contributed baseline data, not one collapsed event.
    assert stats["events_with_any_baseline_data"] == 2
