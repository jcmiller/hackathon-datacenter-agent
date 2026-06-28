#!/usr/bin/env python3
"""Reproducible, READ-ONLY characterization of the Aug-17 06:00 correlated Xid
burst on the Acme kalos cluster (bead aieng26hack-6xk).

Runs on the droplet against the raw kalos telemetry. It only ever READS the
files (open() for read); it never checks out, writes, or mutates the acme-util
tree. Stdlib only, streaming, memory-bounded — scales to the ~750 MB XID file.

Onset semantics replicate the nav-approved empty-aware edge detector
(src/gpusitter/rca/job_join.stream_xid_onsets): an ONSET is a transition INTO a
fault — the cell at t is nonzero and the GPU's prior OBSERVED state was non-fault
(healthy 0.0 or idle empty-cell). Excluded: latched faults (nonzero->nonzero,
the dominant kalos pattern) and a GPU whose first observation is already nonzero
(pre-window history unknown). This is the exact correction that debunked the
spurious "Aug-29 882-GPU" event.

Usage (on droplet):
    python3 /tmp/incident-aug17-0600-analysis.py /root/hackathon-datacenter-agent
"""

import csv
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta

KALOS = "data/acme-util/data/utilization/kalos"
TRACE = "data/acme-util/data/job_trace/trace_kalos.csv"

# Burst window on 2023-08-17 (local +08:00).
DAY = "2023-08-17"
TZ = "+08:00"
WIN_LO = datetime.fromisoformat(f"{DAY} 05:55:00{TZ}")
WIN_HI = datetime.fromisoformat(f"{DAY} 06:05:00{TZ}")
BURST_SAMPLES = (
    datetime.fromisoformat(f"{DAY} 06:00:15{TZ}"),
    datetime.fromisoformat(f"{DAY} 06:00:30{TZ}"),
)
BURST_THRESHOLD = 40  # onsets in a single 15s sample to call it a correlated burst


def _node(canonical):
    return canonical.split("#", 1)[0]


def _canon(raw):
    node, _, idx = raw.rpartition("-")
    return f"{node}#{idx}"


def stream_all_onsets(xid_csv):
    """Full-file empty-aware onset list: [(t, gpu_canonical, code, node)]."""
    onsets = []
    with open(xid_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        gpus = [_canon(h) for h in header[1:]]
        state = [None] * len(gpus)  # None | "idle" | 0.0 | fault-code
        for row in reader:
            t = None
            for i, cell in enumerate(row[1:]):
                prev = state[i]
                if cell == "":
                    state[i] = "idle"
                    continue
                v = float(cell)
                if v != 0.0:
                    if prev == 0.0 or prev == "idle":
                        if t is None:
                            t = datetime.fromisoformat(row[0])
                        onsets.append((t, gpus[i], v, _node(gpus[i])))
                    state[i] = v
                else:
                    state[i] = 0.0
    onsets.sort(key=lambda e: e[0])
    return onsets


def affected_gpu_columns(xid_csv):
    """Map canonical id -> column index, restricted to burst-affected GPUs.

    Second empty-aware pass: re-derives the GPUs that onset in the two burst
    samples (so recovery/precursor passes can restrict to just those columns).
    """
    affected = set()
    with open(xid_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        gpus = [_canon(h) for h in header[1:]]
        state = [None] * len(gpus)
        for row in reader:
            t = datetime.fromisoformat(row[0]) if row else None
            in_burst = t in BURST_SAMPLES
            for i, cell in enumerate(row[1:]):
                prev = state[i]
                if cell == "":
                    state[i] = "idle"
                    continue
                v = float(cell)
                if v != 0.0:
                    if (prev == 0.0 or prev == "idle") and in_burst:
                        affected.add(gpus[i])
                    state[i] = v
                else:
                    state[i] = 0.0
    return affected


def recovery_stats(xid_csv, affected):
    """For each affected GPU, latch duration after its burst onset.

    Walks each affected GPU's post-burst samples: counts consecutive nonzero
    (latched) 15s samples until it clears to 0, goes idle (empty), or the window
    ends. Returns per-GPU dict and a summary Counter of terminal states.
    """
    with open(xid_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        gpus = [_canon(h) for h in header[1:]]
        idx = {g: i for i, g in enumerate(gpus)}
        cols = {g: idx[g] for g in affected if g in idx}
        # per-gpu: list of (t, cell) after first burst sample
        series = defaultdict(list)
        for row in reader:
            t = datetime.fromisoformat(row[0])
            if t < BURST_SAMPLES[0]:
                continue
            for g, ci in cols.items():
                cell = row[ci + 1] if ci + 1 < len(row) else ""
                series[g].append((t, cell))
    durations = []
    terminal = Counter()
    for g, samples in series.items():
        # find first nonzero (the onset/latch), then count latched run
        run = 0
        started = False
        term = "faulted-to-window-end"
        for _, cell in samples:
            if not started:
                if cell not in ("", "0", "0.0"):
                    started = True
                    run = 1
                continue
            if cell == "":
                term = "went-idle"
                break
            if float(cell) == 0.0:
                term = "cleared-to-0"
                break
            run += 1
        if started:
            durations.append(run)
            terminal[term] += 1
    return durations, terminal


def precursor_stats(metric_csv, affected):
    """Per-affected-GPU values in [WIN_LO, first burst sample]; trend summary."""
    with open(metric_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        gpus = [_canon(h) for h in header[1:]]
        idx = {g: i for i, g in enumerate(gpus)}
        cols = {g: idx[g] for g in affected if g in idx}
        series = defaultdict(list)
        for row in reader:
            t = datetime.fromisoformat(row[0])
            if t < WIN_LO:
                continue
            if t > BURST_SAMPLES[0]:
                break
            for g, ci in cols.items():
                cell = row[ci + 1] if ci + 1 < len(row) else ""
                if cell != "":
                    series[g].append((t, float(cell)))
    deltas = []
    for g, vals in series.items():
        if len(vals) >= 2:
            deltas.append(vals[-1][1] - vals[0][1])
    return series, deltas


def job_coincidence(trace_csv, centers, w_minutes):
    """FAILED trace_kalos jobs whose UTC fail_time is within +/- W of a burst center.

    trace fail_time is UTC; the burst centers are +08:00. fromisoformat keeps both
    tz-aware so the comparison is timezone-correct.
    """
    w = timedelta(minutes=w_minutes)
    jobs = []
    with open(trace_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("state") != "FAILED":
                continue
            ft = (row.get("fail_time") or "").strip()
            if not ft or ft.lower() == "nan":
                continue
            try:
                jobs.append((row["job_id"], datetime.fromisoformat(ft)))
            except ValueError:
                continue
        matches = [
            (jid, ft) for jid, ft in jobs
            if any(abs(ft - c) <= w for c in centers)
        ]
    return len(jobs), matches


def main(root):
    import os
    os.chdir(root)
    xid = f"{KALOS}/XID_ERRORS.csv"

    print("=" * 70)
    print("AUG-17 06:00 CORRELATED XID BURST — characterization (READ-ONLY)")
    print("=" * 70)

    onsets = stream_all_onsets(xid)
    print(f"\n[pass A] total empty-aware onsets over full file: {len(onsets)}")

    # --- AC1: onset timeline around the burst -------------------------------
    by_t = Counter(t for t, _, _, _ in onsets)
    print(f"\n## AC1 Onset timeline {DAY} 05:55..06:05 (per 15s sample)")
    t = WIN_LO
    while t <= WIN_HI:
        c = by_t.get(t, 0)
        mark = "  <== BURST" if t in BURST_SAMPLES else ""
        print(f"   {t.isoformat()}  onsets={c}{mark}")
        t += timedelta(seconds=15)

    # --- AC2: scope (union-dedup) -------------------------------------------
    burst = [(t, g, c, n) for (t, g, c, n) in onsets if t in BURST_SAMPLES]
    gpus = {g for _, g, _, _ in burst}
    nodes = {n for _, _, _, n in burst}
    per_node = Counter(n for _, _, _, n in burst)
    dist = Counter(per_node.values())
    print(f"\n## AC2 Burst scope (union of {BURST_SAMPLES[0].time()} + {BURST_SAMPLES[1].time()})")
    print(f"   distinct GPUs={len(gpus)}  distinct nodes={len(nodes)}")
    print(f"   GPUs-per-node distribution (gpus_on_node: n_nodes): {dict(sorted(dist.items()))}")
    print(f"   max GPUs affected on any single node: {max(per_node.values())} of 8")

    # --- AC3: code mix ------------------------------------------------------
    codes = Counter(int(c) for _, _, c, _ in burst)
    print(f"\n## AC3 Code mix over the burst: {dict(sorted(codes.items(), key=lambda kv: -kv[1]))}")

    # --- AC7: recurring bursts (cluster-wide) -------------------------------
    print(f"\n## AC7 Recurring correlated bursts (any 15s sample with >= {BURST_THRESHOLD} onsets)")
    big = sorted([(t, c) for t, c in by_t.items() if c >= BURST_THRESHOLD])
    # merge consecutive 15s samples into bursts
    groups = []
    for t, c in big:
        if groups and (t - groups[-1][-1][0]) <= timedelta(seconds=15):
            groups[-1].append((t, c))
        else:
            groups.append([(t, c)])
    for grp in groups:
        gp_nodes = {n for (tt, _, _, n) in onsets if any(tt == t for t, _ in grp)}
        total = sum(c for _, c in grp)
        span = f"{grp[0][0].isoformat()} .. {grp[-1][0].time()}"
        print(f"   {span}  samples={len(grp)} onsets={total} nodes={len(gp_nodes)}")

    # --- AC4: recovery / latch behavior -------------------------------------
    affected = affected_gpu_columns(xid)
    durations, terminal = recovery_stats(xid, affected)
    if durations:
        durations.sort()
        n = len(durations)
        med = durations[n // 2]
        print(f"\n## AC4 Recovery/latch (affected GPUs={len(affected)})")
        print(f"   latched 15s-sample run: min={min(durations)} median={med} "
              f"max={max(durations)} (x15s = seconds)")
        print(f"   terminal state: {dict(terminal)}")

    # --- AC6: precursors ----------------------------------------------------
    for metric in ("GPU_TEMP", "POWER_USAGE"):
        series, deltas = precursor_stats(f"{KALOS}/{metric}.csv", affected)
        if deltas:
            deltas.sort()
            rising = sum(1 for d in deltas if d > 0)
            print(f"\n## AC6 Precursor {metric} (~5 min before onset, affected GPUs with >=2 samples={len(deltas)})")
            print(f"   delta(last-first) min={deltas[0]:.1f} "
                  f"median={deltas[len(deltas)//2]:.1f} max={deltas[-1]:.1f}; "
                  f"rising={rising}/{len(deltas)}")
        else:
            print(f"\n## AC6 Precursor {metric}: no pre-onset samples for affected GPUs")

    # --- AC5: temporal job correlation --------------------------------------
    for w in (5, 30):
        total, matches = job_coincidence(TRACE, BURST_SAMPLES, w)
        print(f"\n## AC5 Job coincidence (FAILED jobs total={total}, W=+/-{w}min, "
              f"burst centers in +08:00 vs UTC trace)")
        print(f"   FAILED jobs within window: {len(matches)}")
        for jid, ft in matches[:10]:
            print(f"      {jid}  fail_time(UTC)={ft.isoformat()}")

    print("\n" + "=" * 70)
    print("END — all numbers from READ-ONLY streaming passes; re-run to reproduce.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
