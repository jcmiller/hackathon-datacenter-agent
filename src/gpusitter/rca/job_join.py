"""Temporal RCA join: do FAILED jobs coincide with diagnosable GPU faults?

trace_kalos records job failures (``fail_time``) but NOT which physical GPUs a
job held (only counts), and its timestamps are UTC while the DCGM telemetry is
+08:00. So a true per-GPU attribution isn't supported by the data; instead we
quantify TEMPORAL coincidence: for each FAILED job, did an edge-detected Xid
onset fire cluster-wide within +/- W minutes of its fail_time? Cross-timezone
comparison is automatic because both sides are parsed as tz-aware datetimes.

Reuses q2o's store + w28's onset semantics (no raw re-ingest).
"""

import csv
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from ..telemetry.timeparse import parse_time_value


@dataclass
class FailedJob:
    job_id: str
    fail_dt: datetime  # tz-aware


def _parse(ts: str) -> datetime:
    # Handles "2023-08-29 13:57:15+08:00" and "...+00:00" (py>=3.11).
    return datetime.fromisoformat(ts)


# Hardware/job failure states (loader parity).
FAIL_STATES = {"NODE_FAIL", "FAILED"}
_INCIDENT_KEEP = (
    "job_id", "type", "node_num", "gpu_num", "state", "fail_time", "duration", "user",
)


def load_failed_jobs(trace_csv: str, states=FAIL_STATES) -> List[FailedJob]:
    """Failed jobs with a parseable ISO fail_time, as tz-aware records.

    The datetime/real-trace path (used by the coincidence analysis); ``states``
    defaults to FAILED+NODE_FAIL. See :func:`load_incidents` for the numeric
    fail_time path the agent tools use.
    """
    jobs = []
    with open(trace_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("state") not in states:
                continue
            ft = (row.get("fail_time") or "").strip()
            if not ft:
                continue
            try:
                fail_dt = _parse(ft)
            except ValueError:
                continue  # numeric/relative fail_time — not the ISO analysis path
            jobs.append(FailedJob(job_id=row["job_id"], fail_dt=fail_dt))
    return jobs


def load_incidents(trace_csv: str) -> List[dict]:
    """Failure incidents (NODE_FAIL|FAILED, fail_time present), sorted by fail_time.

    Numeric fail_time (relative seconds), dict records — the shape the agent's
    tools consume. Pandas-free port of backend/loader.load_incidents.
    """
    out = []
    with open(trace_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("state") not in FAIL_STATES:
                continue
            ft = (row.get("fail_time") or "").strip()
            if ft == "" or ft.lower() == "nan":
                continue  # legitimately-absent fail_time, not malformed
            # numeric (relative) OR ISO (real AcmeTrace); raises on bad -> fail loud
            rec = {k: row.get(k) for k in _INCIDENT_KEEP}
            rec["fail_time"] = parse_time_value(ft)
            out.append(rec)
    out.sort(key=lambda r: r["fail_time"])
    return out


def correlated_jobs(incidents, fail_time, window):
    """Other failure incidents within +/- ``window`` of ``fail_time`` (excludes self).

    Job<->job temporal correlation (port of loader.correlated_failures) — distinct
    from the Xid-onset cohort finder. ``fail_time`` may be numeric or ISO; for an
    ISO/datetime center ``window`` (seconds) becomes a timedelta so the comparison
    stays type-consistent with the incidents' parsed fail_time.
    """
    center = parse_time_value(fail_time)
    span = timedelta(seconds=window) if isinstance(center, datetime) else window
    return [
        i for i in incidents
        if i["fail_time"] != center and abs(i["fail_time"] - center) <= span
    ]


def xid_onset_events(store) -> List[Tuple[datetime, str]]:
    """All edge-detected Xid onsets in the store as sorted (datetime, gpu)."""
    events = []
    for g in store.gpus():
        prev = None
        for ts, val in store.series("XID_ERRORS", g):  # sorted by time
            # Require an observed prior zero — a missing prev (truncated history)
            # must not be counted as an onset (matches stream_xid_onsets).
            if val != 0.0 and prev == 0.0:
                events.append((_parse(ts), g))
            prev = val
    events.sort(key=lambda e: e[0])
    return events


def stream_xid_onset_records(xid_csv: str) -> List[Tuple[datetime, str, float]]:
    """Edge-detected Xid onsets streamed from a wide XID CSV, WITH the fault code.

    Same empty-aware, memory-bounded edge detection as :func:`stream_xid_onsets`
    (one row + one per-GPU state at a time, scaling to the full ~750 MB kalos XID
    file), but each event also carries the observed Xid CODE at onset — the
    diagnosable signal the runtime RCA tools surface. An onset is a transition
    INTO a fault: the sample at t is nonzero and the GPU's prior OBSERVED state
    was non-fault (healthy 0.0 or idle empty cell). Excluded:
    - latched faults (nonzero -> nonzero), the dominant kalos pattern;
    - a GPU whose very first observation is already nonzero (pre-trace history
      unknown — could have faulted before the window).
    Reading raw (not via the empty-skipping long-record path) is essential: many
    kalos GPUs go idle->fault without ever recording a 0.0, so dropping empties
    would lose those real onsets.
    """
    import csv as _csv

    from ..telemetry.normalize import parse_gpu_id

    events = []
    with open(xid_csv, newline="") as fh:
        reader = _csv.reader(fh)
        header = next(reader)
        gpus = [parse_gpu_id(h).canonical for h in header[1:]]
        # per-GPU last observed state: None (unseen), "idle", 0.0, or a fault code
        state = [None] * len(gpus)
        for row in reader:
            t = None
            for i, cell in enumerate(row[1:]):
                prev = state[i]
                if cell == "":
                    state[i] = "idle"
                    continue
                v = float(cell)
                if v != 0.0:
                    if prev == 0.0 or prev == "idle":  # non-fault -> fault
                        if t is None:
                            t = _parse(row[0])
                        events.append((t, gpus[i], v))
                    state[i] = v
                else:
                    state[i] = 0.0
    events.sort(key=lambda e: e[0])
    return events


def stream_xid_onsets(xid_csv: str) -> List[Tuple[datetime, str]]:
    """Edge-detected Xid onsets streamed from a wide XID CSV as ``(datetime, gpu)``.

    Projection of :func:`stream_xid_onset_records` that drops the fault code;
    kept as the stable contract for callers that only need onset timing/location
    (e.g. the coincidence denominator). See that function for the edge-detection
    semantics (empty-aware, latched-fault and first-sample exclusions).
    """
    return [(t, gpu) for (t, gpu, _code) in stream_xid_onset_records(xid_csv)]


def xid_sample_span(xid_csv: str) -> Optional[Tuple[datetime, datetime]]:
    """(first, last) timestamp of the XID CSV — the telemetry SAMPLE coverage.

    This is the window during which telemetry exists, independent of when faults
    onset. The onset span is narrower (no onset before the first / after the last
    transition); using it as the denominator would wrongly drop FAILED jobs that
    fall in the sample window but outside the onset span. Rows are time-sorted, so
    the first and last data rows bound the span; the tail is read via a small
    seek rather than scanning the whole ~750 MB file.
    """
    with open(xid_csv, "rb") as fh:
        fh.readline()  # header
        first_line = fh.readline()
        if not first_line:
            return None
        first = first_line.decode("utf-8", "ignore").split(",", 1)[0].strip()
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 65536))
        tail = fh.read().decode("utf-8", "ignore").strip().splitlines()
        last = tail[-1].split(",", 1)[0].strip() if tail else first
    return (_parse(first), _parse(last))


def telemetry_span(store) -> Optional[Tuple[datetime, datetime]]:
    """(min, max) XID timestamp as tz-aware datetimes, or None if empty."""
    times = []
    for g in store.gpus():
        for ts, _ in store.series("XID_ERRORS", g):
            times.append(_parse(ts))
    if not times:
        return None
    return (min(times), max(times))


def coincidence(onsets, jobs, w_minutes: float, telemetry_span=None):
    """Match FAILED jobs to Xid onsets within +/- ``w_minutes``.

    ``onsets`` is the sorted output of :func:`xid_onset_events`. When
    ``telemetry_span`` is given, jobs whose fail_time falls outside it are
    dropped (the telemetry only covers part of the trace). Returns
    ``(results, rate)`` where rate = matched / in-window jobs.
    """
    onset_times = [e[0] for e in onsets]
    w = timedelta(minutes=w_minutes)
    results = []
    matched = 0
    in_window = 0
    for job in jobs:
        if telemetry_span is not None:
            lo, hi = telemetry_span
            if not (lo <= job.fail_dt <= hi):
                continue
        in_window += 1
        a = bisect_left(onset_times, job.fail_dt - w)
        b = bisect_right(onset_times, job.fail_dt + w)
        n = b - a
        is_match = n > 0
        if is_match:
            matched += 1
        nearest = None
        if onset_times:
            cand = min(onset_times, key=lambda ot: abs(ot - job.fail_dt))
            nearest = cand
        results.append(
            {
                "job_id": job.job_id,
                "fail_dt": job.fail_dt,
                "matched": is_match,
                "n_onsets": n,
                "nearest_onset": nearest,
            }
        )
    rate = matched / in_window if in_window else 0.0
    return results, rate
