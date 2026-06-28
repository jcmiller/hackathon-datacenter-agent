"""Timestamp seam: parse Time / fail_time as numeric OR ISO, fail-loud on bad.

The trace + telemetry carry two timestamp encodings: relative-second floats
(mock fixtures, some processed pkls) and wall-clock ISO with offset (real Kalos /
AcmeTrace, e.g. ``2023-08-15 15:30:15+08:00``). Parsing must accept both — and
must NEVER silently drop a row it can't parse, or real-data windows quietly come
back empty (samples:0). See bead 65e (timestamp seam).
"""

from datetime import datetime, timedelta

TimeValue = float | datetime


def parse_time_value(value) -> TimeValue:
    """Parse a Time/fail_time cell to ``float`` (relative) or tz-aware ``datetime``.

    Raises ``ValueError`` on genuinely-bad input — there is no sentinel return, so
    callers cannot silently skip a malformed-but-present row.
    """
    if isinstance(value, (int, float, datetime)):
        return value
    s = str(value).strip()
    if s == "":
        raise ValueError("empty timestamp")
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(s)  # handles 'YYYY-MM-DD HH:MM:SS+08:00'
    except ValueError as exc:
        raise ValueError(f"unparseable timestamp: {value!r}") from exc


def window_bounds(center, window):
    """Return ``(lo, hi) = center ± window``.

    ``window`` is in seconds; for an ISO/datetime center it becomes a timedelta so
    the bounds stay the same type as the Time column they'll be compared against.
    """
    c = parse_time_value(center)
    if isinstance(c, datetime):
        delta = timedelta(seconds=window)
        return c - delta, c + delta
    return c - window, c + window
