"""Canonical incident model for the RCA agent (bead 8m6).

The project's incident is an empty-aware per-GPU **Xid onset** (rca/d8z:
`stream_xid_onset_records` — a non-fault→fault transition, not a latched state)
surfaced operationally as an **i6k miss** (`detection.monitor.MissEvent`: a real
onset the incumbent predictor failed to alert on within its horizon). This module
adapts those real sources — plus legacy dashboard/job-trace dicts — into ONE
canonical schema whose defining property is an explicit split between:

- ``observed`` — GROUND TRUTH a sensor/trace actually emitted (the Xid code, the
  GPU, the onset wall-clock). ``observed.xid`` is an int only when a code was
  truly sensed; otherwise ``None`` — the agent must then INFER the fault class
  from priors + telemetry and must never assert a code.
- ``detection`` — the i6k operational context (budget/horizon/model/prior scores)
  present only when the incident came from a miss; this is why triage fired.

`triage_stream` consumes the canonical incident so the agent reasons over the
onset/miss model rather than a stale, untyped raw dict.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _xid_code(value: Any) -> int | float:
    """Render an integral Xid code as an int (43.0 -> 43); pass others through.

    Mirrors :func:`gpusitter.agent.tools._xid_code` so observed codes read the same
    whether they arrive from a tool return or an incident source.
    """
    return int(value) if isinstance(value, float) and value.is_integer() else value


def _observed_xid(value: Any) -> int | float | None:
    """Coerce a *sensed* Xid code, or ``None`` when nothing was observed.

    Absent/blank/``nan`` sentinels are unobserved, not a code: the dashboard card
    omits ``xid`` for non-Xid incidents, and the job trace carries no code at all.
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() == "nan":
            return None
        try:
            value = float(s)
        except ValueError:
            return None
    if isinstance(value, bool):  # guard: bool is an int subclass
        return None
    if isinstance(value, (int, float)):
        return _xid_code(float(value))
    return None


def _gpu_id(raw: dict) -> str | None:
    """Canonical "node-idx" GPU id from a raw incident, or ``None`` when unknown.

    The dashboard card carries ``gpu={"node","idx"}``; an already-canonical record
    carries a string. The job trace carries only node/gpu *counts*, never a per-GPU
    id, so those incidents resolve to ``None`` (the location is genuinely unobserved).
    """
    gpu = raw.get("gpu")
    if isinstance(gpu, dict):
        node, idx = gpu.get("node"), gpu.get("idx")
        if node is not None and idx is not None:
            return f"{node}-{idx}"
        return node if node is not None else None
    if isinstance(gpu, str) and gpu:
        return gpu
    return None


def _iso(t: Any) -> Any:
    """ISO-string a datetime; pass numeric/relative or already-string times through."""
    return t.isoformat() if isinstance(t, datetime) else t


def _canonical(
    *,
    incident_id: str,
    source: str,
    gpu: str | None,
    fail_time: Any,
    observed_xid: int | float | None,
    onset_t: Any,
    detection: dict | None,
    details: dict | None = None,
) -> dict:
    return {
        "incident_id": incident_id,
        "source": source,
        "gpu": gpu,
        "fail_time": _iso(fail_time),
        "observed": {
            "xid": observed_xid,
            "gpu": gpu,
            "onset_t": _iso(onset_t),
        },
        "detection": detection,
        "details": details or {},
    }


def incident_from_miss(miss) -> dict:
    """The preferred OPERATIONAL incident: an i6k miss the agent reasons over.

    A :class:`~gpusitter.detection.monitor.MissEvent` knows the onset GPU and
    wall-clock time (observed) but NOT the Xid code — so ``observed.xid`` is
    ``None`` and the agent recovers the code via ``find_correlated_failures(
    source="xid")``. The ``detection`` block records why triage fired: the
    incumbent ``model_version`` did not alert within ``horizon_s`` at ``budget``.
    """
    return _canonical(
        incident_id=f"{miss.gpu}@{_iso(miss.onset_t)}",
        source="i6k_miss",
        gpu=miss.gpu,
        fail_time=miss.onset_t,
        observed_xid=None,
        onset_t=miss.onset_t,
        detection={
            "missed": not miss.caught,
            "budget": miss.budget,
            "horizon_s": miss.horizon_s,
            "model_version": miss.model_version,
            "n_alerts_in_window": miss.n_alerts_in_window,
            "prior_scores": miss.prior_scores,
        },
        details={"pre_event_features": miss.pre_event_features},
    )


def incident_from_onset(gpu: str, t, code=None) -> dict:
    """A raw rca Xid onset ``(gpu, t[, code])``.

    When ``code`` is given (``stream_xid_onset_records`` carries it) the Xid is an
    OBSERVED fact; otherwise ``observed.xid`` stays ``None``.
    """
    return _canonical(
        incident_id=f"{gpu}@{_iso(t)}",
        source="xid_onset",
        gpu=gpu,
        fail_time=t,
        observed_xid=_observed_xid(code),
        onset_t=t,
        detection=None,
    )


def canonical_incident(raw: dict) -> dict:
    """Normalize any incoming incident dict into the canonical schema (idempotent).

    Handles already-canonical records, dashboard fixture cards (``id``/``ts``/``xid``/
    ``gpu={node,idx}``) and rca job-trace records (``job_id``/``fail_time``, no code).
    ``observed.xid`` is lifted only from a genuinely sensed ``xid``/``observed_xid``.
    """
    if isinstance(raw.get("observed"), dict) and "source" in raw:
        return raw  # already canonical

    observed_xid = _observed_xid(raw.get("observed_xid", raw.get("xid")))
    gpu = _gpu_id(raw)
    fail_time = raw.get("fail_time")
    if fail_time is None:
        fail_time = raw.get("ts", raw.get("onset_t"))
    incident_id = str(raw.get("incident_id") or raw.get("id") or raw.get("job_id") or "unknown")
    source = raw.get("source") or ("xid_onset" if observed_xid is not None else "job_trace")
    reserved = {"observed_xid", "xid", "gpu", "fail_time", "ts", "onset_t", "id", "job_id"}
    details = {k: v for k, v in raw.items() if k not in reserved}
    return _canonical(
        incident_id=incident_id,
        source=source,
        gpu=gpu,
        fail_time=fail_time,
        observed_xid=observed_xid,
        onset_t=raw.get("onset_t", fail_time),
        detection=raw.get("detection"),
        details=details,
    )


def format_incident_prompt(incident: dict) -> str:
    """The user-message text the model triages — explicit OBSERVED vs INFER split.

    When ``observed.xid`` is a code it is presented as ground truth to thread into
    ``search_past_incidents`` and ``find_correlated_failures(source="xid")``. When it
    is ``None`` the prompt states the code is *not directly observed* and instructs
    inference from priors + telemetry (never asserting a code). For an i6k miss the
    detection block frames the onset the incumbent predictor failed to catch.
    """
    inc = canonical_incident(incident)
    obs = inc["observed"]
    lines = [f"Incident {inc['incident_id']} (source: {inc['source']}). Triage it."]
    lines.append("OBSERVED FACTS (ground truth — what a sensor/trace emitted):")
    if obs["xid"] is not None:
        lines.append(
            f"- Xid code: {obs['xid']} — this is OBSERVED. Treat it as ground truth and "
            'thread it into search_past_incidents and find_correlated_failures(source="xid").'
        )
    else:
        lines.append(
            "- Xid code: not directly observed. INFER the likely fault class from priors + "
            'telemetry, and confirm the onset cohort via find_correlated_failures(source="xid"). '
            "Do NOT assert a specific Xid code as fact unless a tool returns one."
        )
    lines.append(f"- GPU: {obs['gpu'] or 'unknown'}")
    lines.append(f"- Onset time: {obs['onset_t'] or inc['fail_time']}")

    det = inc.get("detection")
    if det:
        if det.get("missed"):
            lines.append(
                "DETECTION CONTEXT (i6k operational MISS): the incumbent predictor "
                f"(v{det.get('model_version')}) did NOT alert within horizon "
                f"{det.get('horizon_s')}s at budget {det.get('budget')} — this real onset "
                "was a MISS. That early-warning gap is why you are triaging it."
            )
        else:
            lines.append(
                "DETECTION CONTEXT (i6k): the incumbent predictor "
                f"(v{det.get('model_version')}) alerted on this onset within horizon "
                f"{det.get('horizon_s')}s at budget {det.get('budget')} (caught)."
            )
    return "\n".join(lines)
