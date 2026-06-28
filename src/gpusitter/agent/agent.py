import re
from collections.abc import Generator

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

from . import tools
from .incident import format_incident_prompt
from .priors import DOMAIN_PRIORS

INSTRUCTION = f"""You are the on-call engineer for a GPU training cluster with a growing
memory of past incidents.
An incident is an empty-aware per-GPU Xid ONSET, often delivered as an i6k MISS (a
real onset the early-detection predictor failed to alert on). The incident message
separates OBSERVED FACTS (what a sensor/trace emitted) from what you must INFER:
- If an Xid code is OBSERVED, treat it as ground truth and thread it into your
  find_correlated_failures(source="xid") and search_past_incidents calls.
- If the Xid code is NOT directly observed, infer the likely fault class from priors +
  telemetry, and confirm the onset cohort with find_correlated_failures(source="xid").

Triage it in order:
1. Call get_telemetry to see GPU power/temp around the onset time.
2. Call check_degradation_trend to examine the hours BEFORE the onset — was this sudden
   or a slow build-up?
3. Call find_correlated_failures with source="xid" to recover the observed Xid onset
   cohort + code at the onset time, then (if useful) source="jobs" for job-level spread.
4. Call search_past_incidents with a RICH natural-language description combining:
   the observed/inferred Xid code, telemetry values, degradation pattern, and correlated
   count, but never assert a specific Xid code as fact unless a tool returns one.
   The search is semantic — describe what you see, not just the job type.
   Reference any similar past cases returned.
5. Decide disposition: escalate_to_ops (shared-cause cluster), page_technician (isolated hw fault),
   or restart_and_watch (healthy telemetry + no history of recurrence).
   Call page_technician if hardware replacement is needed.
6. Call record_resolution. Pass incident_id, and the exact numeric values from steps 2 and 3:
   power_spike_ratio, temp_rise_C, correlated_count. These metrics train the predictor.
   In the summary, note pre-onset degradation signals so future triage can predict
   onsets earlier — especially when this incident arrived as a predictor MISS.
7. Call train_and_validate (model_type="logreg") to fit the failure-disposition classifier on all
   accumulated SOP entries. Report the val_auc and whether a new model version was promoted.
Ground every claim in a tool return value. Note if this onset matches a known pattern.

{DOMAIN_PRIORS}"""


def build_agent():
    return Agent(
        name="oncall_rca",
        model="gemini-2.5-flash",
        instruction=INSTRUCTION,
        tools=[
            tools.get_telemetry,
            tools.check_degradation_trend,
            tools.find_correlated_failures,
            tools.search_past_incidents,
            tools.page_technician,
            tools.record_resolution,
            tools.get_sensory,
            tools.train_and_validate,
        ],
    )


def triage_stream(incident: dict) -> Generator[dict]:
    """Yield agent events as the ADK runner fires them. Runs synchronously."""
    runner = InMemoryRunner(agent=build_agent(), app_name="rca")
    session = runner.session_service.create_session_sync(app_name="rca", user_id="demo")
    msg = types.Content(
        role="user",
        parts=[types.Part(text=format_incident_prompt(incident))],
    )

    tools._pending_updates.clear()
    final_text = ""
    ticket_num = None
    all_obs_text = ""

    for ev in runner.run(user_id="demo", session_id=session.id, new_message=msg):
        for tc in ev.get_function_calls():
            yield {"type": "tool_call", "tool": tc.name, "args": str(tc.args)}

        responses = ev.get_function_responses()
        for tr in responses:
            obs_text = str(tr.response)
            all_obs_text += obs_text
            yield {"type": "observation", "text": obs_text}

        if responses:
            for upd in tools._pending_updates:
                yield {"type": "file_update", "path": upd["path"], "entry": upd["entry"]}
            tools._pending_updates.clear()

        if ev.content and ev.content.parts:
            for part in ev.content.parts:
                if getattr(part, "text", None):
                    final_text += part.text
                    all_obs_text += part.text
                    yield {"type": "observation", "text": part.text}

    for t_match in re.finditer(r"TKT-\d+", all_obs_text):
        ticket_num = t_match.group(0)
        break

    disp = "RESTART_AND_WATCH"
    if "escalate" in final_text.lower():
        disp = "ESCALATE_TO_OPS"
    elif ticket_num or re.search(r"\bpage\b.*technician", final_text.lower()):
        disp = "PAGE_TECHNICIAN"

    yield {
        "type": "disposition",
        "disposition": disp,
        "summary": final_text[:120] + "...",
        "action": final_text,
        "ticket": ticket_num,
    }


def triage(incident: dict) -> str:
    """Synchronous wrapper returning final disposition string."""
    for ev in triage_stream(incident):
        if ev.get("type") == "disposition":
            return ev["disposition"]
    return "RESTART_AND_WATCH"
