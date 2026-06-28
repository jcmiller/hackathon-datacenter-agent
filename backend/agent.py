import re
from collections.abc import Generator

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

from backend.priors import DOMAIN_PRIORS
from backend import tools

INSTRUCTION = f"""You are the on-call engineer for a GPU training cluster with a growing memory of past incidents.
An incident just fired. Triage it in order:
1. Call get_telemetry to see GPU power/temp around the failure time.
2. Call check_degradation_trend to examine the hours BEFORE failure — was this sudden or a slow build-up?
3. Call find_correlated_failures to check for cluster-wide spread.
4. Call search_past_incidents with a RICH natural-language description combining: the Xid error code,
   telemetry values, degradation pattern, and correlated count. The search is semantic — describe
   what you see, not just the job type. Reference any similar past cases returned.
5. Decide disposition: escalate_to_ops (shared-cause cluster), page_technician (isolated hw fault),
   or restart_and_watch (healthy telemetry + no history of recurrence).
   Call page_technician if hardware replacement is needed.
6. Call record_resolution. In the summary, explicitly note any pre-failure degradation signals
   (spike ratio, temp rise) so future triage can predict similar failures earlier.
Ground every claim in a tool return value. Note if this failure matches a known pattern.

{DOMAIN_PRIORS}"""


def _build_agent():
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
        ],
    )


def triage_stream(incident: dict) -> Generator[dict, None, None]:
    """Yield agent events as the ADK runner fires them. Runs synchronously."""
    runner = InMemoryRunner(agent=_build_agent(), app_name="rca")
    session = runner.session_service.create_session_sync(app_name="rca", user_id="demo")
    msg = types.Content(
        role="user",
        parts=[types.Part(text=f"Incident fired: {incident}. Triage it.")],
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
            # Flush any file writes the tools triggered
            for upd in tools._pending_updates:
                yield {"type": "file_update", "path": upd["path"], "entry": upd["entry"]}
            tools._pending_updates.clear()

        if ev.content and ev.content.parts:
            for part in ev.content.parts:
                if getattr(part, "text", None):
                    final_text += part.text
                    all_obs_text += part.text
                    yield {"type": "observation", "text": part.text}

    # Extract ticket number from all observations (tool responses include page_technician output)
    for t_match in re.finditer(r"TKT-\d+", all_obs_text):
        ticket_num = t_match.group(0)
        break

    disp = "RESTART_AND_WATCH"
    if "escalate" in final_text.lower():
        disp = "ESCALATE_TO_OPS"
    elif ticket_num or re.search(r'\bpage\b.*technician', final_text.lower()):
        disp = "PAGE_TECHNICIAN"

    yield {
        "type": "disposition",
        "disposition": disp,
        "summary": final_text[:120] + "...",
        "action": final_text,
        "ticket": ticket_num,
    }
