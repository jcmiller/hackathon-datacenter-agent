import re
from collections.abc import Generator

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

from backend.priors import DOMAIN_PRIORS
from backend import tools

INSTRUCTION = f"""You are the on-call engineer for a GPU training cluster.
An incident just fired. Do the triage a human on-call would do:
1. Call get_telemetry to see GPU power/temp (DCGM fields) around the failure time.
2. Call find_correlated_failures to see if other nodes failed in the same window.
3. Call search_past_incidents to reuse a known resolution for this incident type.
4. Decide a disposition: escalate to datacenter ops (shared-cause cluster),
   page_technician (isolated hardware fault), or restart-and-watch (healthy telemetry).
   Page the technician via the tool if hardware replacement is needed.
5. Call record_resolution to log what you found and decided.
Ground every statement in a number a tool returned. Be concise.

{DOMAIN_PRIORS}"""


def _build_agent():
    return Agent(
        name="oncall_rca",
        model="gemini-2.5-flash",
        instruction=INSTRUCTION,
        tools=[
            tools.get_telemetry,
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

    for ev in runner.run(user_id="demo", session_id=session.id, new_message=msg):
        if ev.tool_calls:
            for tc in ev.tool_calls:
                yield {"type": "tool_call", "tool": tc.name, "args": str(tc.args)}

        if ev.tool_responses:
            for tr in ev.tool_responses:
                yield {"type": "observation", "text": str(tr.response)}

            # Flush any file writes the tools triggered
            for upd in tools._pending_updates:
                yield {"type": "file_update", "path": upd["path"], "entry": upd["entry"]}
            tools._pending_updates.clear()

        if ev.content and ev.content.parts:
            for part in ev.content.parts:
                if getattr(part, "text", None):
                    final_text += part.text
                    yield {"type": "observation", "text": part.text}

    # Extract ticket number from observations
    for t_match in re.finditer(r"TKT-\d+", final_text):
        ticket_num = t_match.group(0)
        break

    disp = "RESTART_AND_WATCH"
    if "escalate" in final_text.lower():
        disp = "ESCALATE_TO_OPS"
    elif "page" in final_text.lower() or ticket_num:
        disp = "PAGE_TECHNICIAN"

    yield {
        "type": "disposition",
        "disposition": disp,
        "summary": final_text[:120] + "...",
        "action": final_text,
        "ticket": ticket_num,
    }
