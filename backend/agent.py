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


def build_agent():
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


def triage(incident: dict) -> str:
    runner = InMemoryRunner(agent=build_agent(), app_name="rca")
    session = runner.session_service.create_session_sync(app_name="rca", user_id="demo")
    msg = types.Content(
        role="user",
        parts=[types.Part(text=f"Incident fired: {incident}. Triage it.")],
    )
    final = ""
    for ev in runner.run(user_id="demo", session_id=session.id, new_message=msg):
        if ev.content and ev.content.parts:
            for part in ev.content.parts:
                if getattr(part, "text", None):
                    final = part.text
    return final
