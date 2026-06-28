from backend.agent import build_agent


def test_agent_has_tools_and_priors():
    a = build_agent()
    names = {t.__name__ for t in a.tools}
    assert {"get_telemetry", "find_correlated_failures", "search_past_incidents",
            "page_technician", "record_resolution"} <= names
    assert "NODE_FAIL" in a.instruction  # priors injected
