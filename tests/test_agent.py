from gpusitter.agent.agent import build_agent


def test_agent_has_tools_and_priors():
    a = build_agent()
    names = {t.__name__ for t in a.tools}
    assert {
        "get_telemetry",
        "find_correlated_failures",
        "search_past_incidents",
        "page_technician",
        "record_resolution",
    } <= names
    assert "NODE_FAIL" in a.instruction  # priors injected


def test_agent_has_ml_tools_and_xid_honesty():
    a = build_agent()
    names = {t.__name__ for t in a.tools}
    assert {"get_sensory", "train_and_validate"} <= names
    assert "never assert a specific Xid" in a.instruction


def test_instruction_states_onset_miss_model_not_absent_telemetry():
    instr = build_agent().instruction.lower()
    # The incident model is the empty-aware Xid onset surfaced as an i6k miss.
    assert "onset" in instr
    assert "miss" in instr
    # Must NOT imply Xid telemetry is unavailable/absent.
    assert "no xid" not in instr
    assert "xid telemetry is absent" not in instr
    assert "xid data is absent" not in instr


def test_instruction_distinguishes_observed_from_inferred():
    instr = build_agent().instruction
    low = instr.lower()
    assert "observed" in low and "infer" in low
    # The observed-Xid path must direct the agent to the xid onset cohort tool.
    assert 'source="xid"' in instr or "source='xid'" in instr
