# GPUSitter Multi-Stage Agentic Architecture

## High-Level Flow
```
[Simulator Telemetry Stream]
         |
         v
[Monitoring Loop]  -->  Batch metrics (DCGM fields: GPU util, temp, power, ECC, clocks...)
         |
         v
[Classifier (Small Gemini)] --> Classify: Normal | Failure Type (Overheat | Voltage | ECC | Power | Dust | ...)
         | (if failure)
         v
[Analyzer Agent (Antigravity)] --> Deep RCA, correlation, impact, remediation plan
         |
         v
[Remediation Executor] --> Apply in sim (throttle, isolate, adjust)
         |
         v
[Self-Improvement Agent] --> Reflect on outcome (latency, success), critique, update SKILL.md / playbooks / thresholds
         ^
         | (persistent state via env_id)
         +-- Loop back to Monitoring
```

## Stage Details

### 1. Monitoring Loop
- Python main loop polling simulator.get_telemetry().
- Buffers recent history for context.
- Triggers classifier periodically or on thresholds.

### 2. Classifier
- Lightweight Gemini call (low temp, fast model).
- Prompt: "Classify this telemetry batch. Output JSON: {failure: bool, type: str, confidence: float, reason: str}"
- Fast path for real-time.

### 3. Analyzer
- Antigravity interaction with full SKILL.md (monitor, rca, remediate).
- Uses code execution in sandbox to analyze pandas DataFrames of metrics.
- Stateful for context across incidents.

### 4. Self-Improver
- Separate or chained prompt: Review full trace + metrics (detection time, remediation efficacy).
- Identifies gaps (e.g., "classifier missed early ECC trend").
- Applies updates: Edit SKILL.md sections, add functions, adjust classifier prompts.
- Validates by re-running similar scenario.

## Benefits for Self-Improvement Stack
- Modular stages allow targeted improvement (e.g., better classifier rules or analyzer heuristics).
- Persistent state in Antigravity sandbox enables skill evolution without external DB initially.
- Demonstrates recursive/continual improvement in critical ops domain.

## Implementation Notes
- Start with mocks in Python.
- Use Gemini API directly for classifier; Antigravity for heavier stages.
- Export structured logs for demo visibility.

Update with code examples as implemented.