# GPUSitter Skills (Multi-Stage)

## Classifier Skills (Lightweight)
- Fast triage of telemetry batches.
- Output structured JSON with failure type and confidence.

## Analyzer Skills
- monitor_telemetry: Ingest and baseline DCGM metrics.
- perform_rca: Correlate GPU temp/util/power/ECC across time and servers.
- plan_and_execute_remediation: Safe actions in sim (throttle clocks, load migrate conceptually, isolate).

## Self-Improver Skills
- reflect_and_self_update: Review full trace + outcome metrics. Critique (e.g. missed early signals). Propose + apply updates to prompts, thresholds, or new functions in SKILL.md / playbooks.
- Validate improvements by simulating replay.

## Environment Profiling (Cross-Stage)
- Adapt monitoring for arid (dust proxy via fan effort + temp delta) or coastal (humidity correlations).

All skills support structured outputs and persistent state via Antigravity env_id.