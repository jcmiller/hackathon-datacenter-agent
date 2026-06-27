# EvoSentinel Skills

## monitor_telemetry
Analyze current and historical telemetry from the simulator. Detect anomalies in temperature, voltage, power quality, fan speeds, humidity, etc.

## perform_rca
Correlate multiple metrics for root cause (e.g., high temp + rising fan RPM = possible dust or cooling failure; voltage sags with power draw spikes).

## plan_and_execute_remediation
Propose and execute safe actions in the simulator (e.g., load shed, isolate faulty component, adjust cooling).

## profile_environment
Given location/baseline or env params, identify risks (dust in arid, humidity in coastal) and add/adapt monitoring heuristics.

## reflect_and_self_update
After incident: Review full trace and outcome metrics (detection time, remediation efficacy). Critique performance, propose specific improvements (thresholds, new correlations, new functions), and apply them by editing SKILL.md or creating playbook code. Always validate changes.