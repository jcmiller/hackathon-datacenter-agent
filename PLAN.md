# EvoSentinel DC Hackathon Plan & Notes

## Project Overview
Self-improving AI agent (Antigravity/Gemini 3.5) for data center hardware monitoring/remediation. Focus: GPU Metric MCP using DCGM-inspired telemetry.

**Multi-Stage Agentic System (New Focus)**:
- **Stage 1: Monitoring Loop** - Continuous ingestion of simulator telemetry (DCGM-style GPU metrics: utilization, temp, power, ECC, clocks, etc.).
- **Stage 2: Classifier** - Lightweight/fast Gemini model (e.g., Gemini 1.5 Flash or Flash-Lite) to quickly classify telemetry batches: normal vs. failure type (overheat, voltage, ECC, power quality, dust proxy, etc.). Low latency for real-time loop.
- **Stage 3: Analyzer Agent** - Deeper Antigravity-managed agent for root cause analysis (RCA), correlation across metrics, impact assessment.
- **Stage 4: Self-Improvement Agent** - Dedicated or chained agent that performs post-incident reflection, critiques performance, updates SKILL.md/playbooks/code snippets in persistent state, and validates improvements.

Orchestration: Main Python loop feeds telemetry -> Classifier -> (if failure) Analyzer -> Remediation -> Self-Improver. All leveraging Gemini API / Antigravity for statefulness.

## Theme Alignment
- Primary: The Self-Improvement Stack (evaluation, monitoring, upgrade infrastructure via persistent skills + reflection).
- Bonus: Continual Learning from simulated incidents.

## Key Components
1. **Data Center Simulator** (Python): Servers/racks with GPU components (DCGM-style metrics: temp, power, utilization, ECC, clocks, NVLink). Realistic failure injection.
2. **Classifier Module**: Small Gemini model for fast failure detection/classification.
3. **Analyzer Agent**: Stateful Antigravity for deep RCA and remediation planning.
4. **Self-Improver**: Reflection + skill evolution loop (edits files in sandbox).
5. **GPU MCP Focus**: Mock/realistic NVIDIA DCGM telemetry. Failures: GPU thermal throttling, power spikes, ECC errors, dust-affected cooling under AI load.

## Failure Modes (DCGM-relevant)
- Overheating (GPU core/memory under high load).
- Voltage/power instability (draw spikes, limits).
- Environment-specific: Dust (arid → higher fan effort), humidity effects.
- GPU-specific: ECC uncorrectable, XID errors, NVLink degradation.

## Demo Flow
1. Init simulator + monitoring loop.
2. Run normal + standard failure (classifier detects, analyzer handles).
3. Novel/complex GPU failure (classifier flags, analyzer struggles initially).
4. Self-improver activates: reflection, updates classifier thresholds or analyzer skills/playbooks.
5. Re-run scenario: Faster/better classification + analysis (measurable improvement).

## Milestones (June 27-28, Iterations)
- Iteration 1: Multi-stage architecture docs + classifier skeleton + updated simulator.
- Iteration 2: Analyzer integration with Antigravity.
- Iteration 3: Self-improver loop + file edits in state.
- Iteration 4: Full main_loop.py orchestration + traces.
- Iteration 5: Polish, env profiler, demo script, submission prep.

## Tech Stack
- Gemini API: Classifier (small model), Antigravity for Analyzer/Self-Improver.
- Python simulator (mock DCGM fields).
- Optional: dcgm-exporter patterns, Prometheus-style metrics export.

## Risks & Mitigations
- Scope: Prioritize core 4 stages + 1-2 GPU failures + visible self-update in demo.
- Sandbox/API limits: Mocks + efficient calls; use stateful env_id.
- Time: Iterative pushes; focus on live demo narrative.

## References
- DCGM: https://github.com/NVIDIA/DCGM
- Hackathon Guide (themes, prizes, rules).

Update this file as we progress. Track iterations here.