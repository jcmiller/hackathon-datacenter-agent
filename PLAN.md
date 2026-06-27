# EvoSentinel DC Hackathon Plan & Notes

## Project Overview
Self-improving AI agent (Antigravity/Gemini 3.5) for data center hardware monitoring/remediation. Focus: GPU Metric MCP using DCGM-inspired telemetry.

## Theme Alignment
- Primary: The Self-Improvement Stack (evaluation, monitoring, upgrade infrastructure via persistent skills + reflection).
- Bonus: Continual Learning from simulated incidents.

## Key Components
1. **Data Center Simulator** (Python): Servers/racks with GPU components (DCGM-style metrics: temp, power, utilization, ECC, clocks, NVLink).
2. **Antigravity Agent**: Stateful sandbox with AGENTS.md/SKILL.md. Skills for telemetry, RCA, remediation, env profiling, self-update.
3. **GPU MCP Focus**: Mock/realistic NVIDIA DCGM telemetry. Failures: GPU thermal throttling, power spikes, ECC errors, dust-affected cooling under AI load.
4. **Self-Improvement**: Post-incident reflection → edit SKILL.md/playbooks → measurable improvement on replay.

## Failure Modes (DCGM-relevant)
- Overheating (GPU core/memory under high load).
- Voltage/power instability (draw spikes, limits).
- Environment-specific: Dust (arid → higher fan effort), humidity effects.
- GPU-specific: ECC uncorrectable, XID errors, NVLink degradation.

## Demo Flow
1. Init agent + sim.
2. Standard scenario (successful remediation).
3. Novel GPU failure (initial struggle).
4. Reflection + skill update (visible diff).
5. Re-run: Improved handling.

## Milestones (June 27-28)
- [ ] Simulator + GPU metrics (done initial).
- [ ] Antigravity integration + stateful loop.
- [ ] Self-update mechanism + validation.
- [ ] Env profiler + DCGM field mocks.
- [ ] Polished demo script + traces.
- [ ] Submission video/README polish.

## Tech Stack
- Gemini Managed Agents (Antigravity preview).
- Python simulator (mock DCGM).
- Optional: dcgm-exporter patterns, Prometheus-style metrics.

## Risks & Mitigations
- Scope: Prioritize 3-4 GPU failures + 1 self-update demo.
- Sandbox limits: Pure Python mocks first.
- Time: Focus on live demo story.

## References
- DCGM: https://github.com/NVIDIA/DCGM
- Hackathon Guide (themes, prizes, rules).

Update this file as we progress.