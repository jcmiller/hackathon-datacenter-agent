# EvoSentinel DC

**Self-Improving Multi-Stage Agentic System for Data Center GPU Resilience**

## Hackathon Project for 2026 AI Engineer World's Fair

Built with Gemini 3.5 / Antigravity for the **Self-Improvement Stack** theme. Leverages DCGM-inspired GPU telemetry.

## Multi-Stage Architecture
1. **Monitoring Loop** - Continuous telemetry ingestion.
2. **Classifier** (small Gemini) - Fast failure detection & typing.
3. **Analyzer** (Antigravity) - Deep RCA & remediation.
4. **Self-Improver** - Reflection + autonomous skill evolution.

See ARCHITECTURE.md for detailed flow and diagrams.

## Quick Start
1. `export GEMINI_API_KEY=...`
2. `pip install -r requirements.txt`
3. `python demo/main_loop_demo.py` (basic loop + classifier)
4. Extend with Antigravity for full analyzer/self-improver.

## Key Files
- simulator.py: Enhanced DCGM GPU metrics + env modes.
- classifier.py: Lightweight failure classifier.
- main_loop.py: Orchestration skeleton.
- self_improver.py: Reflection skeleton.
- ARCHITECTURE.md, PLAN.md, SKILL.md, AGENTS.md.

Public repo for submission. Strong foundation after 5 iterations of refinement.

## Demo Highlights for Judges
- Live multi-stage pipeline.
- Visible self-improvement (before/after on novel GPU failures).
- Realistic DCGM telemetry.
- High technicality + originality in critical infrastructure domain.