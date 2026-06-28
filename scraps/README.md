# EvoSentinel DC

**Self-Improving Multi-Stage Agentic System for Data Center GPU Resilience**

## Hackathon Project for 2026 AI Engineer World's Fair

Built with Gemini 3.5 / Antigravity for the **Self-Improvement Stack** theme. Leverages DCGM-inspired GPU telemetry.

## Recent Improvements
- Enhanced simulator with richer DCGM metrics (memory, NVLink proxy, fan effort, throttling reasons, trend buffering).
- Stronger self-improver with persistent edit skeletons and validation replay.
- Monitoring loop with history buffering for better trend detection.
- Expanded documentation with partner leverage plan.

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

## Real-World Data
Ground the simulator/classifier in production GPU-cluster telemetry:

```bash
scripts/download_datasets.sh        # PAI 2020 + Acme (default)
scripts/download_datasets.sh all    # + Philly (best-effort) + extra Alibaba traces
scripts/download_datasets.sh --list # show targets
```

Datasets land in `data/` (gitignored, ~4 GB). PAI's per-worker GPU/mem
utilization (`pai_sensor_table`) is the closest match to our DCGM telemetry;
Acme adds LLM-cluster failure/queue dynamics. **See [docs/DATA.md](docs/DATA.md)
for full schema, sizes, and which trace to use.**

## Key Files
- simulator.py: Enhanced DCGM GPU metrics + env modes.
- classifier.py: Lightweight failure classifier.
- main_loop.py: Orchestration skeleton.
- self_improver.py: Reflection skeleton.
- scripts/download_datasets.sh: Fetch real GPU-cluster traces into `data/`.
- docs/DATA.md: What each dataset contains and how to load it.
- ARCHITECTURE.md, PLAN.md, SKILL.md, AGENTS.md.

## Leveraging Hackathon Partner Resources
- **Modular (MAX + Mojo)**: Use MAX guides and Modular Agent Skills for optimized simulator components or Mojo-based metric processing/remediation. Great for heterogeneous compute performance.
- **Antigravity (Google DeepMind)**: Core for persistent self-edits via env_id sandbox.
- **Gemini/Gemma**: Enhance classifier with Gemma (local), Gemini Live for voice alerts.
- **LiveKit**: Add voice/video interfaces for ops alerts/commands.
- **Digital Ocean**: $200 credits for hosting/scaling demo deployment.
- **MongoDB Atlas**: Persistent storage for telemetry history and improvement logs.
- **MiniMax**: Multimodal extensions (e.g., vision for physical monitoring).

See PLAN.md for integration priorities.

## Demo Highlights for Judges
- Live multi-stage pipeline with realistic failures.
- Visible self-improvement (before/after on novel GPU failures).
- Partner integrations roadmap.
- High technicality + originality in critical infrastructure domain.

Public repo for submission. Strong foundation after improvements.