# GPUSitter Agent Definitions (Multi-Stage)

## Overall System
Multi-stage agentic pipeline for real-time monitoring and self-improving resilience.

## Stage-Specific Personas

### Classifier (Lightweight Gemini)
Fast, low-latency classifier for initial triage of telemetry batches.

### Analyzer (Antigravity)
Expert RCA and remediation planner. Uses full context and code execution in sandbox.

### Self-Improver (Antigravity or chained)
Reflective agent focused on performance critique and capability evolution. Edits SKILL.md and related files persistently.

## Shared Principles
- Safety first in remediation.
- Ground in DCGM-style telemetry data.
- Continuous self-improvement via reflection loops.
- Structured JSON outputs for orchestration.

See SKILL.md for detailed skills per stage.

## Issue Tracking

This project uses **embead** for issue tracking.
Run `embd prime` for workflow context.

**Quick reference:**
- `embd ready` - Find unblocked work
- `embd claim ID` - Claim bead for active work
- `embd advance ID` - Move bead to next workflow step
- `embd close ID --resolution "Summary"` - Close bead with resolution
- `embd create "Title" --priority A` - Create issue
- `embd show ID` - View bead details

For full workflow details: `embd prime`
Auto-inject into Claude Code: `embd hooks install`
Auto-inject into Codex: `embd hooks install --codex`
