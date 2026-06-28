# GPUSitter: On-Call RCA Agent

An AI agent that **automates the GPU-fleet on-call engineer**. When a node fails, it does the triage a human would — pulls telemetry, finds correlated failures, matches past incidents, decides a fix, pages a tech if needed, and logs the resolution. Grounded in real failures, not synthetic data.

## Why it's not "an LLM on a dashboard"

A threshold alert says *what* broke. The agent does the work a human on-call does *after* the alert:

- **Correlate** — pulls telemetry around the crash, finds the common thread across co-failed nodes (same window, same job type) → root, not symptom.
- **Remember** — retrieves the matching past incident + resolution. Gets smarter each run.
- **Reason in-domain** — reads symptoms through GPU-failure priors (Xid/ECC co-occurrence) to name a likely cause class.
- **Decide, grounded** — every claim cites a number a tool returned. Picks a disposition: escalate / page technician / restart.

Reactive only. No prediction.

## Architecture

```
incident fires → ADK agent (Gemini) → [ tools ] → disposition + logged SOP
                      ▲ priors (SKILL.md)
```

- **Brain** — Google ADK + Gemini, thin tool-calling loop. Embedded, controllable, streamed to the dashboard.
- **Tools** — `get_telemetry` · `find_correlated_failures` · `search_past_incidents` · `page_technician` · `record_resolution`.
- **Sensor sim** — FastAPI backend replays real telemetry and fires incidents, mocking the real DCGM / IPMI / Prometheus surface (real field names: `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_GPU_UTIL`).
- **Memory** — JSON SOP store of past incident → resolution.
- **Dashboard** — incident feed lights up → live agent-reasoning tab.

## Data

[AcmeTrace](https://github.com/InternLM/AcmeTrace) — real LLM-training cluster traces (Kalos, A100s). `trace_kalos.csv` gives `state` (NODE_FAIL/FAILED) + `fail_time` = the incident trigger; per-metric wide CSVs give power/temp/util at ~15s cadence. `XID_ERRORS.csv` carries **real per-GPU Xid cause codes** (e.g. 43 = channel exception / GPU reset) — the agent grounds cause in real fault codes, with GPU-failure priors (Xid/ECC co-occurrence) filling gaps.

## Stack

Python · Google ADK + Gemini 2.5 Flash · FastAPI (SSE) · pandas · pytest.

## Run

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=...
uvicorn src.sim:app --reload   # → http://localhost:8000
pytest -v
```

## Stretch

- **Managed Agents spin-off** — for autonomous actuation/coding (write remediation, draft ticket/PR) in a Google sandbox, where it actually fits. Same `SKILL.md` priors.
- MCP exposure of the tools · DSPy-tuned disposition · embedding-based incident memory.

## Build plan

`scraps/` — prior iteration (archived). Full task-by-task plan: see the implementation plan in `docs/`.
