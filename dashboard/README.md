# GPUSitter Dashboard (draft)

Mission-control UI for the GPU on-call RCA agent. Vite + React + TypeScript,
"ops terminal" aesthetic.

## Run

```bash
cd dashboard
npm install
npm run dev        # http://localhost:5173
```

## What you're looking at

Three-pane mission control over a top KPI bar:

- **Incident feed** (left) — Xid-driven incidents. Click one to triage it.
- **Fleet heatmap** (center) — every Kalos GPU (2,344) as a cell, grouped by node;
  red = active Xid fault, green→amber by temp, faint = idle. Below it, the
  selected GPU's `power / temp / util` telemetry (±3 min around the fault).
- **Agent triage** (right) — the agent's live ReAct reasoning (tool calls →
  grounded observations → disposition).

## Data

All panels read **real AcmeTrace Kalos telemetry** (Shanghai AI Lab, Aug 2023)
as static fixtures in `public/fixtures/`, generated around the **Aug-29 13:57
cluster-wide Xid cascade** (882 GPUs across 141 nodes — the team's hero event).
See `../docs/data-findings.md` and `../docs/DATA.md`.

### Wiring to the real backend later

`src/data.ts` reads `/fixtures/*.json`. When the FastAPI sim backend exists,
point those fetches at `/api/*` (already proxied in `vite.config.ts`) and swap
`src/lib/replay.ts` for a real `EventSource('/api/triage')` — the TypeScript
shapes in `src/types.ts` are the contract and don't change.

## Fixtures (data contract)

| File | Shape |
|------|-------|
| `incidents.json` | `Incident[]` |
| `fleet.json` | `Fleet` (per-GPU heatmap snapshot) |
| `telemetry/<id>.json` | `TelemetryWindow` (power/temp/util series) |
| `agentRuns.json` | `Record<incidentId, AgentEvent[]>` |
| `meta.json` | `Meta` (window, cascade ts, totals) |
