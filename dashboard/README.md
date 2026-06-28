# GPUSitter Dashboard

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

All panels read **AcmeTrace Kalos telemetry** (Shanghai AI Lab, Aug 2023) as
static fixtures in `public/fixtures/`, framed around the **Aug-17 06:00
correlated Xid burst** — the verified hero event: **116 GPUs across 74 nodes
fault within ~30 s**, Xid 43-dominant, scattered (≤4-of-8 per node), with no
thermal/power precursor. See `../docs/incident-aug17-0600.md` (bead `6xk`) for
the full characterization, plus `../docs/data-findings.md` and `../docs/DATA.md`.

> **Derived demo fixture.** These are *derived* fixtures: raw Kalos data is
> unchanged and historically correct. The earlier "Aug-29 13:57 / 882 GPU"
> framing was a **latched-state + window-edge artifact** (XID is a latched DCGM
> gauge; a window opening mid-fault misreads pre-latched GPUs as fresh onsets) —
> under empty-aware edge detection Aug-29 has **1** true onset, so it is not used
> here. The fixture's headline numbers (116/74/43-dominant) match the real
> Aug-17 06:00 burst; per-GPU cells are illustrative coordinates sampled from the
> fleet, not the literal onset roster (which lives only in the droplet CSVs).

## Backend

Live triage uses `EventSource('/api/triage')` — wired in `AgentTriage.tsx`.
The FastAPI backend (`src/gpusitter/app/sim.py`) serves the compiled dashboard
and all `/api/*` endpoints. See the root `README.md` for the full API table.

Build and deploy the dashboard:
```bash
npm run build       # outputs to dist/
# then copy dist/* → src/gpusitter/app/dashboard/ and deploy the server
```

## Fixtures (data contract)

| File | Shape |
|------|-------|
| `incidents.json` | `Incident[]` |
| `fleet.json` | `Fleet` (per-GPU heatmap snapshot) |
| `telemetry/<id>.json` | `TelemetryWindow` (power/temp/util series) |
| `meta.json` | `Meta` (window, cascade ts, totals) |
