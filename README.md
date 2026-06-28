# GPUSitter

**A self-improving reliability agent for data-center GPU fleets.**

Hackathon project — 2026 AI Engineer World's Fair. When a GPU node fails, an agent automates the on-call engineer: it investigates the incident with real sensor tool-calls, then **uses what it found to improve a failure-prediction model** — fitting a candidate, validating it, and promoting it only if it beats the incumbent. Grounded in real AcmeTrace cluster data, not pure simulation.

## The loop

```
job records stream in (warm-started past ~100 past incidents) → app starts with NO model
  → incident fires (a FAILED job)
  → agent investigates via sensory tool-calls (telemetry aggregates, correlated failures, past incidents)
  → agent picks a model form (logreg/tree/gboost) + feature set based on what it found
  → train + validate on jobs-streamed-so-far (time-ordered split)
  → candidate ROC-AUC > incumbent? promote as the live predictor
  → dashboard updates the model card; resolution logged to memory (SOP)
```

Two things make this more than "an LLM on a dashboard":
- **The agent's investigation drives real ML** — model selection + feature engineering gated by held-out ROC-AUC, not vibes.
- **Every claim is grounded** in a number a tool returned. The agent never asserts a fault cause it can't read (real Xid codes when available; inference from priors otherwise).

## Live demo

**http://134.199.208.214:8000** — publicly accessible, running on DigitalOcean (SFO3).

Click any incident in the feed → the agent triages it live with real Gemini tool calls streaming
in real time. Each run writes a new entry to the SOP; subsequent similar incidents surface past
cases with semantic similarity scores.

## Architecture (`backend/` package)

| Module | Responsibility |
|--------|----------------|
| `loader.py` | AcmeTrace incidents + telemetry windows + correlation |
| `memory.py` | SOP read/write + `gemini-embedding-001` semantic search (cosine similarity over `data/sop_vectors.json`) |
| `tools.py` | `get_telemetry`, `check_degradation_trend`, `find_correlated_failures`, `search_past_incidents`, `page_technician`, `record_resolution` |
| `priors.py` | GPU-failure domain priors injected into the agent system prompt |
| `agent.py` | Google ADK + Gemini 2.5 Flash: 6-step ReAct triage loop, yields SSE events |
| `sim.py` | FastAPI: SSE incident stream (`/api/incidents`), streaming triage (`/api/triage`), serves compiled React dashboard |

**Recursive self-improvement:** each `record_resolution` call embeds the new SOP entry with `gemini-embedding-001`; `search_past_incidents` does cosine similarity over all prior entries so the agent recognizes failure patterns it has seen before and notes pre-failure degradation signals for earlier prediction next time.

**Harness:** Google ADK + Gemini 2.5 Flash — a thin, embedded tool-calling loop we control (not the hosted Managed-Agents sandbox).

## Quick start (local, no big data needed)

```bash
pip install -r requirements.txt
python scripts/make_mock_jobs.py        # writes a small data/jobs.csv (synthetic, schema-correct)
export GOOGLE_API_KEY=...               # needed only for the live agent /triage
uvicorn backend.sim:app --reload        # → http://localhost:8000
pytest -q                               # offline test suite
```

The mock lets you run the full stream → incident → train/validate → promote loop locally without the 80 GB dataset.

## Data

The 80 GB AcmeTrace telemetry **never runs in the app**. A one-time offline step
(`scripts/precompute_features.py`, run on the box where the data lives) turns it into a
small `data/jobs.csv` — one row per job: metadata + telemetry aggregates + label. The sim
replays that small file. See **[docs/DATA.md](docs/DATA.md)** and **[docs/data-findings.md](docs/data-findings.md)**.

```bash
scripts/download_datasets.sh        # PAI 2020 + Acme (default)
scripts/download_datasets.sh all    # + Philly (best-effort) + extra Alibaba traces
scripts/download_datasets.sh --list # show targets
```

> ⚠️ **AcmeTrace reality check** (verified, read before wiring real data): job failures and
> the 15 s telemetry overlap only ~1.5 days (~113 of 13,836 FAILED jobs have telemetry);
> Kalos has **no `NODE_FAIL`** — incident = `FAILED` + non-null `fail_time`; timestamps are
> **ISO UTC strings**, not epoch; `util_pkl/*.pkl` are CDF distributions, **not** time series
> (real telemetry is `acme-util/.../kalos/*.csv`); and **real Xid codes exist** in
> `XID_ERRORS.csv`. Full detail at the top of [docs/DATA.md](docs/DATA.md).

## DigitalOcean (data + compute)

- **Spaces:** dataset bucket `https://gpu-cluster-trace-datasets.sfo3.digitaloceanspaces.com/`.
- **Droplet:** runs services + telemetry processing — `134.199.208.214`, 4 GB / 2 vCPU / 120 GB, SFO3, Ubuntu 24.04. A 16 GB swap file was added to survive Git-LFS OOM when fetching `acme-util`; the ~80 GB lives in the LFS cache and is read in place via `scripts/lfs_helper.py` (no checkout). See [docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md).

## Status

- ✅ Reactive RCA agent — real Gemini tool calls, live SSE streaming, animated dashboard
- ✅ Semantic memory — `gemini-embedding-001` embeddings, cosine similarity, lazy vector index
- ✅ Pre-failure degradation detection — power spike ratio + temp rise over 4-hour lookback
- ✅ Self-improving SOP — every resolution embeds and becomes searchable for future incidents
- 🚧 ML predictor loop (`dataset`, `classifier`, `train_and_validate`) — in progress
- ⏭️ Stretch: Xid-event-driven incidents, Managed-Agents actuation, MCP tool exposure

`scraps/` holds the earlier iteration. `docs/superpowers/` (local only) holds the design spec + implementation plan.
