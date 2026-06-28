# GPUSitter

**A self-improving reliability agent for data-center GPU fleets.**

Hackathon project — 2026 AI Engineer World's Fair. Targeting two tracks:
- **Continual Learning** — agents that improve from real-world use with no user intervention
- **The Self-Improvement Stack** — infrastructure to continuously evaluate, monitor, and upgrade AI systems

## Live demo

**[http://134.199.208.214:8000](http://134.199.208.214:8000)** — publicly accessible, running on DigitalOcean SFO3.

Click any incident in the feed → Gemini 2.5 Flash triages it live with real tool calls streaming in real time. Every completed triage writes to the SOP and retrains the predictor — the system gets smarter each run.

## The self-improvement loop

```
incident fires
  → agent calls get_telemetry + check_degradation_trend + find_correlated_failures
  → agent calls search_past_incidents (semantic similarity over all prior SOP entries)
      ↳ references similar past cases by name; notes if pattern was seen before
  → agent decides disposition + calls record_resolution with:
      - full summary + resolution text (embedded via gemini-embedding-001 for future search)
      - numeric telemetry metrics: power_spike_ratio, temp_rise_C, correlated_count
  → agent calls train_and_validate
      ↳ fits logreg on all SOP entries that have metrics
      ↳ promotes to v(N+1) if val ROC-AUC beats incumbent (or no incumbent yet)
      ↳ model card updates in the UI: version · model_type · AUC · n_samples
  → outcome auto-saved to memory (no user prompt)
      ↳ SOP entry re-embedded with outcome context for richer future search
```

**What improves with each incident:**
1. **Semantic recall** — the SOP grows; future agents find similar cases by meaning, not keyword
2. **Degradation fingerprints** — summaries explicitly record pre-failure power spike ratios and temp rises; future search surfaces these signals for earlier prediction
3. **Disposition classifier** — trains from incident 1; version increments whenever val AUC improves; AUC shown as `—` until enough varied cases accumulate for a holdout
4. **Outcome-enriched embeddings** — auto-confirmed outcomes re-embed the SOP entry so future semantic search finds confirmed hardware faults vs false alarms

## Architecture (`src/gpusitter/` package)

| Subpackage / module | Responsibility |
|--------|----------------|
| `telemetry/` + `rca/` | AcmeTrace incidents, streaming telemetry windows (no pandas), correlation |
| `memory.py` | SOP read/write + `gemini-embedding-001` semantic search; lazy vector index in `data/sop_vectors.json`; cosine similarity with 0.4 threshold |
| `dataset.py` | `build_xy_from_sop()` — extracts `[power_spike_ratio, temp_rise_C, correlated_count]` feature matrix from SOP entries for classifier training |
| `classifier.py` | `fit_candidate` / `maybe_promote` / `save_state` — in-memory incumbent + persisted model card at `data/model_state.json` |
| `tools.py` | `get_telemetry`, `check_degradation_trend`, `find_correlated_failures`, `search_past_incidents`, `page_technician`, `record_resolution` (stores metrics), `train_and_validate` |
| `priors.py` | GPU-failure domain priors injected into the agent system prompt |
| `agent.py` | Google ADK + Gemini 2.5 Flash: 7-step ReAct triage loop, yields SSE events |
| `sim.py` | FastAPI: `/api/incidents` (SSE stream), `/api/triage` (streaming agent), `/api/model` (model card), `/api/feedback` (outcome recording); serves compiled React dashboard |

## API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/incidents` | GET SSE | Live stream of incidents from AcmeTrace Kalos trace |
| `/api/triage` | POST SSE | Stream Gemini agent events for one incident |
| `/api/model` | GET | Current predictor: version, model_type, val_auc, n_samples |
| `/api/feedback` | POST | Record outcome `{incident_id, outcome}` → re-embeds SOP entry |

## Self-improvement observability

After each triage the UI shows:
- **Model card** (top of triage panel): `predictor · logreg · v2 · AUC 0.847 · 9 samples`
- **SOP written** badge when `record_resolution` fires
- **`✎ saved to memory`** in the disposition header when the run completes

## Quick start (local)

```bash
pip install -e .                        # or: uv sync
export GOOGLE_API_KEY=...
PYTHONPATH=src uvicorn gpusitter.app.sim:app --reload  # → http://localhost:8000
pytest -q                               # offline test suite (no API key needed)
```

No big data needed locally — the app serves fixture incidents from `src/gpusitter/app/dashboard/fixtures/`.

## Status

- ✅ Real Gemini tool calls — no mocks, all 7 tools fire in production
- ✅ Live SSE streaming — animated phase indicator, tool spinner, elapsed timer
- ✅ Semantic SOP memory — `gemini-embedding-001`, cosine similarity, lazy vector index, 0.8+ similarity on known patterns
- ✅ Pre-failure degradation detection — `check_degradation_trend` looks 4 h before failure; power spike ratio >1.5 or temp rise >10 °C flags `gradual_degradation_signal`
- ✅ Disposition classifier — trains from incident 1; promotes on AUC improvement; model card in UI
- ✅ Outcome feedback — auto-saves after every triage; re-embeds SOP entry with outcome context
- ✅ `/api/model` + `/api/feedback` REST endpoints
- 🚧 Xid-event-driven real-time incident ingestion (currently replays trace CSV)
- ⏭️ Stretch: Managed-Agents actuation, MCP tool exposure, model routing by incident type
## Data

The 80 GB AcmeTrace telemetry lives on the droplet. The app reads telemetry CSVs directly at query time (`data/acme-util/data/utilization/kalos/*.csv`). See **[docs/DATA.md](docs/DATA.md)** for schema details and the AcmeTrace reality check.

> ⚠️ **AcmeTrace reality check**: job failures and 15 s telemetry overlap only ~1.5 days; Kalos has no `NODE_FAIL` — incident = `FAILED` + non-null `fail_time`; timestamps are ISO UTC strings; `util_pkl/*.pkl` are CDF distributions not time series; real Xid codes in `XID_ERRORS.csv`.

## Deployment

> **No CI/CD is configured.** Deploy is manual — push to `main` then SSH in and pull.

**Server:** `134.199.208.214` (DigitalOcean SFO3, Ubuntu 24.04, 2 vCPU / 16 GB RAM)

```bash
# 1. SSH into the droplet
ssh root@134.199.208.214

# 2. Pull latest main
cd /root/hackathon-datacenter-agent
git pull origin main

# 3. (If Python deps changed) reinstall
pip install -e .

# 4. (If dashboard changed) rebuild the React app locally first, then copy built output:
#    Local: cd dashboard && npm run build
#    Local: scp -r dashboard/dist/* root@134.199.208.214:/root/hackathon-datacenter-agent/src/gpusitter/app/dashboard/

# 5. Restart the server
pkill -f uvicorn || true
nohup env PYTHONPATH=src GOOGLE_API_KEY="$(cat .env | grep GOOGLE_API_KEY | cut -d= -f2)" \
  uvicorn gpusitter.app.sim:app --host 0.0.0.0 --port 8000 \
  > /tmp/uvicorn.log 2>&1 &
disown
```

The `GOOGLE_API_KEY` is stored in `/root/hackathon-datacenter-agent/.env` on the droplet (not committed).  
Logs: `tail -f /tmp/uvicorn.log`

## Infrastructure

- **Droplet:** `134.199.208.214`, 2 vCPU / 16 GB RAM / 290 GB disk, SFO3, Ubuntu 24.04
- **Spaces:** dataset bucket `gpu-cluster-trace-datasets.sfo3.digitaloceanspaces.com`
- **Stack:** FastAPI + Google ADK + Gemini 2.5 Flash + React + Vite + scikit-learn
