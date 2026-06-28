# Self-Improving Failure Predictor — Design Spec

**Date:** 2026-06-27
**Builds on:** the reactive GPU on-call RCA agent (branch `build/rca-agent`: `backend/loader.py`, `memory.py`, `priors.py`, `tools.py`, `agent.py`, `sim.py`).

## Goal

Add a closed loop where the reactive RCA agent doesn't just narrate an incident — its investigation **drives a self-improving failure-prediction classifier**. On each incident, the agent picks a model form + feature set informed by what it found, we train + validate on the data streamed so far, and promote the candidate only if it beats the incumbent on held-out ROC-AUC.

## Locked Decisions

1. **Agent action space:** the agent chooses the **model form** (LogisticRegression / DecisionTree / GradientBoosting) **and the feature subset**. It does not write model code. We fit + validate the chosen spec.
2. **Prediction target:** per-job **binary** — "will this job end in `NODE_FAIL`/`FAILED`?" — from telemetry-aggregate + metadata features.
3. **Promotion metric:** **ROC-AUC** on a held-out validation split. Promote the candidate iff its val-AUC strictly exceeds the incumbent's. First model (no incumbent) always sets the baseline.
4. **Data model — streaming accretion (no 80GB at runtime):** the sim replays a small per-job table over time, warm-started at a point where ~100 incidents already passed. The app starts with **no classifier**. Training/validation use only **jobs streamed so far**, growing per incident.
5. **Record contents:** each streamed job record carries job-trace metadata + a few **precomputed telemetry aggregates** (power/temp/util mean·max·std). These serve as both classifier features and the data the agent's sensory tools read. The raw 80GB telemetry is never in the sim.

## Architecture

### The loop
```
warm-start history (first slice, includes ~100 incidents) → app init, NO classifier
  → SSE streams more job records over time, history grows in memory
  → incident fires (a NODE_FAIL/FAILED record)
  → agent investigates via sensory tools (telemetry aggregates + correlation over history-so-far)
  → agent chooses model_type + feature subset based on findings
  → train_and_validate: fit candidate on history-so-far (time-ordered train/val split), score val ROC-AUC
  → candidate AUC > incumbent AUC ? promote as live predictor : keep incumbent
  → dashboard updates model card (type/features/AUC/version); record_resolution logs the event
```

### Offline prep (run once, on the DO droplet, never at demo time)
- **`scripts/precompute_features.py`** — reads AcmeTrace (`trace_kalos.csv` + telemetry under `util_pkl/`/`ipmi/`), computes one row per job: metadata fields + per-job telemetry aggregates + label, writes a small **`data/jobs.csv`**. This is the only step that touches the 80GB. Output is small (jobs × ~20 columns).

### Components (units, each one responsibility)

| Module | Responsibility | Key interface |
|--------|----------------|---------------|
| `scripts/precompute_features.py` | 80GB → small `data/jobs.csv` (offline, once) | CLI script; no runtime import |
| `backend/stream.py` | replay `jobs.csv` over time; warm-start; accumulate `history`; expose incidents | `warm_start(path, n_incidents) -> list[dict]`; `stream_jobs(path, start_index)` generator; `HISTORY: list[dict]` |
| `backend/dataset.py` | history → feature matrix + labels + time-ordered split | `build_xy(history, features) -> (X, y)`; `time_split(X, y, val_frac=0.3) -> (Xtr,ytr,Xval,yval)` |
| `backend/classifier.py` | fit candidate, score AUC, hold + promote current model | `fit_candidate(model_type, features, Xtr, ytr) -> Model`; `auc(model, Xval, yval) -> float`; `Incumbent` state `{model, model_type, features, auc, version}`; `maybe_promote(candidate) -> bool` |
| `backend/tools.py` (extend) | agent-callable tools | add `get_sensory(job_id)` (per-job aggregates from history), `train_and_validate(model_type, features) -> dict` |
| `backend/agent.py` (extend) | investigate → choose spec → call `train_and_validate` → report | instruction extended; new tool registered |
| `backend/sim.py` (extend) | drive the stream + serve dashboard + `/triage` | uses `stream`/`history`; SSE emits incidents; model-state endpoint |
| `backend/dashboard/index.html` (extend) | live model card + promote events | shows type/features/AUC/version |

### Feature & label definition
- **Label:** `y = 1` if `state in {NODE_FAIL, FAILED}` else `0`.
- **Candidate features (agent selects a subset):** numeric metadata — `node_num, gpu_num, cpu_num, duration, queue, mem_per_pod_GB`; telemetry aggregates — `power_mean, power_max, power_std, temp_mean, temp_max, temp_std, util_mean, util_max, util_std`; categorical `type` (one-hot encoded in `dataset.py` when selected).
- A job's telemetry aggregates are the precomputed columns from `jobs.csv` — no time-series math at runtime.

### `train_and_validate(model_type, features)` — the core tool
1. `build_xy(HISTORY, features)` → X, y over all jobs streamed so far.
2. `time_split` → train (older) / val (newer), `val_frac=0.3`.
3. Guard: if train or val lacks both classes → return `{"trained": False, "reason": "insufficient class balance"}` (await more data).
4. `fit_candidate(model_type, features, Xtr, ytr)` → candidate; `auc(candidate, Xval, yval)` → val AUC.
5. `maybe_promote`: promote iff `Incumbent is None` or `candidate_auc > Incumbent.auc`. On promote, bump `version`.
6. Return `{"trained": True, "model_type", "features", "val_auc", "incumbent_auc", "promoted": bool, "version"}`.

### Promotion gate
- Time-ordered split (train = earlier stream order, val = later) — honest for a temporal stream, avoids leakage.
- Strict improvement (`>`) so noise doesn't churn the live model. First candidate always promoted (baseline).

## Data flow
`jobs.csv` (small, precomputed) → `warm_start` preloads `HISTORY[0:k]` (k = index past the 100th incident) → `stream_jobs` appends subsequent records to `HISTORY`, emitting NODE_FAIL rows on SSE → incident triggers agent → tools read `HISTORY` → `train_and_validate` fits/scores/promotes → `Incumbent` updated → dashboard + SOP reflect it.

## Error handling / edge cases
- **No data file:** `stream`/tools fail lazily (import never crashes), consistent with existing `sim.py` lazy-load pattern.
- **Single-class split:** `train_and_validate` returns `trained: False` rather than raising (ROC-AUC undefined on one class).
- **No incumbent yet:** first successful train sets baseline, `promoted: True`, `version: 1`.
- **Empty/0-row history at first incident:** warm-start guarantees ≥100 incidents of history, so positives exist before the first live incident.

## Testing (TDD)
- `dataset`: `build_xy` produces correct shape + label vector; `type` one-hot deterministic; `time_split` respects order + `val_frac`.
- `classifier`: `fit_candidate` returns a fitted model; `auc` computes a known value on a tiny set; `maybe_promote` — higher-AUC candidate promotes + bumps version, lower-AUC does not, first candidate sets baseline.
- `tools.train_and_validate`: on a tiny synthetic `HISTORY`, returns correct metrics + promotion decision; single-class history → `trained: False`.
- `stream`: `warm_start` returns the right prefix; `stream_jobs` appends + emits only failure rows.
- `agent`: construction registers the new `train_and_validate` (and `get_sensory`) tool; priors still in instruction.
- `sim`: offline TestClient — model-state endpoint reflects `Incumbent`; stream emits incidents; `/triage` with stubbed agent.
- All tests offline, no `GOOGLE_API_KEY`, pristine output.

## Dependencies
Add **scikit-learn** to `requirements.txt`. (pandas, fastapi, uvicorn, google-adk, google-genai, httpx, pytest already pinned.)

## Out of scope (this spec)
- Real frontend (full dashboard-with-agents) — separate, not yet scoped.
- Managed-Agents actuation spin-off, MCP exposure, DSPy tuning, embedding memory — plan stretch items.
- Failure *prediction at inference time* as a product surface (alerts) — here the predictor is built/improved; serving live predictions to the dashboard is a thin follow-on, not required for the loop.

## Changes vs the existing reactive branch
- `sim.py` shifts from replaying `trace_kalos.csv` directly to driving `stream.py` over `jobs.csv` (with warm-start + accumulation).
- `tools.py` sensory tools read per-job aggregates from `HISTORY` instead of reading telemetry CSV windows; new model tools added.
- Agent instruction extended to drive the model-improvement step.
- Reactive tools (`find_correlated_failures`, `page_technician`, `record_resolution`) retained as evidence + disposition tail.
