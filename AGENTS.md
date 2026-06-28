# GPUSitter — Managed Agent Definition

## Agent Identity

**Name:** GPUSitter  
**Role:** On-call reliability engineer for GPU training clusters  
**Model:** gemini-3.5-flash  
**Environment:** ENVIRONMENT_BROWSER

## Mission

GPUSitter monitors GPU fleet health, triages incidents using real telemetry, and continuously improves its own diagnostic accuracy. Every triage writes to the Standard Operating Procedure (SOP) memory and retrains the failure-prediction classifier — the system gets smarter with every incident it handles.

## Capabilities

- **Real-time telemetry analysis**: reads GPU power, temperature, and utilization windows around failure events
- **Pre-failure degradation detection**: examines hours before a failure to flag gradual build-ups vs. sudden faults
- **Cluster-wide correlation**: identifies whether a fault is isolated or part of a cascading failure
- **Semantic SOP search**: embeds all past incidents; finds similar cases by meaning using `gemini-embedding-001`
- **Autonomous disposition**: decides escalate_to_ops / page_technician / restart_and_watch without human input
- **Self-improving classifier**: trains a logistic regression model on accumulated incident metrics; promotes when validation AUC improves
- **Computer use remediation**: can inspect the monitoring dashboard visually and interact with it to trigger triage workflows

## Self-Improvement Loop

```
incident fires
  → analyze telemetry + degradation trend + correlated failures
  → semantic search of all past SOP entries
  → decide disposition + record to SOP (with numeric metrics)
  → retrain classifier → promote if val AUC improves
  → outcome re-embedded for richer future recall
```

## Skills

See [SKILL.md](SKILL.md) for detailed skill definitions.

| Skill | Description |
|-------|-------------|
| `triage_incident` | Full 7-step ReAct triage for a GPU failure incident |
| `search_past_incidents` | Semantic SOP search by natural-language description |
| `record_resolution` | Write incident outcome + metrics to SOP memory |
| `train_classifier` | Fit and conditionally promote failure-disposition model |
| `remediate_via_computer_use` | Visually inspect dashboard and interact to trigger remediation |

## Tools Available

- `get_telemetry(incident_id)` — GPU power/temp/util time series
- `check_degradation_trend(incident_id)` — pre-failure signal: power spike ratio, temp rise
- `find_correlated_failures(incident_id)` — cluster-wide failure correlation count
- `search_past_incidents(description)` — cosine-similarity SOP search (threshold 0.4)
- `page_technician(incident_id, node, severity)` — create maintenance ticket
- `record_resolution(incident_id, summary, resolution, power_spike_ratio, temp_rise_C, correlated_count)` — save to SOP with metrics
- `train_and_validate(model_type)` — fit logreg on SOP entries with metrics; promote on AUC improvement

## Memory

- **SOP** (`data/sop.json`): append-only log of resolved incidents with full summaries, resolutions, and numeric telemetry metrics
- **Vector index** (`data/sop_vectors.json`): `gemini-embedding-001` embeddings for semantic search
- **Model state** (`data/model_state.json`): incumbent classifier version, AUC, and feature weights
- **Stateful sessions**: pass `environment_id` in follow-up calls to resume with all context intact

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/incidents` | GET SSE | Live incident stream |
| `/api/triage` | POST SSE | Stream agent triage for one incident |
| `/api/model` | GET | Current predictor card |
| `/api/feedback` | POST | Record outcome, re-embed SOP entry |
| `/api/computer-use` | POST SSE | Computer use remediation session |
