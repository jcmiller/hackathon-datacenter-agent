# GPUSitter Skills

Custom skills available to the GPUSitter agent running in Antigravity or any Managed Agent environment.

---

## Skill: triage_incident

**Trigger:** A GPU incident fires — Xid error, job failure, or anomaly detected in telemetry.

**Description:** Performs a full 7-step ReAct triage loop for a single GPU failure incident. Grounds every decision in tool return values. Produces a disposition (escalate_to_ops / page_technician / restart_and_watch) and writes the outcome to persistent SOP memory.

**Steps:**
1. `get_telemetry(incident_id)` — read power/temp/util around failure time
2. `check_degradation_trend(incident_id)` — examine 4 h before failure for gradual build-up
3. `find_correlated_failures(incident_id)` — check cluster-wide spread
4. `search_past_incidents(rich_description)` — semantic SOP search combining Xid code, telemetry values, degradation pattern, and correlated count
5. Decide disposition; call `page_technician` if hardware replacement is needed
6. `record_resolution(incident_id, summary, resolution, power_spike_ratio, temp_rise_C, correlated_count)` — save to SOP with exact numeric metrics
7. `train_and_validate(model_type="logreg")` — retrain failure classifier; report val_auc and promotion status

**Output:** Disposition label + summary + ticket number (if paged) + model version after training.

---

## Skill: search_past_incidents

**Trigger:** User asks "have we seen this before?" or agent needs to check SOP before deciding on a novel failure mode.

**Description:** Semantic search of the full SOP using natural language. The query should be rich — combine Xid error code, telemetry values, degradation pattern, and correlated count. Uses cosine similarity over `gemini-embedding-001` embeddings with a 0.4 threshold.

**Input:** Free-text description of the current incident.  
**Output:** List of similar past incidents with summaries, resolutions, and similarity scores.

---

## Skill: record_resolution

**Trigger:** After a triage is complete and a disposition has been decided.

**Description:** Writes the incident outcome to the Standard Operating Procedure memory. Stores full summary, resolution text, and numeric telemetry metrics used to train the classifier.

**Input:**
- `incident_id` — unique incident identifier
- `summary` — what happened and what signals were present
- `resolution` — what action was taken and why
- `power_spike_ratio` — peak power / baseline power in the 4 h window
- `temp_rise_C` — temperature rise in °C over the 4 h window
- `correlated_count` — number of other GPUs that failed in the same window

**Side effects:** Embeds the new SOP entry for future semantic search.

---

## Skill: train_classifier

**Trigger:** After `record_resolution` — automatically called at the end of every triage.

**Description:** Fits a logistic regression model on all SOP entries that have numeric metrics. Promotes the new model to incumbent if validation AUC beats the current incumbent (or if no incumbent exists yet).

**Input:** `model_type` (default: `"logreg"`)  
**Output:** `val_auc`, `n_samples`, whether a new version was promoted.

---

## Skill: remediate_via_computer_use

**Trigger:** User requests automated remediation, or the agent decides to visually inspect the monitoring dashboard.

**Description:** Launches a computer use session with Gemini 3.5 Flash. Takes a screenshot of the live GPUSitter dashboard, sends it to the model with the `ComputerUse(environment=ENVIRONMENT_BROWSER)` tool enabled. The model analyzes the screen state and generates UI actions (clicks, keystrokes) to navigate to a critical incident and trigger the triage workflow — demonstrating fully autonomous, vision-guided remediation.

**Steps:**
1. Capture screenshot of dashboard at `http://localhost:8000`
2. Send to `gemini-3.5-flash` with `ComputerUse(environment=ENVIRONMENT_BROWSER)` tool
3. Execute returned UI actions (click incident, trigger triage, observe result)
4. Capture updated screenshot after each action
5. Repeat for up to 5 turns or until triage is complete

**Output:** Streamed SSE events — `screenshot` (base64 PNG), `action` (type + coordinates/text), `reasoning` (model text).

---

## Environment Notes

- All tools are available at `http://localhost:8000/api/*` when running locally
- SOP memory persists in `data/sop.json` across sessions
- The classifier model state persists in `data/model_state.json`
- Pass `environment_id` in follow-up Antigravity calls to resume a stateful session
