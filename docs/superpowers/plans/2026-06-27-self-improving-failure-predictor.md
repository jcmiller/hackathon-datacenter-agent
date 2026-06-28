# Self-Improving Failure Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On each streamed incident, the RCA agent picks a model form + feature set, we train + validate a per-job NODE_FAIL classifier on the data streamed so far, and promote it only if it beats the incumbent on held-out ROC-AUC.

**Architecture:** A small precomputed per-job table (`data/jobs.csv`) is replayed over time, warm-started past ~100 incidents. The app starts with no classifier; an accumulating in-memory `HISTORY` grows as jobs stream. When an incident fires, the agent calls a `train_and_validate` tool that fits the chosen model on `HISTORY`, scores a time-ordered validation split, and promotes on strict AUC improvement.

**Tech Stack:** Python 3.11, pandas, scikit-learn, FastAPI (SSE), Google ADK + Gemini 2.5 Flash, pytest. Builds on branch `build/rca-agent`.

## Global Constraints

- **Language:** Python only. Senior-dev-simple — minimum code, no speculative abstraction, no config flags nobody asked for.
- **Package layout:** all app code under `backend/`; imports are `from backend.X import ...`; tests run via `pytest` from repo root; venv at `.venv` (run `.venv/bin/python -m pytest`).
- **Streaming-only data:** training/validation use only `HISTORY` (jobs streamed so far). Never load the 80GB telemetry at runtime; `data/jobs.csv` is small and precomputed offline.
- **Prediction target:** per-job binary — `y = 1` if `state in {NODE_FAIL, FAILED}` else `0`.
- **Promotion metric:** held-out **ROC-AUC**, time-ordered split (train = earlier, val = later), `val_frac = 0.3`. Promote iff candidate val-AUC strictly exceeds incumbent; first model always sets the baseline (version 1).
- **Model forms (exact keys):** `"logreg"` (LogisticRegression), `"tree"` (DecisionTreeClassifier), `"gboost"` (GradientBoostingClassifier).
- **Candidate feature names:** numeric — `node_num, gpu_num, cpu_num, duration, queue, mem_per_pod_GB, power_mean, power_max, power_std, temp_mean, temp_max, temp_std, util_mean, util_max, util_std`; categorical — `type` (one-hot encoded when selected).
- **TDD:** failing test first; all tests offline (no `GOOGLE_API_KEY`); pristine output (repo `pytest.ini` already filters the ADK + starlette deprecation warnings).

---

## File Structure

```
scripts/
  precompute_features.py    # offline, run once on DO: AcmeTrace -> data/jobs.csv
backend/
  dataset.py                # history -> feature matrix + labels + time-ordered split
  classifier.py             # fit candidate, score AUC, hold + promote current model
  stream.py                 # replay jobs.csv over time; warm-start; accumulate HISTORY
  tools.py                  # (extend) get_sensory + train_and_validate
  agent.py                  # (extend) register new tools, extend instruction (+ Xid-honesty)
  sim.py                    # (extend) drive stream, /model endpoint
  dashboard/index.html      # (extend) live model card
tests/
  test_dataset.py
  test_classifier.py
  test_stream.py
  test_tools_ml.py
  test_agent.py             # (extend existing)
  test_sim.py               # (extend existing)
```

---

### Task 1: dataset.py — features + time-ordered split

**Files:**
- Create: `backend/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces:
  - `build_xy(history: list[dict], features: list[str]) -> (pandas.DataFrame, list[int])` — X over the selected features (one-hot `type` when present), y = 1 for `state in {NODE_FAIL, FAILED}` else 0.
  - `time_split(X: pandas.DataFrame, y: list[int], val_frac: float = 0.3) -> (Xtr, ytr, Xval, yval)` — first `1-val_frac` rows train, last `val_frac` val (stream order preserved).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import pandas as pd
from backend.dataset import build_xy, time_split

def test_build_xy_shapes_and_labels():
    hist = [
        {"gpu_num": 8, "duration": 100, "type": "train", "state": "COMPLETED"},
        {"gpu_num": 16, "duration": 200, "type": "eval", "state": "NODE_FAIL"},
    ]
    X, y = build_xy(hist, ["gpu_num", "duration", "type"])
    assert y == [0, 1]
    assert list(X["gpu_num"]) == [8, 16]
    assert "type_train" in X.columns and "type_eval" in X.columns

def test_time_split_respects_order():
    X = pd.DataFrame({"a": list(range(1, 11))})
    y = [0, 1] * 5
    Xtr, ytr, Xval, yval = time_split(X, y, val_frac=0.3)
    assert len(ytr) == 7 and len(yval) == 3
    assert list(Xval["a"]) == [8, 9, 10]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.dataset'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/dataset.py
import pandas as pd

FAIL_STATES = {"NODE_FAIL", "FAILED"}

def build_xy(history, features):
    df = pd.DataFrame(history)
    y = df["state"].isin(FAIL_STATES).astype(int).tolist()
    cols = [f for f in features if f != "type"]
    X = df[cols].copy() if cols else pd.DataFrame(index=df.index)
    if "type" in features:
        X = pd.concat([X, pd.get_dummies(df["type"], prefix="type")], axis=1)
    return X, y

def time_split(X, y, val_frac=0.3):
    k = int(len(y) * (1 - val_frac))
    return X.iloc[:k], y[:k], X.iloc[k:], y[k:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/dataset.py tests/test_dataset.py
git commit -m "feat: dataset — featurize history + time-ordered split"
```

---

### Task 2: classifier.py — fit, score, promote (+ scikit-learn dep)

**Files:**
- Create: `backend/classifier.py`
- Modify: `requirements.txt` (add pinned `scikit-learn`)
- Test: `tests/test_classifier.py`

**Interfaces:**
- Consumes: scikit-learn estimators.
- Produces:
  - `Model` dataclass: `estimator, model_type: str, features: list, auc: float, version: int`.
  - `INCUMBENT: Model | None` — module-level current model.
  - `fit_candidate(model_type: str, features: list, Xtr, ytr) -> estimator`.
  - `auc(estimator, Xval, yval) -> float` — ROC-AUC from `predict_proba[:,1]`.
  - `maybe_promote(estimator, model_type: str, features: list, val_auc: float) -> bool` — promotes (updates `INCUMBENT`, bumps version) iff `INCUMBENT is None` or `val_auc > INCUMBENT.auc`.
  - `reset() -> None` — clears `INCUMBENT` (test helper).

- [ ] **Step 1: Install scikit-learn and pin it**

Run: `.venv/bin/pip install scikit-learn` (or `uv pip install --python .venv/bin/python scikit-learn`). Then add the installed version to `requirements.txt`:

```bash
.venv/bin/python -m pip show scikit-learn | grep -i version
# append e.g. "scikit-learn==1.5.2" to requirements.txt (use the actual version printed)
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_classifier.py
import pandas as pd
import backend.classifier as clf

def test_fit_and_auc_perfect_separation():
    clf.reset()
    Xtr = pd.DataFrame({"x": [0, 0, 1, 1]}); ytr = [0, 0, 1, 1]
    Xval = pd.DataFrame({"x": [0, 1]}); yval = [0, 1]
    est = clf.fit_candidate("logreg", ["x"], Xtr, ytr)
    assert clf.auc(est, Xval, yval) == 1.0

def test_promote_gate():
    clf.reset()
    assert clf.maybe_promote(None, "logreg", ["x"], 0.80) is True   # first -> baseline
    assert clf.INCUMBENT.version == 1 and clf.INCUMBENT.auc == 0.80
    assert clf.maybe_promote(None, "tree", ["x"], 0.75) is False    # worse, kept
    assert clf.INCUMBENT.version == 1
    assert clf.maybe_promote(None, "gboost", ["x"], 0.90) is True   # better -> v2
    assert clf.INCUMBENT.version == 2 and clf.INCUMBENT.model_type == "gboost"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.classifier'`.

- [ ] **Step 4: Write minimal implementation**

```python
# backend/classifier.py
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

_MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "tree": lambda: DecisionTreeClassifier(random_state=0),
    "gboost": lambda: GradientBoostingClassifier(random_state=0),
}

@dataclass
class Model:
    estimator: object
    model_type: str
    features: list
    auc: float
    version: int

INCUMBENT = None

def reset():
    global INCUMBENT
    INCUMBENT = None

def fit_candidate(model_type, features, Xtr, ytr):
    est = _MODELS[model_type]()
    est.fit(Xtr, ytr)
    return est

def auc(estimator, Xval, yval):
    proba = estimator.predict_proba(Xval)[:, 1]
    return float(roc_auc_score(yval, proba))

def maybe_promote(estimator, model_type, features, val_auc):
    global INCUMBENT
    if INCUMBENT is None or val_auc > INCUMBENT.auc:
        version = 1 if INCUMBENT is None else INCUMBENT.version + 1
        INCUMBENT = Model(estimator, model_type, features, val_auc, version)
        return True
    return False
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_classifier.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/classifier.py tests/test_classifier.py requirements.txt
git commit -m "feat: classifier — fit/score/promote on ROC-AUC; pin scikit-learn"
```

---

### Task 3: stream.py — replay, warm-start, accumulate HISTORY

**Files:**
- Create: `backend/stream.py`
- Test: `tests/test_stream.py`

**Interfaces:**
- Produces:
  - `HISTORY: list[dict]` — accumulated job records seen so far (the runtime accretion).
  - `reset_history() -> None`.
  - `warm_start(path: str, n_incidents: int) -> list[dict]` — the record prefix up to and including the `n_incidents`-th failure (full list if fewer failures exist).
  - `stream_jobs(path: str, start_index: int)` — generator yielding records from `start_index` onward.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stream.py
import pandas as pd
from backend.stream import warm_start, stream_jobs

def _write(tmp_path):
    rows = [{"job_id": i, "state": "COMPLETED"} for i in range(5)]
    rows[2]["state"] = "NODE_FAIL"
    rows[4]["state"] = "FAILED"
    p = tmp_path / "jobs.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p), rows

def test_warm_start_includes_nth_failure(tmp_path):
    p, rows = _write(tmp_path)
    pre = warm_start(p, 1)
    assert len(pre) == 3 and pre[-1]["state"] == "NODE_FAIL"  # first failure at index 2

def test_stream_jobs_from_index(tmp_path):
    p, rows = _write(tmp_path)
    out = list(stream_jobs(p, 3))
    assert [r["job_id"] for r in out] == [3, 4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_stream.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.stream'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/stream.py
import pandas as pd

FAIL_STATES = {"NODE_FAIL", "FAILED"}
HISTORY = []

def reset_history():
    HISTORY.clear()

def _load(path):
    return pd.read_csv(path).to_dict("records")

def warm_start(path, n_incidents):
    records = _load(path)
    count = 0
    for i, r in enumerate(records):
        if r["state"] in FAIL_STATES:
            count += 1
            if count == n_incidents:
                return records[: i + 1]
    return records

def stream_jobs(path, start_index):
    for r in _load(path)[start_index:]:
        yield r
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_stream.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/stream.py tests/test_stream.py
git commit -m "feat: stream — replay jobs.csv, warm-start, HISTORY accretion"
```

---

### Task 4: tools.py — get_sensory + train_and_validate

**Files:**
- Modify: `backend/tools.py` (append two functions + imports)
- Test: `tests/test_tools_ml.py`

**Interfaces:**
- Consumes: `backend.stream` (`HISTORY`), `backend.dataset` (`build_xy`, `time_split`), `backend.classifier` (`fit_candidate`, `auc`, `maybe_promote`, `INCUMBENT`).
- Produces:
  - `get_sensory(job_id) -> dict` — telemetry-aggregate fields (`power_*`, `temp_*`, `util_*`) for that job from `stream.HISTORY`; `{}` if not found.
  - `train_and_validate(model_type: str, features: list) -> dict` — builds X,y from `stream.HISTORY`, time-splits, guards single-class, fits + scores + maybe-promotes. Returns `{"trained": bool, ...}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools_ml.py
import backend.stream as stream
import backend.classifier as classifier
import backend.tools as tools

def _seed_history():
    stream.reset_history(); classifier.reset()
    # 10 jobs, separable by power_max; later rows are the val split
    for i in range(10):
        fail = i % 2 == 1
        stream.HISTORY.append({
            "job_id": i, "type": "train", "gpu_num": 8,
            "power_max": 300 if fail else 100,
            "state": "NODE_FAIL" if fail else "COMPLETED",
        })

def test_get_sensory_returns_aggregates():
    _seed_history()
    s = tools.get_sensory(1)
    assert s == {"power_max": 300}

def test_train_and_validate_trains_and_promotes():
    _seed_history()
    out = tools.train_and_validate("logreg", ["power_max", "gpu_num"])
    assert out["trained"] is True
    assert out["promoted"] is True and out["version"] == 1
    assert out["val_auc"] == 1.0

def test_train_and_validate_guards_single_class():
    stream.reset_history(); classifier.reset()
    for i in range(6):
        stream.HISTORY.append({"job_id": i, "gpu_num": 8, "power_max": 100, "state": "COMPLETED"})
    out = tools.train_and_validate("logreg", ["power_max", "gpu_num"])
    assert out["trained"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tools_ml.py -v`
Expected: FAIL with `AttributeError: module 'backend.tools' has no attribute 'get_sensory'`.

- [ ] **Step 3: Write minimal implementation** (append to `backend/tools.py`)

```python
# --- appended to backend/tools.py ---
import backend.stream as stream
import backend.dataset as dataset
import backend.classifier as classifier

def get_sensory(job_id):
    """Return telemetry-aggregate fields for a job from accumulated history."""
    for r in stream.HISTORY:
        if str(r.get("job_id")) == str(job_id):
            return {k: r[k] for k in r if k.startswith(("power_", "temp_", "util_"))}
    return {}

def train_and_validate(model_type, features):
    """Fit a candidate on jobs-so-far, score val ROC-AUC, promote if it beats the incumbent."""
    X, y = dataset.build_xy(stream.HISTORY, features)
    Xtr, ytr, Xval, yval = dataset.time_split(X, y, val_frac=0.3)
    if len(set(ytr)) < 2 or len(set(yval)) < 2:
        return {"trained": False, "reason": "insufficient class balance"}
    est = classifier.fit_candidate(model_type, features, Xtr, ytr)
    val_auc = classifier.auc(est, Xval, yval)
    incumbent_auc = classifier.INCUMBENT.auc if classifier.INCUMBENT else None
    promoted = classifier.maybe_promote(est, model_type, features, val_auc)
    return {"trained": True, "model_type": model_type, "features": features,
            "val_auc": val_auc, "incumbent_auc": incumbent_auc,
            "promoted": promoted, "version": classifier.INCUMBENT.version}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tools_ml.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/tools.py tests/test_tools_ml.py
git commit -m "feat: tools — get_sensory + train_and_validate over HISTORY"
```

---

### Task 5: agent.py — register tools, extend instruction (+ Xid-honesty)

**Files:**
- Modify: `backend/agent.py`
- Test: `tests/test_agent.py` (extend)

**Interfaces:**
- Consumes: `backend.tools` (`get_sensory`, `train_and_validate`, plus existing five).
- Produces: `build_agent()` registers all seven tools; instruction drives the model-improvement step and contains the Xid-honesty sentence and the `DOMAIN_PRIORS`.

- [ ] **Step 1: Write the failing test** (extend `tests/test_agent.py`)

```python
def test_agent_has_ml_tools_and_xid_honesty():
    a = build_agent()
    names = {t.__name__ for t in a.tools}
    assert {"get_sensory", "train_and_validate"} <= names
    assert "never assert a specific Xid" in a.instruction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_agent.py::test_agent_has_ml_tools_and_xid_honesty -v`
Expected: FAIL (tools not registered / sentence absent).

- [ ] **Step 3: Update implementation** — extend `INSTRUCTION` and the tool list in `backend/agent.py`.

Add to the numbered instruction body (after the grounding rule) these lines:

```
- Infer the likely fault class from priors + telemetry pattern — never assert a specific Xid
  code as an observed fact (no Xid data exists in the telemetry).
- Then improve the failure predictor: based on what you found, choose a model form
  ("logreg" | "tree" | "gboost") and a subset of features
  (node_num, gpu_num, cpu_num, duration, queue, mem_per_pod_GB, power_mean/max/std,
  temp_mean/max/std, util_mean/max/std, type), and call train_and_validate(model_type, features).
  Report the val_auc and whether it was promoted.
```

Extend the `tools=[...]` list passed to `Agent(...)` to include `tools.get_sensory` and `tools.train_and_validate` alongside the existing five.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agent.py -v`
Expected: all agent tests pass (existing + new).

- [ ] **Step 5: Commit**

```bash
git add backend/agent.py tests/test_agent.py
git commit -m "feat: agent — register ML tools, drive model improvement, Xid-honesty rule"
```

---

### Task 6: sim.py — drive stream, /model endpoint, dashboard card

**Files:**
- Modify: `backend/sim.py`, `backend/dashboard/index.html`
- Test: `tests/test_sim.py` (extend)

**Interfaces:**
- Consumes: `backend.stream` (`warm_start`, `stream_jobs`, `HISTORY`, `reset_history`), `backend.classifier` (`INCUMBENT`), `backend.agent` (`triage`).
- Produces:
  - Module constants `JOBS_CSV = "data/jobs.csv"`, `WARM_START_INCIDENTS = 100`, `STEP_SECONDS = 3`.
  - On first `/incidents` connection: warm-start `stream.HISTORY`, then stream subsequent jobs (append each to `HISTORY`), emit `data:` SSE only for failure rows.
  - `GET /model` → `{"version", "model_type", "features", "auc"}` from `classifier.INCUMBENT`, or `{"version": 0}` when none.

- [ ] **Step 1: Write the failing test** (extend `tests/test_sim.py`)

```python
def test_model_endpoint_reflects_incumbent(monkeypatch):
    import backend.classifier as clf
    from fastapi.testclient import TestClient
    import backend.sim as sim
    clf.reset()
    clf.maybe_promote(None, "gboost", ["power_max"], 0.91)
    r = TestClient(sim.app).get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1 and body["model_type"] == "gboost" and body["auc"] == 0.91

def test_model_endpoint_empty_when_no_model():
    import backend.classifier as clf
    from fastapi.testclient import TestClient
    import backend.sim as sim
    clf.reset()
    assert TestClient(sim.app).get("/model").json()["version"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sim.py::test_model_endpoint_reflects_incumbent -v`
Expected: FAIL (no `/model` route).

- [ ] **Step 3: Update implementation.**

Replace the trace-replay wiring in `backend/sim.py` so the SSE stream is warm-started and accumulates history, and add the `/model` route. Reference shape:

```python
# backend/sim.py (relevant parts)
import asyncio, json
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
import backend.stream as stream
import backend.classifier as classifier
from backend.agent import triage

app = FastAPI()
JOBS_CSV = "data/jobs.csv"
WARM_START_INCIDENTS = 100
STEP_SECONDS = 3
_started = {"done": False}

@app.get("/")
def index():
    return FileResponse("backend/dashboard/index.html")

@app.get("/model")
def model():
    inc = classifier.INCUMBENT
    if inc is None:
        return {"version": 0}
    return {"version": inc.version, "model_type": inc.model_type,
            "features": inc.features, "auc": inc.auc}

@app.get("/incidents")
async def incidents(request: Request):
    async def gen():
        if not _started["done"]:
            stream.reset_history()
            stream.HISTORY.extend(stream.warm_start(JOBS_CSV, WARM_START_INCIDENTS))
            _started["done"] = True
        start = len(stream.HISTORY)
        for r in stream.stream_jobs(JOBS_CSV, start):
            if await request.is_disconnected():
                break
            stream.HISTORY.append(r)
            if r["state"] in stream.FAIL_STATES:
                yield f"data: {json.dumps(r)}\n\n"
            await asyncio.sleep(STEP_SECONDS)
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/triage")
async def do_triage(incident: dict):
    return {"disposition": triage(incident)}
```

Keep the existing lazy-import discipline (no module-level data load). Update `backend/dashboard/index.html` to add a model card that fetches `/model` (poll every few seconds) and shows `version / model_type / features / auc`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sim.py -v`
Expected: all sim tests pass (existing + new two).

- [ ] **Step 5: Commit**

```bash
git add backend/sim.py backend/dashboard/index.html tests/test_sim.py
git commit -m "feat: sim — warm-started accreting stream + /model endpoint + dashboard card"
```

---

### Task 7: precompute_features.py + README setup + remove empty src/

**Files:**
- Create: `scripts/precompute_features.py`
- Modify: `README.md` (data setup), remove leftover empty `src/`
- Test: `tests/test_precompute.py`

**Interfaces:**
- Produces: `precompute_jobs(trace_csv: str, power_csv: str | None, temp_csv: str | None, util_csv: str | None, out_csv: str) -> None` — writes `data/jobs.csv` with metadata + label + telemetry-aggregate columns. Telemetry aggregates come from per-signal CSVs (Time + per-node columns) joined by the job's `[start_time, end_time]` window when provided, else `0.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_precompute.py
import pandas as pd
from scripts.precompute_features import precompute_jobs

def test_precompute_schema_and_label(tmp_path):
    trace = tmp_path / "trace.csv"
    pd.DataFrame([
        {"job_id": "j1", "type": "train", "node_num": 1, "gpu_num": 8, "cpu_num": 16,
         "duration": 90, "queue": 5, "mem_per_pod_GB": 80, "state": "NODE_FAIL",
         "start_time": 100, "end_time": 200, "fail_time": 190},
        {"job_id": "j2", "type": "eval", "node_num": 1, "gpu_num": 8, "cpu_num": 16,
         "duration": 90, "queue": 5, "mem_per_pod_GB": 80, "state": "COMPLETED",
         "start_time": 100, "end_time": 200, "fail_time": None},
    ]).to_csv(trace, index=False)
    power = tmp_path / "power.csv"
    pd.DataFrame({"Time": [120, 150], "n1": [100.0, 300.0]}).to_csv(power, index=False)
    out = tmp_path / "jobs.csv"
    precompute_jobs(str(trace), str(power), None, None, str(out))
    df = pd.read_csv(out)
    assert {"job_id", "gpu_num", "power_mean", "power_max", "power_std",
            "temp_mean", "util_mean", "state"} <= set(df.columns)
    assert df.loc[df.job_id == "j1", "power_max"].iloc[0] == 300.0
    assert df.loc[df.job_id == "j2", "temp_mean"].iloc[0] == 0.0  # no temp csv -> 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_precompute.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.precompute_features'` (add an empty `scripts/__init__.py` if needed for import).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/precompute_features.py
"""Offline, run-once (on the DigitalOcean box where the raw telemetry lives):
AcmeTrace job trace + per-signal telemetry CSVs -> small data/jobs.csv.
The only step that reads the large telemetry; never imported at runtime."""
import pandas as pd

_META = ["job_id", "type", "node_num", "gpu_num", "cpu_num",
         "duration", "queue", "mem_per_pod_GB", "state", "start_time", "end_time"]
_AGG = ["mean", "max", "std"]

def _window_aggs(signal_csv, start, end):
    if signal_csv is None:
        return {a: 0.0 for a in _AGG}
    df = pd.read_csv(signal_csv)
    win = df[(df["Time"] >= start) & (df["Time"] <= end)]
    vals = win[[c for c in win.columns if c != "Time"]].to_numpy().ravel()
    if vals.size == 0:
        return {a: 0.0 for a in _AGG}
    return {"mean": float(vals.mean()), "max": float(vals.max()),
            "std": float(vals.std())}

def precompute_jobs(trace_csv, power_csv, temp_csv, util_csv, out_csv):
    trace = pd.read_csv(trace_csv)
    rows = []
    for _, j in trace.iterrows():
        row = {k: j.get(k) for k in _META}
        for name, csv in (("power", power_csv), ("temp", temp_csv), ("util", util_csv)):
            aggs = _window_aggs(csv, j["start_time"], j["end_time"])
            for a in _AGG:
                row[f"{name}_{a}"] = aggs[a]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_precompute.py -v`
Expected: 1 passed.

- [ ] **Step 5: README data setup + remove empty src/**

Add a `## Data` subsection to `README.md` describing the offline precompute and the runtime file:

```markdown
## Data (one-time, offline)
The 80GB AcmeTrace telemetry never runs in the app. On the box where it lives:
`python scripts/precompute_features.py` (wire the AcmeTrace paths) → writes small
`data/jobs.csv` (one row/job: metadata + telemetry aggregates + label). The sim
replays that file, warm-started past the first 100 incidents.
```

Then remove the leftover empty package dir:

```bash
git rm -r --ignore-unmatch src/ 2>/dev/null; rmdir src 2>/dev/null || true
```

- [ ] **Step 6: Run full suite + commit**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass, 0 warnings.

```bash
git add scripts/ tests/test_precompute.py README.md
git commit -m "feat: precompute_features script; README data setup; drop empty src/"
```

---

## Self-Review

- **Spec coverage:** offline precompute (Task 7) · streaming accretion + warm-start (Task 3, sim in Task 6) · per-job binary label (Task 1) · model-form+feature action space (Task 4 `train_and_validate`, Task 5 instruction) · ROC-AUC time-ordered gate (Tasks 1–2, 4) · record contents = metadata + aggregates (Tasks 1, 7) · sensory tool over history (Task 4 `get_sensory`) · dashboard model card + /model (Task 6) · Xid-honesty (Task 5) · src/ cleanup (Task 7). Covered.
- **Type consistency:** `build_xy(history, features) -> (X, y)`; `time_split(X, y, val_frac) -> (Xtr,ytr,Xval,yval)`; `fit_candidate(model_type, features, Xtr, ytr)`; `auc(estimator, Xval, yval)`; `maybe_promote(estimator, model_type, features, val_auc)`; `INCUMBENT.version/.model_type/.features/.auc`; `stream.HISTORY`, `stream.FAIL_STATES`, `warm_start(path, n)`, `stream_jobs(path, start)`; `train_and_validate(model_type, features) -> {"trained",...}`. Names identical across tasks 1→6 and their tests.
- **Placeholder scan:** every code/test step carries complete code; no TBD/TODO.
- **Open risk (flagged, not blocking):** `precompute_features.py`'s telemetry join assumes per-signal CSVs with a `Time` column keyed to job `[start_time, end_time]`; the exact AcmeTrace file layout is finalized on the DO box (Task 7 test verifies schema + window logic on synthetic input). `find_correlated_failures` still reads the trace file rather than `HISTORY` — acceptable for this iteration; align to `HISTORY` later if the demo needs strict "data so far" correlation.
