# GPU On-Call RCA Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reactive AI agent that automates the GPU-fleet on-call engineer — after a node failure fires, it pulls telemetry, finds correlated failures, matches past incidents, decides a disposition, and logs the resolution — all grounded in real AcmeTrace data.

**Architecture:** A FastAPI "sim backend" replays real AcmeTrace job/telemetry data and fires incidents over an SSE stream (mocking DCGM/IPMI/Prometheus endpoints). When an incident fires, a Google ADK + Gemini agent runs a ReAct loop over real tools (telemetry query, correlation, incident-memory search, technician paging, SOP append). Every claim cites a number the tools returned. A minimal web dashboard shows the incident feed and a live agent-reasoning tab.

**Tech Stack:** Python 3.11, pandas, FastAPI + Uvicorn (SSE), Google ADK (`google-adk`) with Gemini 2.5 Flash, pytest. Dashboard = single static page (vanilla JS + EventSource) served by FastAPI. Memory = JSON file.

## Global Constraints

- **Language:** Python only. Senior-dev-simple — minimum code, no speculative abstraction, no config flags nobody asked for (CLAUDE.md).
- **Reactive only:** No failure prediction. Agent acts *after* an incident fires.
- **Never reinvent the sensor:** Agent calls tools; tools read real AcmeTrace data. Agent must not compute raw stats the tools already return.
- **Grounding rule:** Every agent conclusion must reference a value returned by a tool call. No ungrounded claims.
- **Data source:** `trace_kalos.csv` (Kalos cluster — only cluster with `fail_time`). GitHub CSV, loaded from local `data/`.
- **Telemetry source:** AcmeTrace util files (CSV for power/DRAM e.g. `ipmi/GPU_AB_Power.csv`; some signals only as `util_pkl/*.pkl`). Loader handles both by extension. AcmeTrace carries only healthy-path util/power/temp + the `NODE_FAIL` label — **no Xid codes, no ECC/NVLink error counters, no cause data.**
- **Real field names (credibility):** tools and sim use the verified DCGM identifiers a real A100 fleet emits — `DCGM_FI_DEV_POWER_USAGE` (W), `DCGM_FI_DEV_GPU_TEMP` (°C), `DCGM_FI_DEV_GPU_UTIL` (%). Source: NVIDIA `dcgm_fields.h`. Not toy column names.
- **Cause inference (honesty rule):** since AcmeTrace has no Xid/ECC, the agent *infers a likely cause class* from telemetry pattern + domain priors (Task 3) — it never claims to read a real Xid. Synthetic Xid injection (Stretch) is allowed only if labeled "injected" in the UI.
- **Harness:** **Google ADK** local runner (`google-adk`, model `gemini-2.5-flash`) — LOCKED. Right shape for an embedded, controllable, low-latency domain reasoner with local tools. (Managed Agents' hosted sandbox is the wrong shape for the brain — see Stretch for where it *does* fit.)
- **Incident definition:** A job-trace row where `state` in {`NODE_FAIL`, `FAILED`} and `fail_time` is present.
- **Timestamps:** Treat all `*_time` columns as Unix epoch seconds (verify in Task 1, Step 2).

---

## File Structure

```
data/
  trace_kalos.csv              # job trace (provided, downloaded separately)
  util/*.csv                   # telemetry (Time + per-node-IP columns)
backend/                       # all app code (frontend/ becomes a sibling later — not yet scoped)
  __init__.py
  loader.py                    # read trace + telemetry, expose incidents & windows
  memory.py                    # JSON incident/SOP store: search + append
  priors.py                    # Meta co-occurrence domain knowledge (agent skill)
  tools.py                     # RCA tool functions the agent calls
  sim.py                       # FastAPI: replay incidents over SSE, telemetry endpoints
  agent.py                     # Google ADK agent: Gemini + tools + priors
  dashboard/index.html         # minimal SSE dashboard (moves to frontend/ when scoped)
tests/
  test_loader.py
  test_memory.py
  test_tools.py
  test_agent.py
conftest.py                    # puts repo root on sys.path (backend importable)
pytest.ini                     # testpaths = tests
data/                          # AcmeTrace data + sop.json (repo root, gitignored; run all cmds from repo root)
```

> **Imports & cwd:** modules import as `from backend.X import ...`; tests run via `pytest` from the repo root; the server runs as `uvicorn backend.sim:app` from the repo root so relative `data/` resolves.

**Lane mapping (Notion board):** loader+sim = Data + Backend/Harness · tools+priors+agent = Sensor Tools + Backend/Harness · dashboard = Frontend.

---

## Real Endpoint Fidelity (research-grounded)

The sim must look like the real sensor stack so the agent calls the same surface a production fleet exposes. Mapping (from endpoint research):

| Real endpoint | What it exposes | AcmeTrace artifact | Our sim mocks it as |
|---|---|---|---|
| **DCGM** (dcgmi / dcgm-exporter→Prometheus `:9400`) | `DCGM_FI_DEV_POWER_USAGE`, `_GPU_TEMP`, `_GPU_UTIL` | `util_pkl/gpu_power_*`, `gpu_temp_*`, `util_gpu_*`; `ipmi/GPU_AB_Power.csv` | `get_telemetry` returns these field names |
| **IPMI / Redfish** (BMC, out-of-band node power) | `ipmitool dcmi power reading` / `PowerConsumedWatts` | `util_pkl/server_power.pkl` | node-power field (stretch) |
| **Xid** (kernel log `NVRM: Xid`) | last fault code (48/79/94…) | **absent** | inferred from priors; synthetic-injected only if labeled |
| **Prometheus / node_exporter** (`:9100`) | `node_cpu_*`, `node_memory_*` | `util_pkl/util_cpu_mem_*` | not in MVP |

**MCP note (board card #3):** no first-party DCGM MCP server exists. Authentic options: (a) wrap `backend/tools.py` as a FastMCP server, or (b) expose a dcgm-exporter-style `/metrics` surface and point a Prometheus MCP server (`pab1it0/prometheus-mcp-server`) at it. MCP is **Stretch under harness B**, **core under harness A** (Managed Agents sandbox reaches tools via MCP).

---

### Task 1: AcmeTrace Loader

**Files:**
- Create: `backend/loader.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: `data/trace_kalos.csv`, telemetry CSV at `data/util/*.csv`.
- Produces:
  - `load_incidents(path: str) -> list[dict]` — each dict has keys `job_id, type, node_num, gpu_num, state, fail_time, duration, user`. Only rows with `state` in {`NODE_FAIL`,`FAILED`} and non-null `fail_time`. Sorted ascending by `fail_time`.
  - `telemetry_window(csv_path: str, start: int, end: int) -> dict` — returns `{"samples": int, "mean": float, "max": float, "min": float}` aggregated across all node columns for rows whose `Time` is within `[start, end]`.
  - `correlated_failures(incidents: list[dict], fail_time: int, window: int) -> list[dict]` — incidents whose `fail_time` is within `±window` seconds of `fail_time` (excluding the exact same `job_id`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loader.py
import pandas as pd
from backend.loader import load_incidents, telemetry_window, correlated_failures

def _write_trace(tmp_path):
    df = pd.DataFrame([
        {"job_id":"j1","user":"u1","node_num":4,"gpu_num":32,"cpu_num":64,"type":"train",
         "state":"NODE_FAIL","submit_time":100,"start_time":110,"end_time":900,
         "duration":790,"queue":10,"gpu_time":1,"mem_per_pod_GB":80,"shared_mem_per_pod":1,
         "fail_time":880,"stop_time":900},
        {"job_id":"j2","user":"u2","node_num":1,"gpu_num":8,"cpu_num":16,"type":"eval",
         "state":"COMPLETED","submit_time":100,"start_time":110,"end_time":500,
         "duration":390,"queue":10,"gpu_time":1,"mem_per_pod_GB":80,"shared_mem_per_pod":1,
         "fail_time":None,"stop_time":500},
        {"job_id":"j3","user":"u3","node_num":2,"gpu_num":16,"cpu_num":32,"type":"train",
         "state":"FAILED","submit_time":100,"start_time":110,"end_time":920,
         "duration":810,"queue":10,"gpu_time":1,"mem_per_pod_GB":80,"shared_mem_per_pod":1,
         "fail_time":900,"stop_time":920},
    ])
    p = tmp_path/"trace.csv"; df.to_csv(p, index=False); return str(p)

def _write_util(tmp_path):
    df = pd.DataFrame({"Time":[870,885,895,910],
                       "10.0.0.1":[50,90,95,40],"10.0.0.2":[60,70,80,30]})
    p = tmp_path/"util.csv"; df.to_csv(p, index=False); return str(p)

def test_load_incidents_filters_and_sorts(tmp_path):
    inc = load_incidents(_write_trace(tmp_path))
    assert [i["job_id"] for i in inc] == ["j1","j3"]      # COMPLETED dropped, sorted by fail_time
    assert inc[0]["fail_time"] == 880

def test_telemetry_window_aggregates(tmp_path):
    w = telemetry_window(_write_util(tmp_path), 880, 900)
    assert w["samples"] == 2                               # rows at 885 and 895
    assert w["max"] == 95
    assert w["min"] == 70

def test_correlated_failures_within_window(tmp_path):
    inc = load_incidents(_write_trace(tmp_path))
    corr = correlated_failures(inc, fail_time=880, window=30)
    assert [c["job_id"] for c in corr] == ["j3"]           # j3 at 900 within ±30, j1 excluded (self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.loader'`.
Also manually confirm timestamp format: `python -c "import pandas as pd; print(pd.read_csv('data/trace_kalos.csv')[['fail_time']].dropna().head())"` — values should be ~10-digit Unix seconds. If they are datetime strings, add `pd.to_datetime(...).astype('int64')//10**9` in the loader.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/loader.py
import pandas as pd

FAIL_STATES = {"NODE_FAIL", "FAILED"}
_KEEP = ["job_id","type","node_num","gpu_num","state","fail_time","duration","user"]

def load_incidents(path):
    df = pd.read_csv(path)
    df = df[df["state"].isin(FAIL_STATES) & df["fail_time"].notna()]
    df = df.sort_values("fail_time")
    return df[_KEEP].to_dict("records")

def _read_telemetry(path):
    # AcmeTrace ships CSV for some signals, pickle for others.
    # SECURITY: read_pickle executes arbitrary code — only load .pkl from the
    # trusted InternLM/AcmeTrace release you downloaded yourself. Prefer CSV when both exist.
    return pd.read_pickle(path) if path.endswith(".pkl") else pd.read_csv(path)

def telemetry_window(csv_path, start, end):
    df = _read_telemetry(csv_path)
    win = df[(df["Time"] >= start) & (df["Time"] <= end)]
    cols = [c for c in win.columns if c != "Time"]
    vals = win[cols].to_numpy().ravel()
    if vals.size == 0:
        return {"samples": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
    return {"samples": len(win), "mean": float(vals.mean()),
            "max": float(vals.max()), "min": float(vals.min())}

def correlated_failures(incidents, fail_time, window):
    return [i for i in incidents
            if i["fail_time"] != fail_time
            and abs(i["fail_time"] - fail_time) <= window]
```

Note: `correlated_failures` excludes by matching `fail_time`; the self-incident is the one whose `fail_time` equals the query. If two real incidents share an exact `fail_time`, switch the exclusion to pass and compare `job_id` instead. Acceptable for MVP.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_loader.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/loader.py tests/test_loader.py
git commit -m "feat: AcmeTrace loader — incidents, telemetry windows, correlation"
```

---

### Task 2: Incident Memory / SOP Store

**Files:**
- Create: `backend/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: JSON file at `data/sop.json` (created on first append).
- Produces:
  - `search_incidents(incident_type: str, path: str = "data/sop.json") -> list[dict]` — past resolved records whose `type` matches `incident_type`. Empty list if file missing.
  - `append_incident(record: dict, path: str = "data/sop.json") -> None` — append a record (`type, summary, disposition, resolution`) to the store, creating the file if absent.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory.py
from backend.memory import search_incidents, append_incident

def test_search_empty_when_no_file(tmp_path):
    assert search_incidents("train", str(tmp_path/"sop.json")) == []

def test_append_then_search_roundtrip(tmp_path):
    p = str(tmp_path/"sop.json")
    append_incident({"type":"train","summary":"NODE_FAIL on 4 nodes",
                     "disposition":"page_technician","resolution":"replaced GPU"}, p)
    append_incident({"type":"eval","summary":"other","disposition":"restart",
                     "resolution":"ok"}, p)
    hits = search_incidents("train", p)
    assert len(hits) == 1
    assert hits[0]["resolution"] == "replaced GPU"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.memory'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/memory.py
import json, os

def search_incidents(incident_type, path="data/sop.json"):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        records = json.load(f)
    return [r for r in records if r.get("type") == incident_type]

def append_incident(record, path="data/sop.json"):
    records = []
    if os.path.exists(path):
        with open(path) as f:
            records = json.load(f)
    records.append(record)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/memory.py tests/test_memory.py
git commit -m "feat: JSON incident/SOP memory store"
```

---

### Task 3: Domain Priors (Agent Skill)

**Files:**
- Create: `backend/priors.py`

**Interfaces:**
- Produces: `DOMAIN_PRIORS: str` — a concise knowledge block (Meta 2024 co-occurrence priors) injected into the agent's system prompt so it reads GPU symptoms like a domain expert.

- [ ] **Step 1: Write the module (no test — static knowledge string)**

```python
# backend/priors.py
# Co-occurrence priors distilled from Meta's 2024 LLM-fleet reliability reporting.
# Loaded into the agent's instruction so it reasons about cause from symptoms.
DOMAIN_PRIORS = """\
GPU fleet failure domain knowledge (priors, not ground truth):
- NODE_FAIL usually means the scheduler lost the node: hardware fault, not user error.
- High sustained GPU power + thermal followed by a drop often precedes Xid 79
  (GPU fell off the bus / PCIe link loss).
- Repeated ECC/memory errors (Xid 48/63/64/94/95) point to a degrading GPU; the fix
  is usually drain + replace, not a job restart.
- Many nodes failing in the same short window with the same job type suggests a shared
  cause: a bad job image, a network/NCCL fault, or a shared power/cooling domain.
- A single isolated NODE_FAIL with normal neighbours suggests a single-node hardware fault.
- Disposition guide: shared-cause cluster -> escalate to datacenter ops; isolated
  hardware fault -> page technician to drain+replace; transient with healthy telemetry
  -> restart the job and watch.
"""
```

- [ ] **Step 2: Verify import**

Run: `python -c "from backend.priors import DOMAIN_PRIORS; print(len(DOMAIN_PRIORS))"`
Expected: prints a positive integer (non-empty).

- [ ] **Step 3: Commit**

```bash
git add backend/priors.py
git commit -m "feat: Meta co-occurrence domain priors as agent skill"
```

---

### Task 4: RCA Tools

**Files:**
- Create: `backend/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `backend.loader` (`telemetry_window`, `correlated_failures`), `backend.memory` (`search_incidents`, `append_incident`).
- Produces (plain functions the ADK agent wraps as tools; all return JSON-serializable dicts/strings):
  - `get_telemetry(fail_time: int, window: int = 120) -> dict` — telemetry around the incident keyed by real DCGM field names (`DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_GPU_TEMP`), each a window aggregate. Reads `POWER_CSV`, `TEMP_CSV`.
  - `find_correlated_failures(fail_time: int, window: int = 120) -> dict` — `{"count": int, "jobs": [job_id...], "shared_type": str|None}`. Reads incidents from `TRACE_CSV`.
  - `search_past_incidents(incident_type: str) -> dict` — `{"count": int, "matches": [...]}`.
  - `page_technician(node_info: str, reason: str) -> dict` — returns `{"paged": True, "ticket": "<id>", "reason": reason}` (simulated).
  - `record_resolution(incident_type: str, summary: str, disposition: str, resolution: str) -> dict` — appends to memory, returns `{"recorded": True}`.

Module-level constants `TRACE_CSV = "data/trace_kalos.csv"` and `TELEMETRY_CSV = "data/util/GPU_AB_Power.csv"` (override in tests via monkeypatch).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools.py
import backend.tools as tools

def test_get_telemetry(tmp_path, monkeypatch):
    import pandas as pd
    pw = tmp_path/"power.csv"; tp = tmp_path/"temp.csv"
    pd.DataFrame({"Time":[100,110,200],"n1":[10,90,10]}).to_csv(pw, index=False)
    pd.DataFrame({"Time":[100,110,200],"n1":[40,42,41]}).to_csv(tp, index=False)
    monkeypatch.setattr(tools, "POWER_CSV", str(pw))
    monkeypatch.setattr(tools, "TEMP_CSV", str(tp))
    out = tools.get_telemetry(fail_time=110, window=20)
    assert out["DCGM_FI_DEV_POWER_USAGE"]["max"] == 90
    assert out["DCGM_FI_DEV_GPU_TEMP"]["samples"] == 2

def test_find_correlated_failures(tmp_path, monkeypatch):
    import pandas as pd
    p = tmp_path/"trace.csv"
    pd.DataFrame([
        {"job_id":"a","type":"train","node_num":1,"gpu_num":8,"state":"NODE_FAIL",
         "fail_time":1000,"duration":1,"user":"u"},
        {"job_id":"b","type":"train","node_num":1,"gpu_num":8,"state":"NODE_FAIL",
         "fail_time":1050,"duration":1,"user":"u"},
    ]).to_csv(p, index=False)
    monkeypatch.setattr(tools, "TRACE_CSV", str(p))
    out = tools.find_correlated_failures(fail_time=1000, window=120)
    assert out["count"] == 1 and out["jobs"] == ["b"] and out["shared_type"] == "train"

def test_page_and_record(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "SOP_PATH", str(tmp_path/"sop.json"))
    assert tools.page_technician("node 4", "drain+replace")["paged"] is True
    assert tools.record_resolution("train","s","page_technician","replaced")["recorded"] is True
    assert tools.search_past_incidents("train")["count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.tools'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/tools.py
from backend.loader import load_incidents, telemetry_window, correlated_failures
from backend.memory import search_incidents, append_incident

TRACE_CSV = "data/trace_kalos.csv"
# Telemetry files mapped to the real DCGM field they represent.
POWER_CSV = "data/util/GPU_AB_Power.csv"   # DCGM_FI_DEV_POWER_USAGE (W)
TEMP_CSV = "data/util/gpu_temp_kalos.pkl"  # DCGM_FI_DEV_GPU_TEMP (C)
SOP_PATH = "data/sop.json"
_TICKET = {"n": 0}

def get_telemetry(fail_time, window=120):
    """GPU telemetry around an incident, keyed by the real DCGM field names a
    production fleet emits (dcgm-exporter). Values are window aggregates."""
    return {
        "DCGM_FI_DEV_POWER_USAGE": telemetry_window(POWER_CSV, fail_time - window, fail_time + window),
        "DCGM_FI_DEV_GPU_TEMP": telemetry_window(TEMP_CSV, fail_time - window, fail_time + window),
    }

def find_correlated_failures(fail_time, window=120):
    """Find other node failures near this incident in time."""
    inc = load_incidents(TRACE_CSV)
    corr = correlated_failures(inc, fail_time, window)
    types = {c["type"] for c in corr}
    return {"count": len(corr), "jobs": [c["job_id"] for c in corr],
            "shared_type": next(iter(types)) if len(types) == 1 else None}

def search_past_incidents(incident_type):
    """Retrieve resolved past incidents of the same type."""
    hits = search_incidents(incident_type, SOP_PATH)
    return {"count": len(hits), "matches": hits}

def page_technician(node_info, reason):
    """Simulate paging a datacenter technician."""
    _TICKET["n"] += 1
    return {"paged": True, "ticket": f"TKT-{_TICKET['n']:04d}",
            "node": node_info, "reason": reason}

def record_resolution(incident_type, summary, disposition, resolution):
    """Append this incident + resolution to the SOP memory."""
    append_incident({"type": incident_type, "summary": summary,
                     "disposition": disposition, "resolution": resolution}, SOP_PATH)
    return {"recorded": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/tools.py tests/test_tools.py
git commit -m "feat: RCA tools — telemetry, correlation, memory search, paging, record"
```

---

### Task 5: Google ADK Agent

**Files:**
- Create: `backend/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `backend.tools` (all five functions), `backend.priors.DOMAIN_PRIORS`.
- Produces:
  - `build_agent() -> google.adk.agents.Agent` — Gemini 2.5 Flash agent with the five tools and priors-injected instruction.
  - `triage(incident: dict) -> str` — runs the agent on one incident dict, returns the final disposition text. (Requires `GOOGLE_API_KEY` in env; the unit test only checks construction + tool registration, not a live call.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent.py
from backend.agent import build_agent

def test_agent_has_tools_and_priors():
    a = build_agent()
    names = {t.__name__ for t in a.tools}
    assert {"get_telemetry","find_correlated_failures","search_past_incidents",
            "page_technician","record_resolution"} <= names
    assert "NODE_FAIL" in a.instruction        # priors injected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.agent'`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/agent.py
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from backend.priors import DOMAIN_PRIORS
from backend import tools

INSTRUCTION = f"""You are the on-call engineer for a GPU training cluster.
An incident just fired. Do the triage a human on-call would do:
1. Call get_telemetry to see GPU power/temp (DCGM fields) around the failure time.
2. Call find_correlated_failures to see if other nodes failed in the same window.
3. Call search_past_incidents to reuse a known resolution for this incident type.
4. Decide a disposition: escalate to datacenter ops (shared-cause cluster),
   page_technician (isolated hardware fault), or restart-and-watch (healthy telemetry).
   Page the technician via the tool if hardware replacement is needed.
5. Call record_resolution to log what you found and decided.
Ground every statement in a number a tool returned. Be concise.

{DOMAIN_PRIORS}"""

def build_agent():
    return Agent(
        name="oncall_rca",
        model="gemini-2.5-flash",
        instruction=INSTRUCTION,
        tools=[tools.get_telemetry, tools.find_correlated_failures,
               tools.search_past_incidents, tools.page_technician,
               tools.record_resolution],
    )

def triage(incident):
    runner = InMemoryRunner(agent=build_agent(), app_name="rca")
    session = runner.session_service.create_session_sync(
        app_name="rca", user_id="demo")
    msg = types.Content(role="user", parts=[types.Part(
        text=f"Incident fired: {incident}. Triage it.")])
    final = ""
    for ev in runner.run(user_id="demo", session_id=session.id, new_message=msg):
        if ev.content and ev.content.parts:
            for part in ev.content.parts:
                if getattr(part, "text", None):
                    final = part.text
    return final
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent.py -v`
Expected: 1 passed. (No live API call — construction only.)

- [ ] **Step 5: Smoke-test a live triage (manual, needs `GOOGLE_API_KEY`)**

Run:
```bash
GOOGLE_API_KEY=... python -c "from backend.agent import triage; \
print(triage({'job_id':'j1','type':'train','node_num':4,'state':'NODE_FAIL','fail_time':1700000000}))"
```
Expected: A short disposition citing telemetry numbers and a decision. If the ADK API surface differs from the version installed, adjust the runner call per `pip show google-adk` docs — keep the tool list and instruction unchanged.

- [ ] **Step 6: Commit**

```bash
git add backend/agent.py tests/test_agent.py
git commit -m "feat: Google ADK + Gemini on-call RCA agent"
```

---

### Task 6: Sim Backend (SSE incident stream + dashboard)

**Files:**
- Create: `backend/sim.py`, `backend/dashboard/index.html`

**Interfaces:**
- Consumes: `backend.loader.load_incidents`, `backend.agent.triage`.
- Produces (FastAPI app `app`):
  - `GET /` → serves `dashboard/index.html`.
  - `GET /incidents` → SSE stream: emits the next incident every few seconds (replaying `trace_kalos.csv` in `fail_time` order, accelerated).
  - `POST /triage` (body: incident JSON) → `{"disposition": str}` by calling `triage`.

- [ ] **Step 1: Write the app**

```python
# backend/sim.py
import asyncio, json
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from backend.loader import load_incidents
from backend.agent import triage

app = FastAPI()
INCIDENTS = load_incidents("data/trace_kalos.csv")
STEP_SECONDS = 3  # demo replay speed

@app.get("/")
def index():
    return FileResponse("backend/dashboard/index.html")

@app.get("/incidents")
async def incidents(request: Request):
    async def gen():
        for inc in INCIDENTS:
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps(inc)}\n\n"
            await asyncio.sleep(STEP_SECONDS)
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/triage")
async def do_triage(incident: dict):
    return {"disposition": triage(incident)}
```

- [ ] **Step 2: Write the dashboard**

```html
<!-- backend/dashboard/index.html -->
<!doctype html><meta charset="utf-8"><title>GPU On-Call RCA</title>
<style>
 body{font:14px system-ui;margin:0;display:grid;grid-template-columns:1fr 1fr;height:100vh}
 .col{padding:16px;overflow:auto}#feed{border-right:1px solid #ddd}
 .card{border:1px solid #ddd;border-radius:8px;padding:10px;margin:8px 0;cursor:pointer}
 .fail{border-left:4px solid #e03}.trace{white-space:pre-wrap;background:#f6f6f6;padding:12px;border-radius:8px}
 h2{margin:0 0 8px}
</style>
<div class="col" id="feed"><h2>Incident feed</h2></div>
<div class="col"><h2>Agent triage</h2><div id="trace" class="trace">Click an incident…</div></div>
<script>
 const feed=document.getElementById('feed'), trace=document.getElementById('trace');
 new EventSource('/incidents').onmessage=e=>{
   const inc=JSON.parse(e.data), c=document.createElement('div');
   c.className='card fail';
   c.textContent=`${inc.state} · job ${inc.job_id} · ${inc.type} · ${inc.node_num} nodes`;
   c.onclick=async()=>{ trace.textContent='Agent triaging…';
     const r=await fetch('/triage',{method:'POST',headers:{'Content-Type':'application/json'},
       body:JSON.stringify(inc)});
     trace.textContent=(await r.json()).disposition; };
   feed.prepend(c);
 };
</script>
```

- [ ] **Step 3: Run the server and verify the stream**

Run: `GOOGLE_API_KEY=... uvicorn backend.sim:app --reload`
Then: open `http://localhost:8000`, confirm incident cards appear, click one, confirm the agent disposition renders. Also verify the stream endpoint directly: `curl -N http://localhost:8000/incidents | head -c 300` shows `data: {...}` lines.

- [ ] **Step 4: Commit**

```bash
git add backend/sim.py backend/dashboard/index.html
git commit -m "feat: sim backend SSE incident stream + dashboard"
```

---

### Task 7: End-to-End Demo Wiring

**Files:**
- Create: `README.md`, `requirements.txt`

**Interfaces:**
- Consumes: everything above.
- Produces: a one-command demo path + dependency pin list.

- [ ] **Step 1: Write `requirements.txt`**

```
pandas
fastapi
uvicorn[standard]
google-adk
google-genai
pytest
```

- [ ] **Step 2: Write `README.md` demo steps**

```markdown
# GPU On-Call RCA Agent
Reactive agent that automates GPU-fleet on-call triage on real AcmeTrace data.

## Setup
1. `pip install -r requirements.txt`
2. Put `trace_kalos.csv` in `data/` and a telemetry CSV at `data/util/GPU_AB_Power.csv`
   (from https://github.com/InternLM/AcmeTrace).
3. `export GOOGLE_API_KEY=...`

## Run
`uvicorn backend.sim:app --reload` → open http://localhost:8000

## Test
`pytest -v`

## What the agent does (per incident)
telemetry lookup → correlated-failure search → past-incident memory match →
disposition (escalate / page technician / restart) → record resolution.
```

- [ ] **Step 3: Full test pass**

Run: `pytest -v`
Expected: all tests in `tests/` pass (loader 3, memory 2, tools 3, agent 1).

- [ ] **Step 4: Commit**

```bash
git add README.md requirements.txt
git commit -m "docs: demo wiring, requirements, run instructions"
```

---

## Stretch (only after the critical path demos end-to-end)

- **DSPy improvement loop:** optimize the agent instruction against a labelled set of incidents whose correct disposition is known; the SOP memory becomes the training signal.
- **Robot-arm actuation:** a `hot_swap_gpu(node)` tool (simulated) the agent calls instead of paging, for the "physical actuation" demo beat.
- **Embedding memory:** replace `search_past_incidents` type-match with Gemini-embedding cosine similarity over past summaries.
- **MCP exposure:** wrap `backend/tools.py` as a FastMCP server so the tools are reusable by any MCP client, not just this agent.
- **Deployed coding/actuation agent (Gemini Managed Agents):** when a disposition needs *autonomous code/ops work* — generate a remediation script, write the runbook, draft the ticket/PR, patch a config — the ADK brain spins off a **deployed Managed Agent** in a Google sandbox (the "repair arm" of the Notion vision; sandbox + code-exec is the right tool here, unlike for the brain). Feasibility: **skills port cleanly** (same `SKILL.md` priors feed both); **local domain tools reach it only via MCP or a handoff payload** (for pure coding tasks it mostly uses its own sandbox tools and won't need them). Doubles as the host's "Managed Agents" point — used where it actually fits.

---

## Self-Review

- **Spec coverage:** Notion solution → monitoring/repair loop (Tasks 5–6), tracing tools DCGM/IPMI/Prom (Task 4 `get_telemetry` over real util CSV), human-in-loop paging (Task 4 `page_technician`), SOP memory (Tasks 2,4 `record_resolution`/`search_past_incidents`), Grafana-type dash + agent tab (Task 6), AcmeTrace data (Task 1), Meta priors as skill (Task 3). DSPy + robot arm = explicitly deferred to Stretch. Covered.
- **Reactive-only constraint:** no prediction task anywhere — agent triggers post-incident. Held.
- **Grounding:** instruction enforces tool-cited claims; every tool returns real numbers from real CSVs. Held.
- **Type consistency:** `fail_time` int and tool names are identical across loader → tools → agent → tests. `get_telemetry/find_correlated_failures/search_past_incidents/page_technician/record_resolution` spelled identically in Tasks 4, 5, and tests.
- **Open risk flagged:** exact telemetry filename and timestamp format are verified at runtime (Task 1 Step 2, Task 4 constants) — adjust the single constant if the downloaded file differs.
```
