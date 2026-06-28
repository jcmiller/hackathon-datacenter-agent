"""Sim backend: SSE incident stream, streaming triage, model card, feedback."""

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..agent.agent import triage_stream
from ..agent.tools import MODEL_STATE_PATH, SOP_PATH
from ..detection import classifier
from . import computer_use as cu

app = FastAPI()

JOBS_CSV = "data/jobs.csv"
TRACE_CSV = "data/acme-util/data/job_trace/trace_kalos.csv"
WARM_START_INCIDENTS = 100
STEP_SECONDS = 3

# Operational reactive-trigger substrate (bead i6k): the labeled per-GPU feature
# table (lys/r7j) the incumbent scores — NOT jobs.csv — and the registry holding
# the usable pickled incumbent. Overridable in tests / on the droplet.
MONITOR_DATA_PATH = "data/early_detection.parquet"
MONITOR_REGISTRY_PATH = "models/early_detection"

_DASHBOARD = Path(__file__).parent / "dashboard" / "index.html"
_assets_dir = _DASHBOARD.parent / "assets"
_fixtures_dir = _DASHBOARD.parent / "fixtures"

if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")
if _fixtures_dir.exists():
    app.mount("/fixtures", StaticFiles(directory=str(_fixtures_dir)), name="fixtures")

_incidents_cache: dict[str, list] = {}
_started = {"done": False}


def _get_incidents() -> list[dict]:
    """Load incidents from fixture JSON (relative path) or trace CSV (absolute/injected path)."""
    # lazy import: avoids pulling the RCA/job-join stack at module load time
    from ..rca.job_join import load_incidents as _load

    fixtures_path = _DASHBOARD.parent / "fixtures" / "incidents.json"
    key = str(TRACE_CSV)
    if key not in _incidents_cache:
        # Use fixtures when the path is the default relative path; skip for absolute paths
        # (e.g. test-injected tmp files) so tests can inject their own CSV.
        if not Path(TRACE_CSV).is_absolute() and fixtures_path.exists():
            with open(fixtures_path) as f:
                _incidents_cache[key] = json.load(f)
        else:
            _incidents_cache[key] = _load(TRACE_CSV)
    return _incidents_cache[key]


@app.get("/")
def index():
    return FileResponse(str(_DASHBOARD))


def _json(obj):
    """json.dumps with datetime → ISO string fallback."""
    return json.dumps(obj, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o))


@app.get("/api/incidents")
async def incidents(request: Request):
    async def gen():
        for inc in _get_incidents():
            if await request.is_disconnected():
                break
            yield f"data: {_json(inc)}\n\n"
            await asyncio.sleep(STEP_SECONDS)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/triage")
async def do_triage(incident: dict):
    loop = asyncio.get_running_loop()
    event_q: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for ev in triage_stream(incident):
                asyncio.run_coroutine_threadsafe(event_q.put(ev), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(event_q.put(None), loop).result()

    threading.Thread(target=run, daemon=True).start()

    async def gen():
        while True:
            item = await event_q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/model")
def get_model():
    m = classifier.INCUMBENT
    if m is not None:
        return {
            "model": {
                "version": m.version,
                "model_type": m.model_type,
                "features": m.features,
                "val_auc": round(m.auc, 3) if m.auc else None,
                "n_samples": m.n_samples,
            }
        }
    state = classifier.load_state(MODEL_STATE_PATH)
    if state:
        return {"model": state}
    return {"model": None, "message": "not yet trained — need ≥1 incident with metrics"}


@app.get("/api/monitor")
def get_monitor(budget: float | None = None, horizon: float | None = None):
    """Per-row risk scores + alert/miss status from the incumbent (bead i6k).

    Scores the labeled per-GPU feature table (the lys/r7j substrate) with the usable
    pickled incumbent, derives alert-budget thresholds, and runs the horizon-grid
    miss detector. Exposes the per-row risk timeline, alert flags, and per-horizon
    recall (caught onsets / total) to the dashboard. Degrades honestly when the
    dataset or a persisted incumbent is absent.

    Optional ``budget`` / ``horizon`` query params narrow the budget/horizon grid.
    """
    import os

    from ..detection import monitor
    from ..detection.harness import ModelRegistry, load_dataset

    registry = ModelRegistry(MONITOR_REGISTRY_PATH)
    if registry.incumbent is None:
        return {"available": False, "reason": "no persisted incumbent in registry"}
    if not os.path.exists(MONITOR_DATA_PATH):
        return {
            "available": False,
            "reason": f"feature table not found at {MONITOR_DATA_PATH}",
            "incumbent": registry.describe_incumbent(),
        }
    df = load_dataset(MONITOR_DATA_PATH)
    scorer = monitor.RowScorer.from_registry(registry)
    budgets = (budget,) if budget else monitor.DEFAULT_BUDGETS
    horizons = (horizon,) if horizon else monitor.DEFAULT_HORIZONS_S
    return monitor.monitor_report(df, scorer, budgets=budgets, horizons_s=horizons)


@app.post("/api/feedback")
async def record_feedback(body: dict):
    incident_id = body.get("incident_id", "")
    outcome = body.get("outcome", "")
    if not incident_id or not outcome:
        return JSONResponse({"error": "incident_id and outcome required"}, status_code=400)
    if not Path(SOP_PATH).exists():
        return JSONResponse({"error": "no SOP entries yet"}, status_code=404)

    with open(SOP_PATH) as f:
        entries = json.load(f)

    idx = next((i for i, e in enumerate(entries) if e.get("incident_id") == incident_id), None)
    if idx is None:
        return JSONResponse({"error": f"no entry found for {incident_id}"}, status_code=404)

    entries[idx]["outcome"] = outcome
    with open(SOP_PATH, "w") as f:
        json.dump(entries, f, indent=2)

    from ..agent.memory import _embed, _load_vectors, _record_text, _save_vectors

    text = _record_text(entries[idx]) + f" outcome:{outcome}"
    vec = _embed(text)
    if vec is not None:
        vectors = _load_vectors()
        if len(vectors) > idx:
            vectors[idx] = vec
            _save_vectors(vectors)

    return {"recorded": True, "incident_id": incident_id, "outcome": outcome}


@app.post("/api/computer-use")
async def computer_use_session(body: dict):
    """Stream a Gemini 3.5 Flash computer use remediation session as SSE."""
    task = body.get("task") or None

    async def gen():
        async for event in cu.run_session(task=task):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# Legacy endpoints for backwards compatibility
@app.get("/model")
def model_legacy():
    return get_model()


@app.post("/triage")
async def triage_legacy(incident: dict):
    return {"disposition": triage_stream.__module__}
