"""Sim backend: SSE incident stream + triage endpoint."""
import asyncio
import json
import threading
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import logging

from backend.loader import load_incidents
from backend.agent import triage_stream

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

TRACE_CSV = "data/acme-util/data/job_trace/trace_kalos.csv"
STEP_SECONDS = 3

_DASHBOARD = Path(__file__).parent / "dashboard" / "index.html"

# Mount static assets if build exists
_assets_dir = _DASHBOARD.parent / "assets"
if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

# Mount fixtures if build exists
_fixtures_dir = _DASHBOARD.parent / "fixtures"
if _fixtures_dir.exists():
    app.mount("/fixtures", StaticFiles(directory=str(_fixtures_dir)), name="fixtures")

# Simple cache: maps csv_path -> list[dict]. Cleared by tests via sim._incidents_cache.clear().
_incidents_cache: dict[str, list] = {}


def _get_incidents() -> list[dict]:
    path = TRACE_CSV
    if path not in _incidents_cache:
        # Use high-fidelity fixtures for the default relative path;
        # skip for absolute paths (e.g. test-injected tmp files).
        fixtures_path = _DASHBOARD.parent / "fixtures" / "incidents.json"
        if not Path(path).is_absolute() and fixtures_path.exists():
            try:
                with open(fixtures_path, "r") as f:
                    _incidents_cache[path] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load incidents.json fixture: {e}")
                _incidents_cache[path] = load_incidents(path)
        else:
            _incidents_cache[path] = load_incidents(path)
    return _incidents_cache[path]


@app.get("/")
def index():
    return FileResponse(str(_DASHBOARD))


@app.get("/api/incidents")
async def incidents(request: Request):
    async def gen():
        for inc in _get_incidents():
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps(inc)}\n\n"
            await asyncio.sleep(STEP_SECONDS)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/triage")
async def do_triage(incident: dict):
    loop = asyncio.get_event_loop()
    event_q: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for ev in triage_stream(incident):
                asyncio.run_coroutine_threadsafe(event_q.put(ev), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(event_q.put(None), loop).result()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    async def gen():
        while True:
            item = await event_q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
