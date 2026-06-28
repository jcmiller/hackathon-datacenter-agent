"""Sim backend: SSE incident stream + triage endpoint."""
import asyncio, json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from backend.loader import load_incidents
from backend.agent import triage  # noqa: F401 — re-exported so tests can monkeypatch

app = FastAPI()

TRACE_CSV = "data/trace_kalos.csv"
STEP_SECONDS = 3

_DASHBOARD = Path(__file__).parent / "dashboard" / "index.html"

# Simple cache: maps csv_path -> list[dict]. Cleared by tests via sim._incidents_cache.clear().
_incidents_cache: dict[str, list] = {}


def _get_incidents() -> list[dict]:
    path = TRACE_CSV
    if path not in _incidents_cache:
        _incidents_cache[path] = load_incidents(path)
    return _incidents_cache[path]


@app.get("/")
def index():
    return FileResponse(str(_DASHBOARD))


@app.get("/incidents")
async def incidents(request: Request):
    async def gen():
        for inc in _get_incidents():
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps(inc)}\n\n"
            await asyncio.sleep(STEP_SECONDS)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/triage")
async def do_triage(incident: dict):
    return {"disposition": triage(incident)}
