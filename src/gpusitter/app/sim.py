"""Sim backend: warm-started accreting job stream + /model + triage endpoint."""
import asyncio, json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from ..detection import stream
from ..detection import classifier
from ..agent.agent import triage  # noqa: F401 — re-exported so tests can monkeypatch

app = FastAPI()

JOBS_CSV = "data/jobs.csv"
# Must be < the number of FAILED jobs in JOBS_CSV, else warm-start consumes the
# whole file and the live SSE stream emits nothing. (mock jobs.csv has 232 FAILED.)
WARM_START_INCIDENTS = 100
STEP_SECONDS = 3

_DASHBOARD = Path(__file__).parent / "dashboard" / "index.html"
_started = {"done": False}


@app.get("/")
def index():
    return FileResponse(str(_DASHBOARD))


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
            # warm_start populates stream.HISTORY internally
            stream.warm_start(JOBS_CSV, WARM_START_INCIDENTS)
            _started["done"] = True
        start = len(stream.HISTORY)
        # stream_jobs appends each yielded record to stream.HISTORY internally
        for r in stream.stream_jobs(JOBS_CSV, start):
            if await request.is_disconnected():
                break
            if r["state"] in stream.FAIL_STATES:
                yield f"data: {json.dumps(r)}\n\n"
            await asyncio.sleep(STEP_SECONDS)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/triage")
async def do_triage(incident: dict):
    return {"disposition": triage(incident)}
