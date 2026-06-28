"""Sim backend: SSE incident stream, streaming triage, model card, feedback."""

import asyncio
import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..agent.agent import triage_stream
from ..agent.tools import MODEL_STATE_PATH, SOP_PATH
from ..detection import classifier
from . import computer_use as cu
from . import dashboard_substrate as ds

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

# Committed honest demo fixture (bead jds): a portable fallback so /api/monitor
# renders off the droplet, when the real artifacts above are absent. Built by
# gpusitter.app.monitor_fixture; paths anchored to the same tracked on-disk dir.
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "early_detection"
MONITOR_FIXTURE_DATA_PATH = str(_FIXTURE_DIR / "features.csv")
MONITOR_FIXTURE_REGISTRY_PATH = str(_FIXTURE_DIR / "registry")

# Self-improvement learning curve (v0->vN keep-if-better history). The repo-root
# artifact (built by scripts/learning_curve_demo.py) wins; a committed in-package
# copy is the off-droplet fallback so the curve always renders. Overridable in tests.
LEARNING_CURVE_PATH = "docs/learning_curve.json"
LEARNING_CURVE_FALLBACK = str(Path(__file__).parent / "fixtures" / "learning_curve.json")

_DASHBOARD = Path(__file__).parent / "dashboard" / "index.html"
_assets_dir = _DASHBOARD.parent / "assets"
_fixtures_dir = _DASHBOARD.parent / "fixtures"

# Dashboard data source selection (bead h7w). The REAL artifact is the committed,
# real-derived substrate built from edge-detected Kalos Xid onsets (bead t7p); it
# ships in the package so the dashboard serves real-derived telemetry off-droplet.
# When that artifact is absent we fall back — explicitly badged — to the
# hand-derived demo fixtures (bead 69c). Both paths overridable in tests.
DASHBOARD_SUBSTRATE_DIR = str(ds.SUBSTRATE_DIR)
DASHBOARD_FIXTURE_DIR = str(_fixtures_dir)
DASHBOARD_FIXTURE_NOTE = (
    "Illustrative hand-derived demo fixture (Aug-17 06:00 hero snapshot) — NOT "
    "live telemetry. Raw Kalos data unchanged. Configure the real substrate to "
    "serve derived-real onsets."
)

if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")
if _fixtures_dir.exists():
    app.mount("/fixtures", StaticFiles(directory=str(_fixtures_dir)), name="fixtures")

_incidents_cache: dict[str, list] = {}
_started = {"done": False}


@dataclass
class DashboardSource:
    """A resolved dashboard data source plus its honesty badge (bead h7w).

    ``data_source`` is ``"real_substrate"`` for the committed real-derived artifact
    or ``"fixture"`` for the demo fallback; ``provenance`` is the manifest (real)
    or a fixture badge so callers can never mistake illustrative data for live
    telemetry.
    """

    meta: dict
    fleet: dict
    incidents: list[dict]
    telemetry: dict[str, dict]
    data_source: str
    provenance: dict


def _resolve_dashboard_source() -> DashboardSource | None:
    """Resolve the dashboard data source, real-first, with explicit fixture fallback.

    Precedence (mirrors the jds/aow ``_resolve_monitor_registry`` contract):

    1. **Real substrate wins** — the committed, real-derived artifact (manifest
       ``kind=real``; edge-detected Xid onsets, bead t7p) at
       ``DASHBOARD_SUBSTRATE_DIR``. Its manifest becomes the provenance.
    2. **Explicit demo fixture** — the hand-derived snapshot at
       ``DASHBOARD_FIXTURE_DIR`` (bead 69c), badged ``fixture`` with a note so it
       cannot pass as live telemetry.
    3. **Neither present** — ``None`` (the honest unavailable floor).
    """
    if ds.substrate_available(DASHBOARD_SUBSTRATE_DIR):
        sub = ds.load_substrate(DASHBOARD_SUBSTRATE_DIR)
        return DashboardSource(
            meta=sub.meta,
            fleet=sub.fleet,
            incidents=sub.incidents,
            telemetry=sub.telemetry,
            data_source="real_substrate",
            provenance=dict(sub.manifest),
        )

    fx = Path(DASHBOARD_FIXTURE_DIR)
    if (fx / "incidents.json").exists():
        return _load_fixture_source(fx)
    return None


def _load_fixture_source(fixture_dir: Path) -> DashboardSource:
    """Build a :class:`DashboardSource` from the hand-derived demo fixtures (badged)."""

    def _read(name: str, default):
        p = fixture_dir / name
        return json.loads(p.read_text()) if p.exists() else default

    meta = _read("meta.json", {})
    telemetry: dict[str, dict] = {}
    tel_dir = fixture_dir / "telemetry"
    if tel_dir.exists():
        for f in sorted(tel_dir.glob("INC-*.json")):
            telemetry[f.stem] = json.loads(f.read_text())
    provenance = {
        "kind": "fixture",
        "telemetryKind": "fixture",
        "source": meta.get("source"),
        "fixture_note": DASHBOARD_FIXTURE_NOTE,
    }
    return DashboardSource(
        meta=meta,
        fleet=_read("fleet.json", {}),
        incidents=_read("incidents.json", []),
        telemetry=telemetry,
        data_source="fixture",
        provenance=provenance,
    )


def _incident_feed() -> tuple[list[dict], str, dict]:
    """Incidents for the SSE stream + (data_source, provenance), real-first.

    An absolute ``TRACE_CSV`` is treated as an explicit job-trace override (tests
    / custom traces) and routed to the legacy ``load_incidents`` path so existing
    behavior is preserved. Otherwise the dashboard substrate resolver decides,
    preferring real over fixture; only an absent substrate AND absent fixture
    degrades to the job-trace fallback.
    """
    if Path(TRACE_CSV).is_absolute():
        from ..rca.job_join import load_incidents as _load

        key = str(TRACE_CSV)
        if key not in _incidents_cache:
            _incidents_cache[key] = _load(TRACE_CSV)
        return _incidents_cache[key], "trace", {"kind": "trace", "path": key}

    src = _resolve_dashboard_source()
    if src is not None:
        return src.incidents, src.data_source, src.provenance

    from ..rca.job_join import load_incidents as _load

    key = str(TRACE_CSV)
    if key not in _incidents_cache:
        _incidents_cache[key] = _load(TRACE_CSV)
    return _incidents_cache[key], "trace", {"kind": "trace", "path": key}


@app.get("/")
def index():
    return FileResponse(str(_DASHBOARD))


def _json(obj):
    """json.dumps with datetime → ISO string fallback."""
    return json.dumps(obj, default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o))


@app.get("/api/incidents")
async def incidents(request: Request):
    """Stream the incident feed as SSE, real-first with explicit fixture badging.

    Backward compatibility (bead h7w review): the provenance is emitted as a NAMED
    SSE event (``event: provenance``), which the browser ``EventSource.onmessage``
    handler ignores — only unnamed ``message`` frames reach it. So every default
    frame the current React consumer sees is a real Incident (it dereferences
    ``incident.gpu``); a non-Incident frame on the default channel would crash it.
    Each incident is still tagged ``dataSource`` (a harmless extra field) so a
    fixture incident can never be rendered as live telemetry. The 8co.31n React
    rewrite consumes the provenance event via ``addEventListener('provenance')``.
    """
    feed, data_source, provenance = _incident_feed()
    header = {"dataSource": data_source, "provenance": provenance}

    async def gen():
        # Named event: invisible to the default onmessage handler, so it cannot
        # break clients that treat every onmessage frame as an Incident.
        yield f"event: provenance\ndata: {_json(header)}\n\n"
        for inc in feed:
            if await request.is_disconnected():
                break
            yield f"data: {_json({**inc, 'dataSource': data_source})}\n\n"
            await asyncio.sleep(STEP_SECONDS)

    return StreamingResponse(gen(), media_type="text/event-stream")


def _unavailable(kind: str) -> dict:
    """Honest unavailable floor when neither real substrate nor fixture exists."""
    return {
        "available": False,
        "dataSource": "unavailable",
        "reason": f"no real dashboard substrate and no committed demo fixture for {kind}",
    }


@app.get("/api/meta")
def get_meta():
    """Dashboard meta (event window, cascade ts, fleet totals) + source provenance."""
    src = _resolve_dashboard_source()
    if src is None:
        return _unavailable("meta")
    return {**src.meta, "dataSource": src.data_source, "provenance": src.provenance}


@app.get("/api/fleet")
def get_fleet():
    """Real per-GPU fleet snapshot (heatmap cells) + source provenance.

    Fault cells are edge-detected hero-burst members only; every other GPU's
    status is derived from utilization, never a latched Xid value (bead t7p).
    """
    src = _resolve_dashboard_source()
    if src is None:
        return _unavailable("fleet")
    return {**src.fleet, "dataSource": src.data_source, "provenance": src.provenance}


@app.get("/api/telemetry")
def get_telemetry(incident: str | None = None):
    """Per-GPU real telemetry window around a selected incident + provenance.

    Without an ``incident`` query param, returns the index of available incident
    ids. An unknown id returns 404 with the available ids so the caller cannot
    silently render an empty chart.
    """
    src = _resolve_dashboard_source()
    if src is None:
        return _unavailable("telemetry")
    if incident is None:
        return {
            "incidents": sorted(src.telemetry),
            "dataSource": src.data_source,
            "provenance": src.provenance,
        }
    rec = src.telemetry.get(incident)
    if rec is None:
        return JSONResponse(
            {
                "error": f"no telemetry window for {incident}",
                "available": sorted(src.telemetry),
                "dataSource": src.data_source,
            },
            status_code=404,
        )
    return {
        **rec,
        "incident": incident,
        "dataSource": src.data_source,
        "provenance": src.provenance,
    }


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


def _resolve_monitor_registry():
    """Resolve the one registry that BOTH ``/api/model`` and ``/api/monitor`` serve.

    Model identity must agree across the two surfaces (bead aow). The only way to
    guarantee that off-droplet is to resolve the artifact ONCE, here, with the exact
    same precedence both endpoints honor:

    1. **Real droplet artifacts win** — a persisted incumbent at
       ``MONITOR_REGISTRY_PATH`` *and* the feature table at ``MONITOR_DATA_PATH``
       (``fixture=False``). Both are required: the monitor needs the table to score,
       and demanding it here keeps the model card and the per-row scores on the same
       registry.
    2. **Committed honest demo fixture** (bead jds) — when the real artifacts are
       absent (off the droplet, no ~80 GB trace) but the committed fixture table
       exists, serve the fixture-backed registry (``fixture=True`` +
       ``fixture_note``).
    3. **Nothing honest to serve** — returns ``None``.

    Returns ``(registry, data_path, is_fixture, fixture_note)`` or ``None``.
    """
    import os

    from ..detection.harness import ModelRegistry

    registry = ModelRegistry(MONITOR_REGISTRY_PATH)
    if registry.incumbent is not None and os.path.exists(MONITOR_DATA_PATH):
        return registry, MONITOR_DATA_PATH, False, None
    if os.path.exists(MONITOR_FIXTURE_DATA_PATH):
        from .monitor_fixture import FIXTURE_NOTE, load_fixture_registry

        fixture_reg = load_fixture_registry(
            MONITOR_FIXTURE_DATA_PATH, MONITOR_FIXTURE_REGISTRY_PATH
        )
        return fixture_reg, MONITOR_FIXTURE_DATA_PATH, True, FIXTURE_NOTE
    return None


def _registry_model_card(
    card, *, is_fixture: bool = False, fixture_note: str | None = None
) -> dict:
    """Render a rigorous registry ModelCard as the canonical dashboard model card.

    This is the SAME incumbent ``/api/monitor`` scores from, so the headline metric
    here (held-out ROC-AUC behind the strict time-split + permutation baseline + dual
    leakage probes + holdout-identity pin) and the per-row operational scores there are
    one model, not two contradictory stories (bead aow). ``fixture`` mirrors the
    ``/api/monitor`` flag so a fixture-backed card is badged illustrative the same way.
    """
    primary_is_auc = card.primary_metric == "roc_auc"
    out = {
        "model": {
            "version": card.version,
            "model_type": card.model_type,
            "features": list(card.features),
            # Back-compat field the React bundle reads; only meaningful when the
            # primary metric is ROC-AUC (it is, by default).
            "val_auc": round(card.primary_value, 3) if primary_is_auc else None,
            "primary_metric": card.primary_metric,
            "primary_value": round(card.primary_value, 4),
            "n_samples": card.n_train + card.n_test,
            "n_train": card.n_train,
            "n_test": card.n_test,
            "holdout_id": (card.holdout_id or "")[:12],
            "training_window": list(card.training_window),
        },
        "source": "registry",
        "rigorous": True,
        "fixture": is_fixture,
    }
    if is_fixture:
        out["fixture_note"] = fixture_note
    return out


def _provisional_model_card(model: dict) -> dict:
    """Wrap the weak in-process triage card, explicitly badged as NOT canonical.

    The live triage agent's ``classifier.INCUMBENT`` (promoted by ``maybe_promote`` on
    ``val_auc > incumbent`` alone — no leakage probe, no holdout-identity guard, no
    time-split) is a fast in-process fit, never the self-improvement headline. It
    surfaces only when no rigorous registry exists (e.g. before bead jds ships the
    prebuilt registry) and is flagged so the dashboard cannot mistake it for the
    keep-if-better registry incumbent (bead aow, AC#2/#4).
    """
    return {
        "model": model,
        "source": "in_process",
        "rigorous": False,
        "note": (
            "provisional in-process triage fit (maybe_promote, no leakage/holdout "
            "guards) — not the rigorous keep-if-better registry incumbent"
        ),
    }


@app.get("/api/model")
def get_model():
    """Canonical model card for the dashboard.

    Source of truth is the rigorous keep-if-better ``ModelRegistry`` (bead glf,
    detection/harness.py): strict time-ordered split, 8x permutation baseline, dual
    leakage probes, holdout-identity pin. When a registry incumbent exists it is the
    canonical, honest self-improvement surface and is the SAME model ``/api/monitor``
    scores (bead aow). The legacy in-process ``classifier`` path is a provisional
    fallback, explicitly badged.

    Off-droplet the canonical incumbent is the committed demo fixture (bead jds),
    resolved by :func:`_resolve_monitor_registry` — the SAME resolver ``/api/monitor``
    uses — so both surfaces serve one fixture-backed model (model.version ==
    monitor.model_version), badged ``fixture=true``.

    Model-type menu divergence is intentional: the live triage ``classifier`` offers
    logreg/tree/gboost for fast in-process fits, while the rigorous harness offers
    logreg/hgb deliberately matched to the lys offline eval models so harness AUC
    equals the standalone eval report. The canonical headline menu is logreg/hgb.
    """
    resolved = _resolve_monitor_registry()
    if resolved is not None:
        registry, _data_path, is_fixture, fixture_note = resolved
        return _registry_model_card(
            registry.incumbent, is_fixture=is_fixture, fixture_note=fixture_note
        )

    m = classifier.INCUMBENT
    if m is not None:
        return _provisional_model_card(
            {
                "version": m.version,
                "model_type": m.model_type,
                "features": m.features,
                "val_auc": round(m.auc, 3) if m.auc else None,
                "n_samples": m.n_samples,
            }
        )
    state = classifier.load_state(MODEL_STATE_PATH)
    if state:
        return _provisional_model_card(state)
    return {"model": None, "message": "not yet trained — need ≥1 incident with metrics"}


@app.get("/api/monitor")
def get_monitor(budget: float | None = None, horizon: float | None = None):
    """Per-row risk scores + alert/miss status from the incumbent (bead i6k).

    Scores the labeled per-GPU feature table (the lys/r7j substrate) with the usable
    pickled incumbent, derives alert-budget thresholds, and runs the horizon-grid
    miss detector. Exposes the per-row risk timeline, alert flags, and per-horizon
    recall (caught onsets / total) to the dashboard.

    Artifact resolution (bead jds): the REAL droplet artifacts win
    (``MONITOR_DATA_PATH`` + a persisted incumbent at ``MONITOR_REGISTRY_PATH``,
    ``fixture:false``). When they are absent — e.g. off the droplet, without the
    ~80 GB trace — it falls back to the committed honest demo fixture
    (``fixture:true`` + a ``fixture_note`` labeling it illustrative). It degrades to
    ``available:false`` only when even the committed fixture is gone.

    Optional ``budget`` / ``horizon`` query params narrow the budget/horizon grid.
    """
    from ..detection import monitor
    from ..detection.harness import ModelRegistry, load_dataset

    # Shared resolver (bead aow): the same registry /api/model serves, so the model
    # card and these per-row scores are always one model — never a fixture-vs-real mix.
    resolved = _resolve_monitor_registry()
    if resolved is None:
        return {
            "available": False,
            "reason": "no monitor data/registry and no committed demo fixture",
            "incumbent": ModelRegistry(MONITOR_REGISTRY_PATH).describe_incumbent(),
        }

    registry, data_path, is_fixture, fixture_note = resolved
    df = load_dataset(data_path)
    scorer = monitor.RowScorer.from_registry(registry)

    budgets = (budget,) if budget else monitor.DEFAULT_BUDGETS
    horizons = (horizon,) if horizon else monitor.DEFAULT_HORIZONS_S
    report = monitor.monitor_report(df, scorer, budgets=budgets, horizons_s=horizons)
    report["fixture"] = is_fixture
    if is_fixture:
        report["fixture_note"] = fixture_note
    return report


@app.get("/api/learning-curve")
def get_learning_curve():
    """The v0->vN self-improvement curve — the keep-if-better registry history.

    Each entry carries the held-out ROC-AUC, the signal gap over the no-skill
    baseline, and the hypothesis/reflection that drove the iteration, so the
    dashboard can render the self-improvement story as a real learning curve
    rather than a single static metric (bead 31n AC).

    Resolution mirrors the other surfaces: the repo-root artifact at
    ``LEARNING_CURVE_PATH`` wins; an in-package committed copy is the off-droplet
    fallback; an honest ``available:false`` floor when neither exists. The curve is
    explicitly badged ``synthetic`` (the table is a deliberately weak demo, see the
    embedded ``honest_note`` / ``real_data_reference``).
    """
    for path in (LEARNING_CURVE_PATH, LEARNING_CURVE_FALLBACK):
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            data["available"] = True
            data.setdefault("dataSource", "synthetic")
            return data
    return {
        "available": False,
        "dataSource": "unavailable",
        "reason": "no learning-curve artifact (run scripts/learning_curve_demo.py)",
    }


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
