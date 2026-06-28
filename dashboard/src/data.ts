// Live data loaders. Normal mode reads the REAL FastAPI dashboard endpoints
// (/api/*, bead h7w) which serve derived-real Kalos onsets with explicit, badged
// fixture fallback server-side. The static /fixtures JSON is kept ONLY as an
// offline demo fallback for when the API itself is unreachable (e.g. a pure
// static `vite preview` with no backend) — and it is badged `offline` so it can
// never be mistaken for live telemetry (bead 31n).
import type {
  Fleet,
  LearningCurve,
  Meta,
  ModelResponse,
  MonitorReport,
  TelemetryWindow,
} from "./types";

const api = "/api";
const fixtures = "/fixtures";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`fetch ${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

// Try the live API first; on any network/HTTP failure fall back to the static
// offline fixture and stamp it `offline` so the UI badges it honestly.
async function liveOrOffline<T extends { dataSource?: string }>(
  apiPath: string,
  offlinePath: string,
): Promise<T> {
  try {
    return await getJSON<T>(apiPath);
  } catch {
    const data = await getJSON<T>(offlinePath);
    return { ...data, dataSource: "offline" };
  }
}

export const loadFleet = () =>
  liveOrOffline<Fleet>(`${api}/fleet`, `${fixtures}/fleet.json`);

export const loadMeta = () =>
  liveOrOffline<Meta>(`${api}/meta`, `${fixtures}/meta.json`);

export const loadTelemetry = (incidentId: string) =>
  liveOrOffline<TelemetryWindow>(
    `${api}/telemetry?incident=${encodeURIComponent(incidentId)}`,
    `${fixtures}/telemetry/${incidentId}.json`,
  );

// Self-improvement surfaces — no static offline fixture; if the API is down the
// caller renders an explicit unavailable state rather than fabricating numbers.
export const loadModel = () => getJSON<ModelResponse>(`${api}/model`);
export const loadMonitor = () => getJSON<MonitorReport>(`${api}/monitor`);
export const loadLearningCurve = () =>
  getJSON<LearningCurve>(`${api}/learning-curve`);
