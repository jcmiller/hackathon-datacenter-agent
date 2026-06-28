// Fixture loaders. Today these read static JSON from /fixtures (generated from
// real AcmeTrace Kalos telemetry). Swapping to the live backend later = point
// these fetches at /api/* — the shapes are identical (see types.ts).
import type {
  Fleet,
  Incident,
  Meta,
  TelemetryWindow,
} from "./types";

const base = "/fixtures";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`fetch ${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

export const loadIncidents = () => getJSON<Incident[]>(`${base}/incidents.json`);
export const loadFleet = () => getJSON<Fleet>(`${base}/fleet.json`);
export const loadMeta = () => getJSON<Meta>(`${base}/meta.json`);
export const loadTelemetry = (incidentId: string) =>
  getJSON<TelemetryWindow>(`${base}/telemetry/${incidentId}.json`);
