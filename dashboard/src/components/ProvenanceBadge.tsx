import type { DataSource, Provenance, SourceBadge } from "../types";

// Honest data-source badge. Judges must never mistake illustrative fixtures for
// live telemetry, so every state is labelled and the provenance note is exposed on
// hover (bead 31n). Real = derived-real Kalos onsets; fixture/offline = explicit
// demo data; unavailable = nothing honest to show.
const META: Record<
  DataSource,
  { label: string; cls: string; live: boolean }
> = {
  real_substrate: { label: "REAL · derived Kalos onsets", cls: "prov-real", live: true },
  fixture: { label: "FIXTURE · illustrative demo", cls: "prov-fixture", live: false },
  synthetic: { label: "SYNTHETIC · demo curve", cls: "prov-fixture", live: false },
  trace: { label: "TRACE · job-trace replay", cls: "prov-fixture", live: false },
  offline: { label: "OFFLINE · static fixture (no backend)", cls: "prov-fixture", live: false },
  unavailable: { label: "UNAVAILABLE · no honest data", cls: "prov-unavail", live: false },
};

function provTitle(p: Provenance | null): string {
  if (!p) return "data source provenance";
  return (
    p.fixture_note ||
    p.note ||
    p.source ||
    `kind: ${p.kind ?? "unknown"}`
  );
}

export function ProvenanceBadge({ source }: { source: SourceBadge | null }) {
  const ds = source?.dataSource ?? "unavailable";
  const meta = META[ds] ?? META.unavailable;
  return (
    <span
      className={`prov-badge ${meta.cls}`}
      title={provTitle(source?.provenance ?? null)}
    >
      <span className={`prov-dot ${meta.live ? "live" : ""}`} />
      {meta.label}
    </span>
  );
}
