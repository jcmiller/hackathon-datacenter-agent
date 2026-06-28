import type { Fleet, Incident, Meta } from "../types";

export function TopBar({
  meta,
  incidents,
  fleet,
}: {
  meta: Meta | null;
  incidents: Incident[];
  fleet: Fleet | null;
}) {
  const active = incidents.filter((i) => i.state === "triaging").length;
  return (
    <header className="panel topbar">
      <div className="brand">
        GPU<span className="dot">·</span>SITTER
        <span className="dim" style={{ fontWeight: 400, marginLeft: 8 }}>
          on-call RCA
        </span>
      </div>
      <div className="kpis">
        <Kpi v={meta ? meta.totalGpus.toLocaleString() : "—"} l="GPUs" />
        <Kpi
          v={fleet ? fleet.faulted.toLocaleString() : "—"}
          l="faulted"
          cls="crit"
        />
        <Kpi v={meta ? String(meta.nodesAffected) : "—"} l="nodes hit" cls="warn" />
        <Kpi v={String(active)} l="triaging" cls="accent" />
        <Kpi v="4m" l="MTTR" />
      </div>
    </header>
  );
}

function Kpi({ v, l, cls }: { v: string; l: string; cls?: string }) {
  return (
    <div className="kpi">
      <span className={`v mono-num ${cls ?? ""}`}>{v}</span>
      <span className="l">{l}</span>
    </div>
  );
}
