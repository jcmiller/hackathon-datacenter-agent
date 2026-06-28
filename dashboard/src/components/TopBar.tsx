import type { Fleet, Incident, Meta } from "../types";

export function TopBar({
  meta,
  incidents,
  fleet,
  feedCollapsed,
  setFeedCollapsed,
  triageCollapsed,
  setTriageCollapsed,
}: {
  meta: Meta | null;
  incidents: Incident[];
  fleet: Fleet | null;
  feedCollapsed: boolean;
  setFeedCollapsed: (c: boolean) => void;
  triageCollapsed: boolean;
  setTriageCollapsed: (c: boolean) => void;
}) {
  const active = incidents.filter((i) => i.state === "triaging").length;
  return (
    <header className="panel topbar">
      <div className="brand" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span>
          GPU<span className="dot">·</span>SITTER
        </span>
        <span className="dim" style={{ fontWeight: 400 }}>
          on-call RCA
        </span>
        
        <div style={{ display: 'flex', gap: 6, marginLeft: 20 }}>
          <button 
            onClick={() => setFeedCollapsed(!feedCollapsed)}
            style={{
              background: feedCollapsed ? 'rgba(255,255,255,0.03)' : 'rgba(99,102,241,0.15)',
              color: feedCollapsed ? '#64748b' : '#a5b4fc',
              border: '1px solid rgba(255,255,255,0.08)',
              padding: '4px 10px',
              borderRadius: 4,
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              outline: 'none'
            }}
          >
            {feedCollapsed ? "Show Feed" : "Hide Feed"}
          </button>
          <button 
            onClick={() => setTriageCollapsed(!triageCollapsed)}
            style={{
              background: triageCollapsed ? 'rgba(255,255,255,0.03)' : 'rgba(99,102,241,0.15)',
              color: triageCollapsed ? '#64748b' : '#a5b4fc',
              border: '1px solid rgba(255,255,255,0.08)',
              padding: '4px 10px',
              borderRadius: 4,
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              outline: 'none'
            }}
          >
            {triageCollapsed ? "Show Triage" : "Hide Triage"}
          </button>
        </div>
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
