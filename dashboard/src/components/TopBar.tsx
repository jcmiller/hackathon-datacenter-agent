import type { Fleet, Incident, Meta } from "../types";

export function TopBar({
  meta,
  incidents,
  fleet,
  feedCollapsed,
  setFeedCollapsed,
  triageCollapsed,
  setTriageCollapsed,
  curveCollapsed,
  setCurveCollapsed,
  onComputerUse,
}: {
  meta: Meta | null;
  incidents: Incident[];
  fleet: Fleet | null;
  feedCollapsed: boolean;
  setFeedCollapsed: (c: boolean) => void;
  triageCollapsed: boolean;
  setTriageCollapsed: (c: boolean) => void;
  curveCollapsed: boolean;
  setCurveCollapsed: (c: boolean) => void;
  onComputerUse?: () => void;
}) {
  const active = incidents.filter((i) => i.state === "triaging").length;
  // Derive nodes-hit from the current faulted cells so it stays consistent with
  // the live "faulted" count instead of a static fixture total.
  const nodesHit = fleet
    ? new Set(
        fleet.cells.filter((c) => c.status === "fault").map((c) => c.node),
      ).size
    : null;
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
          <button
            onClick={() => setCurveCollapsed(!curveCollapsed)}
            style={{
              background: curveCollapsed ? 'rgba(255,255,255,0.03)' : 'rgba(99,102,241,0.15)',
              color: curveCollapsed ? '#64748b' : '#a5b4fc',
              border: '1px solid rgba(255,255,255,0.08)',
              padding: '4px 10px',
              borderRadius: 4,
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              outline: 'none'
            }}
          >
            {curveCollapsed ? "Show Predictor" : "Hide Predictor"}
          </button>
          {onComputerUse && (
            <button
              onClick={onComputerUse}
              style={{
                background: 'rgba(16,185,129,0.15)',
                color: '#6ee7b7',
                border: '1px solid rgba(16,185,129,0.3)',
                padding: '4px 12px',
                borderRadius: 4,
                fontSize: 11,
                fontWeight: 700,
                cursor: 'pointer',
                outline: 'none',
                letterSpacing: '0.02em',
              }}
            >
              🖥 Computer Use
            </button>
          )}
        </div>
      </div>
      <div className="kpis">
        <Kpi v={meta ? meta.totalGpus.toLocaleString() : "—"} l="GPUs" />
        <Kpi
          v={fleet ? fleet.faulted.toLocaleString() : "—"}
          l="faulted"
          cls="crit"
        />
        <Kpi v={nodesHit != null ? String(nodesHit) : "—"} l="nodes hit" cls="warn" />
        <Kpi v={String(active)} l="triaging" cls="accent" />
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
