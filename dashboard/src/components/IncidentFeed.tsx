import type { Incident } from "../types";

function fmtTs(ts: string) {
  const d = new Date(ts);
  const mo = d.toLocaleString("en-US", { month: "short" });
  const day = d.getDate();
  const hhmm = d.toTimeString().slice(0, 5);
  return `${mo} ${day} · ${hhmm}`;
}

export function IncidentFeed({
  incidents,
  selectedId,
  onSelect,
}: {
  incidents: Incident[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <section className="col panel">
      <div className="panel-title">
        <span>Incident feed</span>
        <span className="faint">{incidents.length}</span>
      </div>
      <div className="body">
        {incidents.map((inc) => (
          <div
            key={inc.id}
            className={`inc ${inc.severity} ${selectedId === inc.id ? "sel" : ""}`}
            onClick={() => onSelect(inc.id)}
          >
            <div className="top">
              <span className="xid">
                Xid {inc.xid}{" "}
                <span className={inc.severity === "crit" ? "crit" : "warn"}>
                  ●
                </span>
              </span>
              {inc.hero && <span className="badge">cascade</span>}
            </div>
            <div className="lbl">{inc.xidLabel}</div>
            <div className="meta">
              <span>
                {inc.gpu.node}-{inc.gpu.idx}
              </span>
              <span className="mono-num">{fmtTs(inc.ts)}</span>
              <span
                className="mono-num"
                title={`GPU impact count — ${inc.correlatedCount} other GPUs across the fleet failed in the same time window as this fault (i.e. correlated with it). A high count means a shared, cluster-wide cause (network / power / cooling / a bad job) rather than an isolated GPU fault.`}
              >
                {inc.correlatedCount} GPUs impacted
              </span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
