import type { Incident } from "../types";

function hhmmss(ts: string) {
  return ts.slice(11, 19);
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
              <span className="mono-num">{hhmmss(inc.ts)}</span>
              <span className="mono-num">+{inc.correlatedCount} corr</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
