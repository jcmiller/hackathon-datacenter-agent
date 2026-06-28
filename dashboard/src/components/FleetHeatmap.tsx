import { useMemo } from "react";
import type { Fleet, FleetCell } from "../types";

// color a cell: faults pulse red; active GPUs ramp green->amber by temp; idle faint.
function cellColor(c: FleetCell): string {
  if (c.status === "fault") return "var(--crit)";
  if (c.status === "idle") return "#2a2521";
  const t = c.temp ?? 0;
  // 30C -> green, 75C -> amber
  const f = Math.max(0, Math.min(1, (t - 30) / 45));
  const g = { r: 127, g: 168, b: 107 };
  const a = { r: 217, g: 164, b: 65 };
  const mix = (k: "r" | "g" | "b") => Math.round(g[k] + (a[k] - g[k]) * f);
  return `rgb(${mix("r")},${mix("g")},${mix("b")})`;
}

export function FleetHeatmap({
  fleet,
  selectedGpu,
  onSelectGpu,
}: {
  fleet: Fleet | null;
  selectedGpu: string | null;
  onSelectGpu: (gpu: string) => void;
}) {
  const nodes = useMemo(() => {
    if (!fleet) return [];
    const m = new Map<string, FleetCell[]>();
    for (const c of fleet.cells) {
      if (!m.has(c.node)) m.set(c.node, []);
      m.get(c.node)!.push(c);
    }
    // stable node order (by IP) so faulted nodes are interspersed with healthy
    // ones — shows the cascade as a mosaic across the fleet, not a wall of red
    const ipKey = (n: string) =>
      n.split(".").map((p) => p.padStart(3, "0")).join(".");
    return [...m.entries()]
      .map(([node, cells]) => ({
        node,
        cells: cells.sort((a, b) => a.idx - b.idx),
        faults: cells.filter((c) => c.status === "fault").length,
      }))
      .sort((a, b) => ipKey(a.node).localeCompare(ipKey(b.node)));
  }, [fleet]);

  return (
    <section className="col panel">
      <div className="panel-title">
        <span>Fleet heatmap · DCGM</span>
        <span className="legend">
          <span>
            <span className="sw" style={{ background: "var(--crit)" }} />
            Xid fault
          </span>
          <span>
            <span className="sw" style={{ background: "var(--ok)" }} />
            active
          </span>
          <span>
            <span className="sw" style={{ background: "#2a2521" }} />
            idle
          </span>
          {fleet && (
            // Not a live clock — the fixture captures a single cascade instant,
            // so label it as the snapshot time rather than implying it ticks.
            <span className="faint mono-num" title={`fleet snapshot — ${fleet.ts}`}>
              as of {fleet.ts.slice(11, 19)}
            </span>
          )}
        </span>
      </div>
      <div className="fleet-wrap">
        {!fleet ? (
          <div className="empty">loading fleet…</div>
        ) : (
          <div className="fleet-grid">
            {nodes.map((n) => (
              <div className="node-block" key={n.node} title={n.node}>
                <div className="nid">{n.node.replace("172.31.", "·")}</div>
                <div className="gpu-row">
                  {n.cells.map((c) => {
                    const gid = `${c.node}-${c.idx}`;
                    return (
                      <div
                        key={gid}
                        className={`gpu-cell ${c.status === "fault" ? "pulse" : ""} ${
                          selectedGpu === gid ? "sel" : ""
                        }`}
                        style={{ background: cellColor(c) }}
                        title={`${gid} · ${
                          c.status === "fault"
                            ? `Xid ${c.xid}`
                            : `${c.temp ?? "—"}°C · ${c.util ?? "—"}%`
                        }`}
                        onClick={() => onSelectGpu(gid)}
                      />
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
