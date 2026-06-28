import type { Point, TelemetryWindow } from "../types";

function Sparkline({
  points,
  color,
  unit,
}: {
  points: Point[];
  color: string;
  unit: string;
}) {
  const vals = points.map((p) => p[1]).filter((v): v is number => v != null);
  const last = vals.length ? vals[vals.length - 1] : null;
  const max = vals.length ? Math.max(...vals) : 1;
  const min = vals.length ? Math.min(...vals) : 0;
  const W = 220;
  const H = 46;
  const span = max - min || 1;
  const n = points.length || 1;
  const path = points
    .map((p, i) => {
      const x = (i / (n - 1)) * W;
      const v = p[1] ?? min;
      const y = H - ((v - min) / span) * H;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
      <path d={path} fill="none" stroke={color} strokeWidth={1.5} />
      {last != null && (
        <text
          x={W}
          y={12}
          textAnchor="end"
          fill={color}
          fontSize="10"
          fontFamily="var(--mono)"
        >
          {last.toFixed(0)}
          {unit}
        </text>
      )}
    </svg>
  );
}

function lastVal(points: Point[]): number | null {
  for (let i = points.length - 1; i >= 0; i--) if (points[i][1] != null) return points[i][1];
  return null;
}

export function TelemetryStrip({ tele }: { tele: TelemetryWindow | null }) {
  const fields = [
    {
      key: "power" as const,
      field: "DCGM_FI_DEV_POWER_USAGE",
      unit: "W",
      color: "var(--accent)",
    },
    {
      key: "temp" as const,
      field: "DCGM_FI_DEV_GPU_TEMP",
      unit: "°C",
      color: "var(--warn)",
    },
    {
      key: "util" as const,
      field: "DCGM_FI_DEV_GPU_UTIL",
      unit: "%",
      color: "var(--info)",
    },
  ];
  return (
    <section className="col panel">
      <div className="panel-title">
        <span>Telemetry · ±3 min around fault</span>
        {tele && <span className="faint">{tele.gpu}</span>}
      </div>
      {!tele ? (
        <div className="empty">select an incident to inspect its GPU telemetry</div>
      ) : (
        <div className="tele">
          {fields.map((f) => {
            const pts = tele.series[f.key];
            const v = lastVal(pts);
            return (
              <div className="spark" key={f.key}>
                <div className="head">
                  <span className="field">{f.field}</span>
                </div>
                <div className="val" style={{ color: f.color }}>
                  {v != null ? v.toFixed(0) : "—"}
                  <span className="field"> {f.unit}</span>
                </div>
                <Sparkline points={pts} color={f.color} unit={f.unit} />
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
