import { useEffect, useState } from "react";
import type { Point, PredictGpu, TelemetryWindow } from "../types";

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

const LABEL_VAR: Record<"alert" | "watch" | "ok", string> = {
  alert: "var(--crit)",
  watch: "var(--warn)",
  ok: "var(--ok)",
};

function FailureBadge({ incidentId }: { incidentId: string }) {
  const [pred, setPred] = useState<PredictGpu | null>(null);

  useEffect(() => {
    let cancelled = false;
    setPred(null);
    fetch("/api/predict-gpu", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ incident_id: incidentId }),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`predict-gpu ${res.status}`);
        return res.json() as Promise<PredictGpu>;
      })
      .then((p) => {
        if (!cancelled) setPred(p);
      })
      .catch(() => {
        if (!cancelled) setPred(null);
      });
    return () => {
      cancelled = true;
    };
  }, [incidentId]);

  // Render nothing until we have an available prediction — never a fake number.
  if (!pred || !pred.available) return null;

  const color = LABEL_VAR[pred.label];
  return (
    <div className="predict">
      <div className="predict-row">
        <span className="predict-label">FAILURE LIKELIHOOD</span>
        <span className="predict-val" style={{ color }}>
          {pred.likelihood.toFixed(2)}
        </span>
        <span className="predict-tag" style={{ color }}>
          ⚠ {pred.label.toUpperCase()}
        </span>
      </div>
      <div className="predict-note">
        {pred.note} · model v{pred.model.version} · AUC{" "}
        {pred.model.val_auc.toFixed(3)}
      </div>
    </div>
  );
}

export function TelemetryStrip({
  tele,
  incidentId,
}: {
  tele: TelemetryWindow | null;
  incidentId: string | null;
}) {
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
        <>
          {incidentId && <FailureBadge incidentId={incidentId} />}
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
        </>
      )}
    </section>
  );
}
