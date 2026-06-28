import { useEffect, useState } from "react";
import type { LearningCurveData } from "../types";

// Self-improving predictor learning curve — hand-rolled inline SVG (no chart dep).
// Climbing line through promoted versions vs a flat no-skill floor, with the
// rejected keep-if-better candidates as ghost dots and a real-Kalos anchor.

const W = 960;
const H = 280;
const PAD_L = 56;
const PAD_R = 150;
const PAD_T = 26;
const PAD_B = 34;
const PLOT_W = W - PAD_L - PAD_R;
const PLOT_H = H - PAD_T - PAD_B;
const CURVE_W = PLOT_W * 0.74; // versions occupy the left, ghost lane + anchor right

const Y_MIN = 0.45;
const Y_MAX = 0.75;

const yPix = (v: number) =>
  PAD_T + (1 - (v - Y_MIN) / (Y_MAX - Y_MIN)) * PLOT_H;
const xVersion = (i: number, n: number) =>
  PAD_L + (n <= 1 ? 0 : i / (n - 1)) * CURVE_W;

export function LearningCurve() {
  const [data, setData] = useState<LearningCurveData | null>(null);

  useEffect(() => {
    fetch("/api/learning-curve")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: LearningCurveData) => setData(d))
      .catch((e) => console.warn("learning-curve load failed", e));
  }, []);

  if (!data) return null;

  const curve = data.curve;
  const n = curve.length;
  const pts = curve.map((c, i) => ({
    ...c,
    x: xVersion(i, n),
    y: yPix(c.roc_auc),
  }));
  const linePath = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");

  const rejected = data.rounds.filter((r) => !r.promoted);
  const ghostX0 = PAD_L + CURVE_W + 36;
  const ghostStep = 40;

  const anchorVal = data.real_data_reference.best_real.roc_auc;
  const anchorX = PAD_L + PLOT_W + 6;
  const anchorY = yPix(anchorVal);

  const v3 = curve[curve.length - 1];
  const yTicks = [0.5, 0.6, 0.7];

  return (
    <section className="panel lc-panel">
      <div className="panel-title">
        <span>Self-improving predictor · learning curve</span>
        <span className="faint">
          {data.n_promotions} promotions · held-out {data.primary_metric}
        </span>
      </div>

      <div className="lc-chart">
        <svg viewBox={`0 0 ${W} ${H}`} className="lc-svg" preserveAspectRatio="xMidYMid meet">
          {/* y gridlines + labels */}
          {yTicks.map((t) => (
            <g key={t}>
              <line
                className="lc-grid"
                x1={PAD_L}
                x2={PAD_L + PLOT_W}
                y1={yPix(t)}
                y2={yPix(t)}
              />
              <text className="lc-axis" x={PAD_L - 8} y={yPix(t) + 3} textAnchor="end">
                {t.toFixed(2)}
              </text>
            </g>
          ))}

          {/* no-skill baseline floor — the OFF/ON contrast */}
          <line
            className="lc-floor"
            x1={PAD_L}
            x2={PAD_L + PLOT_W}
            y1={yPix(0.5)}
            y2={yPix(0.5)}
          />
          <text className="lc-floor-label" x={PAD_L + 6} y={yPix(0.5) - 6}>
            no-skill prior (0.500)
          </text>

          {/* rejected keep-if-better candidates — faint ghost dots */}
          {rejected.map((r, j) => (
            <g key={`rej-${r.round}`}>
              <circle
                className="lc-ghost"
                cx={ghostX0 + j * ghostStep}
                cy={yPix(r.roc_auc)}
                r={4}
              />
              <text
                className="lc-ghost-label"
                x={ghostX0 + j * ghostStep}
                y={yPix(r.roc_auc) + 16}
                textAnchor="middle"
              >
                {r.roc_auc.toFixed(2)}
              </text>
            </g>
          ))}
          <text
            className="lc-ghost-caption"
            x={ghostX0 + ((rejected.length - 1) * ghostStep) / 2}
            y={PAD_T + PLOT_H + 24}
            textAnchor="middle"
          >
            rejected candidates
          </text>

          {/* climbing line through promoted versions */}
          <path className="lc-line" d={linePath} />
          {pts.map((p) => (
            <g key={p.version}>
              <circle className="lc-dot" cx={p.x} cy={p.y} r={5} />
              <text className="lc-dot-label" x={p.x} y={p.y - 14} textAnchor="middle">
                {p.model_type}
              </text>
              <text className="lc-dot-auc" x={p.x} y={p.y - 2} textAnchor="middle">
                {p.roc_auc.toFixed(3)}
              </text>
              <text
                className="lc-axis"
                x={p.x}
                y={PAD_T + PLOT_H + 24}
                textAnchor="middle"
              >
                {p.version}
              </text>
            </g>
          ))}

          {/* real-Kalos anchor marker */}
          <line
            className="lc-anchor-guide"
            x1={ghostX0 + (rejected.length - 1) * ghostStep + 14}
            x2={anchorX}
            y1={anchorY}
            y2={anchorY}
          />
          <path
            className="lc-anchor"
            d={`M${anchorX},${anchorY - 6} L${anchorX + 6},${anchorY} L${anchorX},${anchorY + 6} L${anchorX - 6},${anchorY} Z`}
          />
          <text className="lc-anchor-label" x={anchorX + 12} y={anchorY - 4}>
            real Kalos
          </text>
          <text className="lc-anchor-val" x={anchorX + 12} y={anchorY + 9}>
            {anchorVal.toFixed(3)}
          </text>
          <text className="lc-anchor-sub" x={anchorX + 12} y={anchorY + 21}>
            leakage-free
          </text>
        </svg>
      </div>

      <div className="lc-footer">
        <div className="lc-hypothesis">
          <span className="lc-hyp-tag">v{String(data.final_incumbent.version).replace(/^v?/, "")} · {data.final_incumbent.model_type}</span>
          {v3.hypothesis}
        </div>
        <div className="lc-badges">
          <span className="lc-badge lc-badge-anchor">same loop · real Kalos · leakage-free</span>
          <span className="lc-badge lc-badge-honest">synthetic demo data · real fits &amp; held-out ROC-AUC</span>
        </div>
      </div>
    </section>
  );
}
