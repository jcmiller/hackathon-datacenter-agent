import { useEffect, useState } from "react";
import type {
  BudgetReport,
  HorizonResult,
  LearningCurve,
  LearningCurvePoint,
  MonitorReport,
} from "../types";
import { loadLearningCurve, loadMonitor } from "../data";

// The self-improvement surface (bead 31n). Two honest, live views of the SAME
// keep-if-better early-warning model:
//   1. The learning curve — v0->vN held-out ROC-AUC over the no-skill baseline.
//   2. The operational monitor — per-horizon recall / missed onsets at a chosen
//      alert budget, plus the per-row risk timeline.
// Both are explicitly badged when fixture/synthetic so the demo stays honest: the
// signal is deliberately weak (real Kalos is a NO-GO predictor); the reusable
// deliverable is the self-improving LOOP, not a headline AUC.

function pct(x: number | undefined | null): string {
  return x != null && Number.isFinite(x) ? `${(x * 100).toFixed(0)}%` : "—";
}

export function SelfImprovement() {
  const [curve, setCurve] = useState<LearningCurve | null>(null);
  const [monitor, setMonitor] = useState<MonitorReport | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    loadLearningCurve().then(setCurve).catch(() => setErr("learning curve unavailable"));
    loadMonitor().then(setMonitor).catch(() => setErr("monitor unavailable"));
  }, []);

  const fixture =
    (monitor?.fixture ?? false) || curve?.dataSource === "synthetic";

  return (
    <section className="col panel self-improve">
      <div className="panel-title">
        <span>Self-improvement · early-warning model</span>
        {fixture && (
          <span
            className="prov-badge prov-fixture"
            title={
              monitor?.fixture_note ||
              curve?.honest_note ||
              "illustrative demo numbers — not real Kalos results"
            }
          >
            <span className="prov-dot" />
            illustrative
          </span>
        )}
      </div>
      <div className="body si-body">
        {err && !curve && !monitor && (
          <div className="empty" style={{ color: "var(--crit)" }}>{err}</div>
        )}
        {/* Honest scope label (bead 31n / 1b4): the incumbent is a per-GPU
            early-warning scorer applied fleet-wide — NOT a topology/cascade model.
            Until bead 4sz exposes parameters/contributions, only the feature set
            (card metadata) is shown, not learned coefficients. */}
        <div className="si-scope faint">
          per-GPU early-warning scorer, applied fleet-wide — not a topology/cascade model
          {monitor?.features?.length ? ` · features: ${monitor.features.join(", ")}` : ""}
        </div>
        <LearningCurveView curve={curve} />
        <MonitorView monitor={monitor} />
      </div>
    </section>
  );
}

// ---- learning curve: v0->vN held-out ROC-AUC, baseline at 0.5 ----------------
function LearningCurveView({ curve }: { curve: LearningCurve | null }) {
  if (!curve) return <div className="si-block empty">loading learning curve…</div>;
  if (!curve.available || !curve.curve?.length)
    return (
      <div className="si-block empty">
        {curve.reason ?? "no learning-curve artifact"}
      </div>
    );

  const pts = curve.curve;
  const W = 260;
  const H = 90;
  const PAD = 4;
  const aucs = pts.map((p) => p.roc_auc);
  const lo = Math.min(0.5, ...aucs) - 0.02;
  const hi = Math.max(...aucs) + 0.02;
  const span = hi - lo || 1;
  const x = (i: number) => PAD + (i / Math.max(1, pts.length - 1)) * (W - 2 * PAD);
  const y = (v: number) => H - PAD - ((v - lo) / span) * (H - 2 * PAD);
  const path = pts
    .map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.roc_auc).toFixed(1)}`)
    .join(" ");
  const baselineY = y(0.5).toFixed(1);
  const final = curve.final_incumbent;

  return (
    <div className="si-block">
      <div className="si-head">
        <span className="si-label">learning curve</span>
        <span className="faint">
          {curve.n_promotions ?? pts.length} promotions · {curve.primary_metric ?? "roc_auc"}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} className="si-curve">
        {/* no-skill baseline */}
        <line
          x1={PAD}
          x2={W - PAD}
          y1={baselineY}
          y2={baselineY}
          stroke="var(--dim, #555)"
          strokeDasharray="3 3"
          strokeWidth={1}
        />
        <text x={W - PAD} y={Number(baselineY) - 3} textAnchor="end" fontSize="8" fill="var(--dim,#777)">
          no-skill 0.50
        </text>
        <path d={path} fill="none" stroke="var(--ok, #7fa86b)" strokeWidth={1.5} />
        {pts.map((p: LearningCurvePoint, i) => (
          <g key={p.version}>
            <circle cx={x(i)} cy={y(p.roc_auc)} r={2.5} fill="var(--ok,#7fa86b)">
              <title>
                {p.version} · {p.label} · ROC-AUC {p.roc_auc.toFixed(3)} (gap +
                {p.signal_gap.toFixed(3)})
                {p.reflection ? `\n${p.reflection}` : ""}
              </title>
            </circle>
            <text x={x(i)} y={H - 0.5} textAnchor="middle" fontSize="7" fill="var(--dim,#888)">
              {p.version}
            </text>
          </g>
        ))}
      </svg>
      {final && (
        <div className="si-final faint">
          incumbent v{final.version} · {final.model_type} · AUC{" "}
          {final.roc_auc.toFixed(3)} · {final.n_features} feat
        </div>
      )}
    </div>
  );
}

// ---- monitor: per-horizon recall / miss at a chosen alert budget -------------
function MonitorView({ monitor }: { monitor: MonitorReport | null }) {
  const [budgetIdx, setBudgetIdx] = useState(0);
  if (!monitor) return <div className="si-block empty">loading monitor…</div>;
  if (!monitor.available)
    return (
      <div className="si-block empty">{monitor.reason ?? "monitor unavailable"}</div>
    );

  const budgets = monitor.budgets ?? [];
  const sel: BudgetReport | undefined = budgets[Math.min(budgetIdx, budgets.length - 1)];
  const horizons: HorizonResult[] = sel
    ? Object.values(sel.grid.by_horizon).sort((a, b) => a.horizon_s - b.horizon_s)
    : [];

  return (
    <div className="si-block">
      <div className="si-head">
        <span className="si-label">miss detector · v{monitor.model_version}</span>
        <span className="faint">
          {monitor.n_onsets} onsets · {monitor.n_rows} rows
        </span>
      </div>

      <div className="si-budget-tabs">
        {budgets.map((b, i) => (
          <button
            key={b.budget}
            className={`si-tab ${i === budgetIdx ? "sel" : ""}`}
            onClick={() => setBudgetIdx(i)}
            title={`alert budget ${pct(b.budget)} — threshold ${b.threshold.toFixed(3)}`}
          >
            {pct(b.budget)} budget
          </button>
        ))}
      </div>

      <table className="si-table">
        <thead>
          <tr>
            <th>horizon</th>
            <th>recall</th>
            <th>caught</th>
            <th>missed</th>
          </tr>
        </thead>
        <tbody>
          {horizons.map((h) => (
            <tr key={h.horizon_s}>
              <td className="mono-num">{Math.round(h.horizon_s)}s</td>
              <td className="mono-num">
                <RecallBar recall={h.recall} />
              </td>
              <td className="mono-num">{h.caught}</td>
              <td className="mono-num crit">{h.missed}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {sel && (
        <div className="si-final faint">
          alert rate {pct(sel.alert_rate)} · threshold {sel.threshold.toFixed(3)}
        </div>
      )}
    </div>
  );
}

function RecallBar({ recall }: { recall: number }) {
  const r = Number.isFinite(recall) ? recall : 0;
  return (
    <span className="recall-bar" title={`recall ${pct(recall)}`}>
      <span className="recall-fill" style={{ width: `${Math.round(r * 100)}%` }} />
      <span className="recall-val">{pct(recall)}</span>
    </span>
  );
}
