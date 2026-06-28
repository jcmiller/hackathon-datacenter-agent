// Dashboard data contract. The static fixtures under /public/fixtures match these
// shapes exactly; when the FastAPI backend lands it should serve the same JSON.

export type Severity = "crit" | "warn";
export type CellStatus = "fault" | "active" | "idle";

// Honesty contract (bead h7w/31n). Every dashboard surface reports where its data
// came from so illustrative fixtures can never be mistaken for live telemetry.
// "real_substrate" = derived-real Kalos onsets; "fixture"/"synthetic" = explicit
// demo data; "unavailable" = no honest data to serve; "offline" = the static
// fixture fallback used when the API itself is unreachable.
export type DataSource =
  | "real_substrate"
  | "fixture"
  | "synthetic"
  | "trace"
  | "unavailable"
  | "offline";

export interface Provenance {
  kind?: string; // "real" | "fixture" | "trace" | ...
  telemetryKind?: string;
  source?: string;
  note?: string;
  fixture_note?: string;
  [k: string]: unknown;
}

// A resolved dashboard data source + its honesty badge, threaded through App state.
export interface SourceBadge {
  dataSource: DataSource;
  provenance: Provenance | null;
}

export interface Incident {
  id: string;
  ts: string; // ISO datetime
  gpu: { node: string; idx: number };
  xid: number;
  xidLabel: string;
  severity: Severity;
  nodeCofaults: number; // other GPUs on the same node that also faulted
  correlatedCount: number; // cluster-wide co-faults in the same window
  correlated: string[]; // sample of correlated GPU ids
  hero: boolean;
  state: string;
}

export type Point = [string, number | null]; // [ISO ts, value]

export interface TelemetryWindow {
  gpu: string;
  centerTs: string;
  series: { temp: Point[]; power: Point[]; util: Point[] };
  dataSource?: DataSource;
  provenance?: Provenance;
}

export interface FleetCell {
  node: string;
  idx: number;
  temp: number | null;
  util: number | null;
  xid: number;
  status: CellStatus;
}

export interface Fleet {
  ts: string;
  nodes: number;
  faulted: number;
  cells: FleetCell[];
  dataSource?: DataSource;
  provenance?: Provenance;
}

export interface Meta {
  window: [string, string];
  cascadeTs: string;
  totalGpus: number;
  faulted: number;
  nodesAffected: number;
  source: string;
  dataSource?: DataSource;
  provenance?: Provenance;
}

export interface ModelCard {
  version: number;
  model_type: string;
  features: string[];
  val_auc: number | null;
  n_samples: number;
  // Provenance + rigorous metrics, present when /api/model serves the canonical
  // keep-if-better registry incumbent (bead aow). The same model /api/monitor scores:
  // model.version === monitor.model_version. Absent on the provisional in-process card.
  primary_metric?: string;
  primary_value?: number;
  n_train?: number;
  n_test?: number;
  holdout_id?: string;
  training_window?: [string, string];
}

// /api/model envelope. `source` distinguishes the canonical registry incumbent from
// the provisional in-process triage fit; `rigorous` is true only for the former.
// `fixture` mirrors /api/monitor: true when the model card is the committed
// off-droplet demo incumbent (numbers illustrative, not real Kalos results).
export interface ModelResponse {
  model: ModelCard | null;
  source?: "registry" | "in_process";
  rigorous?: boolean;
  fixture?: boolean;
  fixture_note?: string;
  note?: string;
  message?: string;
}

// /api/monitor — per-row risk scores + per-horizon miss/recall from the incumbent
// (bead i6k). This is the operational self-improvement surface: the same model
// /api/model describes, scored over the labeled feature table.
export interface MonitorRow {
  gpu: string;
  t_ref: string;
  horizon_s: number;
  label: number;
  risk_score: number;
  alert_flag: boolean;
  model_version: number;
  features: Record<string, number>;
}

export interface HorizonResult {
  horizon_s: number;
  n_onsets: number;
  caught: number;
  missed: number;
  recall: number; // caught / total onsets (NaN serialized as null)
  misses: unknown[];
}

export interface BudgetReport {
  budget: number;
  threshold: number;
  alert_rate: number;
  grid: {
    model_version: number;
    budget: number;
    alert_rate: number;
    by_horizon: Record<string, HorizonResult>;
  };
}

export interface MonitorReport {
  available: boolean;
  reason?: string;
  model_version?: number;
  features?: string[];
  n_rows?: number;
  n_onsets?: number;
  budgets?: BudgetReport[];
  rows?: MonitorRow[];
  fixture?: boolean;
  fixture_note?: string;
}

// /api/learning-curve — the v0->vN keep-if-better history (self-improvement visual).
export interface LearningCurvePoint {
  version: string;
  label: string;
  model_type: string;
  n_features: number;
  roc_auc: number;
  signal_gap: number;
  hypothesis?: string;
  reflection?: string;
}

export interface LearningCurve {
  available: boolean;
  reason?: string;
  dataSource?: DataSource;
  primary_metric?: string;
  curve?: LearningCurvePoint[];
  n_promotions?: number;
  final_incumbent?: {
    version: number;
    model_type: string;
    roc_auc: number;
    n_features: number;
    features: string[];
  };
  honest_note?: string;
  real_data_reference?: Record<string, unknown>;
}

// Agent reasoning stream events
export type AgentEvent =
  | { type: "user"; text: string }
  | { type: "tool_call"; tool: string; args: string }
  | { type: "observation"; text: string }
  | { type: "file_update"; path: string; entry: Record<string, unknown> }
  | {
      type: "disposition";
      disposition: string;
      summary: string;
      action: string;
      ticket: string | null;
    };
