// Dashboard data contract. The static fixtures under /public/fixtures match these
// shapes exactly; when the FastAPI backend lands it should serve the same JSON.

export type Severity = "crit" | "warn";
export type CellStatus = "fault" | "active" | "idle";

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
}

export interface Meta {
  window: [string, string];
  cascadeTs: string;
  totalGpus: number;
  faulted: number;
  nodesAffected: number;
  source: string;
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
export interface ModelResponse {
  model: ModelCard | null;
  source?: "registry" | "in_process";
  rigorous?: boolean;
  note?: string;
  message?: string;
}

// Self-improving predictor learning curve. Mirrors the JSON served by
// GET /api/learning-curve (a verbatim copy of docs/learning_curve.json).
export interface LearningCurvePoint {
  version: string; // "v0".."v3"
  label: string;
  model_type: string;
  n_features: number;
  roc_auc: number;
  signal_gap: number;
  hypothesis: string;
  reflection: string;
}

export interface LearningCurveRound {
  round: number;
  model_type: string;
  n_features: number;
  features?: string[];
  roc_auc: number;
  signal_gap: number;
  leaks: boolean;
  promoted: boolean;
  version: number | string | null;
  hypothesis: string;
  reflection: string;
}

export interface LearningCurveData {
  primary_metric: string;
  dataset: {
    source: string;
    synthetic: boolean;
    n_rows: number;
    n_features: number;
    n_positive: number;
    base_rate: number;
  };
  baseline_v0: {
    name: string;
    roc_auc: number;
    avg_precision: number;
    base_rate: number;
    n_train: number;
    n_test: number;
  };
  curve: LearningCurvePoint[];
  rounds: LearningCurveRound[];
  n_promotions: number;
  final_incumbent: {
    version: number | string;
    model_type: string;
    roc_auc: number;
    n_features: number;
    features: string[];
  };
  honest_note: string;
  real_data_reference: {
    source: string;
    dataset?: Record<string, unknown>;
    best_real: {
      model: string;
      horizon: string;
      roc_auc: number;
      permuted_baseline: number;
    };
    best_hgb_roc_auc: number;
    verdict: string;
  };
}

// POST /api/predict-gpu — per-GPU failure-likelihood prediction for the selected
// incident's telemetry window. Either an available prediction or a reason it isn't.
export type PredictGpu =
  | {
      available: true;
      likelihood: number;
      threshold: number;
      label: "alert" | "watch" | "ok";
      features: Record<string, number>;
      model: {
        version: number;
        model_type: string;
        val_auc: number;
        fixture: boolean;
      };
      note: string;
    }
  | { available: false; reason: string };

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
