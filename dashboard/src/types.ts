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

export type FeedbackOutcome = "confirmed" | "false_alarm" | "uncertain";

export interface ModelCard {
  version: number;
  model_type: string;
  features: string[];
  val_auc: number;
  n_samples: number;
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
