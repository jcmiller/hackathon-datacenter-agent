# backend/priors.py
# Co-occurrence priors distilled from Meta's 2024 LLM-fleet reliability reporting.
# Loaded into the agent's instruction so it reasons about cause from symptoms.
DOMAIN_PRIORS = """\
Incident model (canonical): an incident is an empty-aware per-GPU Xid ONSET — a
non-fault -> fault transition in XID_ERRORS (observed, edge-detected; NOT a latched
repeated code). Operationally these onsets arrive as i6k MISSES: a real onset the
incumbent early-detection predictor failed to alert on within its horizon. The miss
is the early-warning gap that triage closes. Xid telemetry IS available
(XID_ERRORS.csv, real per-GPU codes); the onset cohort + observed code are reachable
via find_correlated_failures(source="xid").

GPU fleet failure domain knowledge (priors, not ground truth):
- NODE_FAIL usually means the scheduler lost the node: hardware fault, not user error.
- High sustained GPU power + thermal followed by a drop often precedes Xid 79
  (GPU fell off the bus / PCIe link loss).
- Repeated ECC/memory errors (Xid 48/63/64/94/95) point to a degrading GPU; the fix
  is usually drain + replace, not a job restart.
- Many nodes failing in the same short window with the same job type suggests a shared
  cause: a bad job image, a network/NCCL fault, or a shared power/cooling domain.
- A single isolated NODE_FAIL with normal neighbours suggests a single-node hardware fault.
- Disposition guide: shared-cause cluster -> escalate to datacenter ops; isolated
  hardware fault -> page technician to drain+replace; transient with healthy telemetry
  -> restart the job and watch.
"""
