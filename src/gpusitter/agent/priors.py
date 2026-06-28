# backend/priors.py
# Co-occurrence priors distilled from Meta's 2024 LLM-fleet reliability reporting.
# Loaded into the agent's instruction so it reasons about cause from symptoms.
DOMAIN_PRIORS = """\
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
