# GPUSitter Hackathon Plan & Notes

## Project Overview
... (full previous content)

**Current Status (after Iteration 5)**: Full multi-stage system prototyped.
- Architecture docs complete.
- Classifier, main_loop, enhanced simulator, self_improver skeletons.
- AGENTS.md, SKILL.md, README polished for multi-stage.

## Iteration Log (Complete)
- Iteration 1: Multi-stage architecture (PLAN + ARCHITECTURE.md).
- Iteration 2: classifier.py implementation.
- Iteration 3: main_loop.py + demo entrypoint.
- Iteration 4: Enhanced simulator.py (DCGM metrics, env modes) + AGENTS.md.
- **Iteration 5 (done)**: self_improver.py skeleton + SKILL.md updates + README polish.

**Repo now in strong state**: Ready for Antigravity integration, live demo scripting, and hackathon submission. Next steps in real dev: wire Antigravity env_id for persistent self-updates, add more failure scenarios, export traces for judging.

Milestones all addressed. Strong technical foundation with clear self-improvement narrative.

Update as needed post-hackathon.

## Sponsor Tools & Infrastructure Integration
- **DigitalOcean Spaces (Storage)**: An S3-compatible Spaces bucket (`https://gpu-cluster-trace-datasets.sfo3.digitaloceanspaces.com/`) is available to host our trace datasets.
- **DigitalOcean Droplet (Compute)**: Runs the GPUSitter active services (monitoring loop, classifier, and self-improver).
  - Droplet Name: `ubuntu-s-2vcpu-4gb-120gb-intel-sfo3-aie-hack` (IP: `134.199.208.214`).
  - Storage/Memory Resolution: We download trace datasets directly to the droplet's 120 GB local disk. To resolve VM Out-Of-Memory (OOM) failures during Git LFS operations (cloning `acme-util`), we created and enabled a **16 GB swap file** on the VM.
- **Orchestration Layer**: Orchestrated via **Antigravity / Gemini 3.5 Flash** for intelligent monitoring, service deployment, and dataset management on the Droplet.