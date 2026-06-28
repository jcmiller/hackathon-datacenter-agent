# Learning-curve demo — the self-improving classifier harness

**Bead:** aieng26hack-5fq.8co · **Builds on:** glf (eval harness + registry), rnh
(agent-authored loop), lys (labeled dataset + eval). **Artifact:**
[`learning_curve.json`](./learning_curve.json) (regenerate with the command below).

This is the headline of the 5fq epic: a classifier that **rewrites itself and gets
better on held-out data**, with every improvement gated by an immutable judge. The
curve starts at an honest no-skill floor (v0) and climbs as the agent's
keep-if-better promotions (v1..vN) beat it on the *same* strict time-ordered split.

## The curve (deterministic synthetic demo)

| version | model | features | held-out ROC-AUC | what the agent did |
|--:|:--|--:|--:|:--|
| **v0** | no-skill (base rate) | 0 | **0.500** | the floor — predict the base rate, zero learned signal |
| **v1** | hgb | 4 | **0.625** | baseline over all features establishes the incumbent |
| **v2** | logreg | 4 | **0.682** | switches model form — linear beats boosted on this signal |
| **v3** | logreg | 2 | **0.684** | keeps `power_mean`, `temp_mean`; drops `util_mean`, `mem_last` as noise |

Rejected along the way (keep-if-better refused them — the gate is not theatre):

| round | candidate | ROC-AUC | outcome |
|--:|:--|--:|:--|
| 4 | hgb / 2 feat | 0.606 | rejected (< incumbent 0.684) |
| 5 | logreg / 1 feat | 0.659 | rejected (ablation too aggressive) |
| 6 | hgb / 1 feat | 0.552 | rejected |

Everything above the v0 line of 0.500 is, by exactly that margin, *learned skill* —
the floor is a constant-score model whose ROC-AUC is 0.5 by construction. The full
per-round HYPOTHESIS + REFLECTION (the agent's version diffs / reasoning) is in
`learning_curve.json`.

## Why these numbers are honest

The signal is **deliberately weak** (~0.6–0.68), not a staged 0.95. That mirrors
the real Kalos finding from the lys evaluation
([`early_detection_eval.json`](./early_detection_eval.json),
[`early-detection-eval.md`](./early-detection-eval.md)), embedded in the artifact's
`real_data_reference` block:

- **Best real held-out signal:** logreg @ 300s horizon, ROC-AUC **0.653**, against a
  no-signal permutation baseline of **0.540** — weak but real and leakage-free.
- **Boosted models do worse on real data:** best hgb **0.587**. The synthetic demo
  *independently reproduces this* — the agent promotes hgb first (v1) then discovers
  linear is better and switches to logreg (v2). The mechanism rediscovers a true
  property of the data, it does not paper over it.
- **Verdict (lys):** not a standalone reliable predictor (**NO-GO**); the dataset +
  self-improving harness are the reusable deliverable (**GO**). The
  loop/mechanism is the story, not a headline AUC.

The real `data/early_detection.csv` (6,978 rows, 40 features, 685 GPUs) is gitignored
— datasets are never committed — so the *committed* artifact is generated on the
weak-signal synthetic table for reproducibility. Run on the real data anywhere it
exists (e.g. the droplet) with `--data`.

## How it works (separation of powers)

1. **v0 baseline** (`gpusitter.detection.baseline.evaluate_baseline`) — a no-skill
   `DummyClassifier(prior)` scored on the harness's strict time-ordered split. Same
   holdout as every other point, so the curve is apples-to-apples.
2. **The loop** (`gpusitter.detection.agent_loop.run_loop`) — the agent authors a
   typed candidate (model form + feature subset), the **immutable** harness
   (`gpusitter.detection.harness`) trains it on the past, scores it on the strictly
   future held-out set, runs leakage probes, and applies keep-if-better. The agent
   reflects on the metrics and revises. It cannot edit the judge, so it cannot cheat
   the gate — it can only find real signal.
3. **The curve** = the registry's promotion history. Strictly increasing by
   construction (keep-if-better), so the line only ever climbs or holds.

## Reproduce

```bash
uv run python scripts/learning_curve_demo.py            # synthetic, writes docs/learning_curve.json
uv run python scripts/learning_curve_demo.py --data data/early_detection.csv   # real Kalos data
```
