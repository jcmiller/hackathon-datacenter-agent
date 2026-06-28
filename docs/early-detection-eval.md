# Early-detection dataset — evaluation & go/no-go verdict

**Bead:** aieng26hack-5fq.lys · **Date:** 2026-06-28 · **Data:** real Kalos
telemetry (droplet `134.199.208.214`, `data/acme-util` LFS cache).

This is the honest answer to the bead's central question: *does Kalos telemetry
carry usable pre-Xid warning signal?* Short version: **only weak, marginally
above-chance signal — NO-GO for a standalone predictor, but the dataset +
harness work end-to-end and are the reusable deliverable.**

## How it was built

```
PYTHONPATH=src python scripts/build_early_dataset.py \
    --repo-dir data/acme-util \
    --metrics GPU_TEMP POWER_USAGE GPU_UTIL MEMORY_TEMP \
    --horizons 60 300 600 --lookback 600 \
    --gpu-batch-size 150 \
    --control-gpus <30 non-onset GPUs> \
    --out data/early_detection.parquet      # -> early_detection.csv (no pyarrow)
```

- **Onsets:** 852 observed Xid onsets / 851 distinct GPUs, 2023-08-15..30 (dense
  Aug 15-18), via the empty-aware edge detector (idle/0 -> nonzero; first-row
  nonzero excluded as left-censored).
- **Rows:** one per `(gpu, t_ref)` prediction point. Positives at `t_event - H`;
  negatives = same-GPU pre-event controls (`t_event - 3600s`) **and** 30
  time-matched cluster-control GPUs at every positive `t_ref`. Leakage guard
  drops any negative whose horizon contains a real onset.
- **Features (40):** for each of GPU_TEMP / POWER_USAGE / GPU_UTIL / MEMORY_TEMP,
  the windowed `count, coverage, present, mean, std, min, max, last, delta,
  slope` over `[t_ref-600s, t_ref]`, strictly before `t_ref`. Missingness is
  explicit (`present`/`coverage`); never zero-filled. Rows with no telemetry
  coverage in any metric are dropped.
- **Result:** **6,978 rows · 1,965 positive · 685 GPUs.** Per horizon: 655 pos /
  1,671 neg. (Most cluster-control candidates fell on idle windows and were
  dropped by the coverage filter, leaving a near-balanced same-GPU-dominated
  set.)

The cached table is `data/early_detection.csv` on the droplet (gitignored here;
datasets are never committed). Rebuild is run-once; experiments read the cache.

## Held-out metrics — time-ordered, GPU-grouped split

Split: GPUs ordered by earliest `t_ref`; earliest 70% (by rows) -> train, rest
-> test. **GPU-atomic**, so a positive and its paired same-GPU negative can never
straddle the split — the leak that inflates naive AUC. Per horizon, recall is the
fraction of real faults catchable with that lead time at the alert budget.

| Horizon | model | n_test | pos_test | ROC-AUC | AP | permuted-AUC | recall@5% | prec@5% | recall@10% | prec@10% |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 60s  | logreg | 697 | 319 | 0.553 | 0.483 | 0.470 | 0.047 | 0.429 | 0.107 | 0.486 |
| 60s  | hgb    | 697 | 319 | **0.573** | 0.529 | 0.474 | 0.056 | 0.514 | 0.122 | 0.557 |
| 300s | logreg | 697 | 319 | 0.549 | 0.474 | 0.475 | 0.041 | 0.371 | 0.094 | 0.429 |
| 300s | hgb    | 697 | 319 | **0.573** | 0.507 | 0.516 | 0.047 | 0.429 | 0.125 | 0.571 |
| 600s | logreg | 697 | 319 | 0.556 | 0.492 | 0.508 | 0.053 | 0.486 | 0.103 | 0.471 |
| 600s | hgb    | 697 | 319 | 0.558 | 0.489 | 0.507 | 0.041 | 0.371 | 0.100 | 0.457 |

Test base rate ≈ 0.458, so AP ≈ 0.47–0.53 is **at or barely above the prior** —
the ranking is close to random. Alert-budget precision at 5% (0.37–0.51) ≈ base
rate: flagging the top-5%-scored points catches faults no better than chance.

## Reading the numbers

1. **Honest AUC is 0.55–0.57 everywhere.** Both a linear (logreg) and a
   non-linear (HistGradientBoosting) model land in the same narrow band. That is
   weak: 0.5 is a coin flip, ~0.7 is the usual "worth shipping" floor.
2. **Lift over the no-signal control is +0.04 to +0.10 AUC.** The permutation
   baseline (same model, shuffled train labels) sits at 0.47–0.52. The model
   beats its own no-signal control only slightly — and at 300s/600s logreg is
   essentially tied with it. The signal is real but small.
3. **Flat across horizons.** AUC does not improve at any lead time (60s ≈ 300s ≈
   600s). There is no developing precursor that sharpens as the fault nears; the
   weak separability is horizon-insensitive, consistent with small static
   differences rather than a building warning.
4. **The leakage demonstration (headline).** The *leaky* stratified-random split
   — where a GPU's positive and its pre-event negative can land on opposite sides
   — scores **HGB AUC 0.873 / logreg 0.673**. Against the honest **0.57**, that
   is ~0.30 AUC of apparent "predictive power" that is **pairing leakage, not
   foresight.** This is why the GPU-grouped split is mandatory and why quick
   earlier baselines (~0.61 on a tiny random test) overstated the case.

This corroborates the independent 6xk characterization (GPU_TEMP/POWER median
delta ≈ 0, rising≈falling in the 5 min pre-onset) and the lys design's note that
the first droplet run showed weak/noisy lift.

## Verdict

**NO-GO for a standalone, reliable pre-Xid predictor** from these windowed
DCGM-utilization features. Under honest, leakage-free evaluation the signal is
weak (ROC-AUC ~0.55–0.57, only marginally above a no-signal baseline) and does
not strengthen toward the event. Treating top-scored GPUs as imminent-failure
alerts would fire at roughly the base rate — not actionable on its own.

**GO for the dataset + harness as infrastructure.** The cache-safe streaming
builder and the leakage-aware, time-ordered, permutation-controlled evaluator
run end-to-end on the real ~76 GB Kalos trace and produce reproducible,
honestly-measured numbers. That pipeline — not a claim of predictive power — is
the deliverable, and it is what the downstream self-improving classifier
(glf→rnh→8co) builds on. Plausible paths to a stronger signal (out of lys scope):
add metrics excluded here (SM_ACTIVE via a pod/IP alias map, MEM_CLOCK,
ECC/retired-page counters if present), widen the lookback, and join NODE_FAIL /
`fail_time` events — but on the current feature set the warning signal is weak,
measured honestly.

## Reproduce

```
# dataset (run-once on the droplet; ~22 min — reader is O(rows*all_cols),
# see follow-up aieng26hack-fo1):
PYTHONPATH=src python scripts/build_early_dataset.py --repo-dir data/acme-util \
    --metrics GPU_TEMP POWER_USAGE GPU_UTIL MEMORY_TEMP \
    --horizons 60 300 600 --lookback 600 --gpu-batch-size 150 \
    --control-gpus <ids> --out data/early_detection.parquet

# evaluation (reads the cached table; seconds):
PYTHONPATH=src python scripts/eval_early_dataset.py --data data/early_detection.csv \
    --out-json data/early_detection_eval.json --out-md docs/early-detection-eval.md
```
