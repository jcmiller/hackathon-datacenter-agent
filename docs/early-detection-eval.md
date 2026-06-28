# Early-detection dataset — evaluation & go/no-go verdict

**Bead:** aieng26hack-5fq.lys · **Updated:** 2026-06-28 · **Data:** real Kalos
telemetry (droplet `134.199.208.214`, `data/acme-util` LFS cache). Numbers
regenerated under the canonical `uv` env (scikit-learn 1.9.0).

Honest answer to the bead's central question — *does Kalos telemetry carry
usable pre-Xid warning signal?* **A weak but real, leakage-free linear signal
(held-out ROC-AUC ~0.64–0.65) — not enough for a standalone reliable predictor
(NO-GO), but the dataset + harness are the reusable deliverable (GO).**

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
  1,671 neg.

The cached table is `data/early_detection.csv` on the droplet (gitignored here;
datasets are never committed). Build is run-once; experiments read the cache.

## Held-out metrics — strict time-ordered split

Split: a single `t_ref` threshold per horizon — earliest ~70% by time -> train,
the rest -> test, with **`max(train.t_ref) < min(test.t_ref)`**. Every training
point is strictly in the past of every test point, so the model can never see
future telemetry. (The first submission bucketed by GPU and filled train by row
count, which let a train GPU's later rows post-date test rows — *not* actually
time-ordered; fixed, with a regression test in `tests/test_eval_early_dataset.py`.)
Per horizon: train 1,578 (522 pos) / test 748 (133 pos, base rate **0.178**). The
no-signal baseline is the same model retrained on shuffled labels, **averaged
over 8 shuffles** (a single shuffle swung 0.39–0.58 on this small test set).

| Horizon | model | n_test | pos_test | ROC-AUC | AP | permuted-AUC (×8) | recall@5% | prec@5% | recall@10% | prec@10% |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 60s  | logreg | 748 | 133 | **0.650** | 0.235 | 0.504 | 0.030 | 0.108 | 0.083 | 0.147 |
| 60s  | hgb    | 748 | 133 | 0.587 | 0.259 | 0.521 | 0.090 | 0.324 | 0.173 | 0.307 |
| 300s | logreg | 748 | 133 | **0.653** | 0.251 | 0.540 | 0.045 | 0.162 | 0.128 | 0.227 |
| 300s | hgb    | 748 | 133 | 0.579 | 0.228 | 0.492 | 0.075 | 0.270 | 0.135 | 0.240 |
| 600s | logreg | 748 | 133 | **0.641** | 0.240 | 0.490 | 0.045 | 0.162 | 0.098 | 0.173 |
| 600s | hgb    | 748 | 133 | 0.560 | 0.222 | 0.477 | 0.083 | 0.297 | 0.158 | 0.280 |

## Reading the numbers

1. **A real, weak, *linear* signal.** Logistic regression holds **ROC-AUC
   0.64–0.65 at every horizon**, a consistent **+0.11 to +0.15 over the averaged
   no-signal baseline (~0.50)**. It is small (0.5 = coin flip, ~0.8 = the usual
   "ship it" floor) but it is genuine and reproducible, not noise.
2. **The signal is leakage-free.** Logreg honest **0.65** ≈ logreg on the *leaky*
   stratified-random split **0.673** — almost no inflation. So the linear signal
   survives a strict temporal split; it is foresight, not pos/neg pairing.
3. **HGB's apparent power is mostly leakage.** HistGradientBoosting scores
   **0.873 leaky** but only **0.56–0.59 honest** — ~0.30 AUC evaporates under the
   time split. Under temporal covariate shift the boosted trees overfit the
   train-period distribution; the durable signal is the linear part.
4. **Ranking is only modestly better than the prior.** AP 0.22–0.26 vs a 0.178
   base rate (~1.3–1.5×). At a 10% alert budget, HGB catches ~16–17% of faults at
   ~28–31% precision (~1.6–1.7× base rate); logreg is higher-AUC but flatter at
   the top of the ranking. Useful as a weak prior, not a precise alarm.
5. **Flat across horizons.** AUC does not sharpen from 600s -> 60s; there is no
   accelerating precursor in these features — the weak separability is roughly
   static in the 10 minutes before onset, consistent with the 6xk finding
   (GPU_TEMP/POWER median delta ≈ 0 pre-onset).

## Verdict

**NO-GO for a standalone, reliable pre-Xid predictor** on these windowed
DCGM-utilization features. The best honest AUC (~0.65, logreg) is well below a
deployable bar, alert-budget precision is only ~1.5–1.7× the base rate, and the
non-linear model barely beats chance out-of-time. Treating top-scored GPUs as
imminent-failure alarms would fire mostly at the base rate — not trustworthy on
its own.

**But the signal is weak-yet-real and leakage-free**, which is a more useful
finding than "no signal": the linear model extracts ~0.65 AUC that survives a
strict temporal split. That makes it a plausible *contributing input* to an
ensemble (alongside RCA / job-context signals) and a baseline to beat by adding
the metrics excluded here — SM_ACTIVE via a pod/IP alias map, MEM_CLOCK,
ECC/retired-page counters if present — widening the lookback, and joining
NODE_FAIL / `fail_time` events. Those are out of lys scope.

**GO for the dataset + harness as infrastructure.** The cache-safe streaming
builder and the strict-time-ordered, permutation-controlled evaluator run
end-to-end on the real ~76 GB Kalos trace and produce reproducible, honestly
measured numbers. That pipeline — not a claim of strong predictive power — is the
deliverable the self-improving classifier chain (glf→rnh→8co) builds on.

## Reproduce

```
# dataset (run-once on the droplet; ~22 min — reader is O(rows*all_cols),
# see follow-up aieng26hack-fo1):
PYTHONPATH=src python scripts/build_early_dataset.py --repo-dir data/acme-util \
    --metrics GPU_TEMP POWER_USAGE GPU_UTIL MEMORY_TEMP \
    --horizons 60 300 600 --lookback 600 --gpu-batch-size 150 \
    --control-gpus <ids> --out data/early_detection.parquet

# evaluation (reads the cached table; seconds; run under uv for canonical env):
uv run python scripts/eval_early_dataset.py --data data/early_detection.csv \
    --out-json data/early_detection_eval.json --out-md docs/early-detection-eval.md
```

Raw per-horizon/per-model metrics: `docs/early_detection_eval.json`.

## Portable demo fixture (bead jds)

The numbers above are the canonical real result. For a demo that renders the
monitor surface **off the droplet** (no ~80 GB trace, no
`data/early_detection.parquet`), a small **synthetic, illustrative** fixture +
prebuilt registry ships in-tree at
`src/gpusitter/app/fixtures/early_detection/` and is served by `/api/monitor`
when the real artifacts are absent (payload marked `fixture:true`). Its numbers
are **not** a Kalos result — they exist only to exercise the per-row scoring,
alert-budget, and horizon-grid miss-detector shapes. Regenerate with
`python scripts/build_monitor_fixture.py`. The real held-out evaluation here
stays the single source of truth.
