# early_detection demo fixture (bead jds)

**Illustrative / synthetic — NOT real Kalos numbers.**

`features.csv` is a small deterministic synthetic labeled early-detection
table (real-schema feature names) and `registry/` is a prebuilt
`ModelRegistry` (promoted logreg incumbent: model card + pickled estimator
+ manifest). They exist so the dashboard's `/api/monitor` surface renders
off the droplet, without the ~80 GB trace or `data/early_detection.parquet`.

Regenerate with `python scripts/build_monitor_fixture.py`.

The real held-out evaluation (weak-but-real linear signal, ROC-AUC
~0.64–0.65, NO-GO for a standalone predictor) is canonical in
`docs/early-detection-eval.md`. Do not cite the fixture as a result.
