import gpusitter.agent.tools as tools
import gpusitter.detection.classifier as classifier
import gpusitter.detection.stream as stream


def _seed_history():
    stream.reset_history()
    classifier.reset()
    # 10 jobs, separable by power_max; later rows are the val split
    for i in range(10):
        fail = i % 2 == 1
        stream.HISTORY.append(
            {
                "job_id": i,
                "type": "train",
                "gpu_num": 8,
                "power_max": 300 if fail else 100,
                "state": "NODE_FAIL" if fail else "COMPLETED",
            }
        )


def test_get_sensory_returns_aggregates():
    _seed_history()
    s = tools.get_sensory(1)
    assert s == {"power_max": 300}


def test_train_and_validate_trains_and_promotes():
    _seed_history()
    out = tools.train_and_validate("logreg", ["power_max", "gpu_num"])
    assert out["trained"] is True
    assert out["promoted"] is True and out["version"] == 1
    assert out["val_auc"] == 1.0


def test_train_and_validate_guards_single_class():
    stream.reset_history()
    classifier.reset()
    for i in range(6):
        stream.HISTORY.append({"job_id": i, "gpu_num": 8, "power_max": 100, "state": "COMPLETED"})
    out = tools.train_and_validate("logreg", ["power_max", "gpu_num"])
    assert out["trained"] is False


def test_train_and_validate_not_promoted():
    """Retrain with a non-predictive feature must not beat the incumbent (no churn)."""
    _seed_history()
    # Establish incumbent with a perfectly separating feature
    first = tools.train_and_validate("logreg", ["power_max"])
    assert first["promoted"] is True and first["version"] == 1

    # gpu_num is constant (8 for every row) → model has zero signal → AUC ~0.5 < 1.0
    out = tools.train_and_validate("logreg", ["gpu_num"])
    assert out["trained"] is True
    assert out["promoted"] is False
    assert out["version"] == 1  # incumbent version unchanged — no churn


def test_train_and_validate_unknown_feature():
    _seed_history()
    out = tools.train_and_validate("logreg", ["does_not_exist"])
    assert out == {"trained": False, "reason": "unknown feature"}
