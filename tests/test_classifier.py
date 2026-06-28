import pandas as pd
import backend.classifier as clf


def test_fit_and_auc_perfect_separation():
    clf.reset()
    Xtr = pd.DataFrame({"x": [0, 0, 1, 1]}); ytr = [0, 0, 1, 1]
    Xval = pd.DataFrame({"x": [0, 1]}); yval = [0, 1]
    est = clf.fit_candidate("logreg", ["x"], Xtr, ytr)
    assert clf.auc(est, Xval, yval) == 1.0


def test_promote_gate():
    clf.reset()
    assert clf.maybe_promote(None, "logreg", ["x"], 0.80) is True   # first -> baseline
    assert clf.INCUMBENT.version == 1 and clf.INCUMBENT.auc == 0.80
    assert clf.maybe_promote(None, "tree", ["x"], 0.75) is False    # worse, kept
    assert clf.INCUMBENT.version == 1
    assert clf.maybe_promote(None, "gboost", ["x"], 0.90) is True   # better -> v2
    assert clf.INCUMBENT.version == 2 and clf.INCUMBENT.model_type == "gboost"
