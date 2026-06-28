import json, os
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

_MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "tree": lambda: DecisionTreeClassifier(random_state=0),
    "gboost": lambda: GradientBoostingClassifier(random_state=0),
}


@dataclass
class Model:
    estimator: object
    model_type: str
    features: list
    auc: float
    version: int
    n_samples: int = 0


INCUMBENT = None


def reset():
    global INCUMBENT
    INCUMBENT = None


def fit_candidate(model_type, features, Xtr, ytr):
    est = _MODELS[model_type]()
    est.fit(Xtr, ytr)
    return est


def auc(estimator, Xval, yval):
    proba = estimator.predict_proba(Xval)[:, 1]
    return float(roc_auc_score(yval, proba))

auc_from_lists = auc  # alias — both work on plain lists now


def maybe_promote(estimator, model_type, features, val_auc, n_samples=0):
    global INCUMBENT
    if INCUMBENT is None or val_auc > INCUMBENT.auc:
        version = 1 if INCUMBENT is None else INCUMBENT.version + 1
        INCUMBENT = Model(estimator, model_type, features, val_auc, version, n_samples)
        return True
    return False


def save_state(path: str) -> None:
    if INCUMBENT is None:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "version": INCUMBENT.version,
            "model_type": INCUMBENT.model_type,
            "features": INCUMBENT.features,
            "val_auc": round(INCUMBENT.auc, 4),
            "n_samples": INCUMBENT.n_samples,
        }, f, indent=2)


def load_state(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
