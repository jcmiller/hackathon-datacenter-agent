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


def maybe_promote(estimator, model_type, features, val_auc):
    global INCUMBENT
    if INCUMBENT is None or val_auc > INCUMBENT.auc:
        version = 1 if INCUMBENT is None else INCUMBENT.version + 1
        INCUMBENT = Model(estimator, model_type, features, val_auc, version)
        return True
    return False
