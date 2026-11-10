from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def _logreg(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500, random_state=seed))


def _linear_svm(seed):
    return make_pipeline(StandardScaler(), LinearSVC(C=0.01, dual="auto", max_iter=5000, random_state=seed))


MODELS = {"logreg": _logreg, "linear_svm": _linear_svm}


def make_model(name, seed):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}, choose from {sorted(MODELS)}")
    return MODELS[name](seed)
