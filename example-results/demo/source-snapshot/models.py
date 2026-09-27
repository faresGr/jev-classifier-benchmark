"""Small validation grids, trained without looking at the test set."""
from time import perf_counter
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


class AdaptiveSVD(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=128, random_state=0):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        n = max(1, min(self.n_components, X.shape[0] - 1, X.shape[1] - 1))
        self.reducer_ = TruncatedSVD(n_components=n, random_state=self.random_state)
        self.reducer_.fit(X)
        return self

    def transform(self, X):
        return self.reducer_.transform(X)


def candidates(name, config, seed):
    def features():
        return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                              max_features=config["max_features"], dtype=np.float32)
    if name == "dummy":
        return [(Pipeline([("tfidf", features()), ("classifier", DummyClassifier(strategy="prior"))]), {})]
    result = []
    values = [0.1, 1.0, 10.0] if name in {"logreg", "linear_svm", "naive_bayes"} else [3, 6]
    for value in values:
        steps = [("tfidf", features())]
        if name == "logreg":
            clf = LogisticRegression(C=value, max_iter=1500, random_state=seed)
            params = {"C": value}
        elif name == "naive_bayes":
            clf = MultinomialNB(alpha=value)
            params = {"alpha": value}
        elif name == "linear_svm":
            # The whole text pipeline is inside each calibration fold: no vocabulary leakage.
            base = Pipeline(steps + [("classifier", LinearSVC(C=value, random_state=seed))])
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=1)
            result.append((clf, {"C": value, "calibration": "sigmoid_3fold_training_only"}))
            continue
        elif name in {"xgboost_tfidf", "xgboost_svd"}:
            from xgboost import XGBClassifier
            if name == "xgboost_svd":
                steps.append(("svd", AdaptiveSVD(config["svd_components"], seed)))
            clf = XGBClassifier(n_estimators=config["xgb_estimators"], max_depth=value,
                learning_rate=0.08, tree_method="hist", subsample=0.9, colsample_bytree=0.9,
                n_jobs=config["threads"], random_state=seed, eval_metric="mlogloss")
            params = {"max_depth": value, "n_estimators": config["xgb_estimators"]}
        else:
            raise ValueError(f"Unknown model: {name}")
        result.append((Pipeline(steps + [("classifier", clf)]), params))
    return result


def fit_best(name, train, validation, labels, config, seed):
    mapping = {label: i for i, label in enumerate(labels)}
    X, y = [r["text"] for r in train], [mapping[r["label"]] for r in train]
    vx, vy = [r["text"] for r in validation], [mapping[r["label"]] for r in validation]
    start = perf_counter()
    best, score, best_params, trials = None, -1, None, []
    for model, params in candidates(name, config, seed):
        model.fit(X, y)
        current = float(f1_score(vy, model.predict(vx), labels=list(range(len(labels))), average="macro", zero_division=0))
        trials.append({"parameters": params, "validation_macro_f1": current})
        if current > score:
            best, score, best_params = model, current, params
    return best, {"tuning_and_fit_seconds": perf_counter() - start,
                  "selected_parameters": best_params, "validation_trials": trials}


def predict_local(model, rows, labels):
    records = []
    # Warm up once; benchmark sequential, single-record inference including vectorization.
    model.predict_proba([rows[0]["text"]])
    for row in rows:
        start = perf_counter()
        raw = model.predict_proba([row["text"]])[0]
        p = np.zeros(len(labels))
        p[np.asarray(model.classes_, dtype=int)] = raw
        p /= p.sum()  # Float32 estimators can accumulate rounding error.
        elapsed = perf_counter() - start
        records.append({"id": row["id"], "split": row["split"], "true_label": row["label"],
            "predicted_label": labels[int(p.argmax())], "probabilities": p.tolist(),
            "confidence": None, "latency_seconds": elapsed, "attempts": 1, "error": None})
    return records
