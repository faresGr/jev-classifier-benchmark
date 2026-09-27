import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, recall_score


def probability_diagnostics(y, p):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    bins = []
    for i in range(10):
        mask = (confidence >= i / 10) & ((confidence < (i + 1) / 10) if i < 9 else (confidence <= 1))
        if mask.any():
            bins.append({"count": int(mask.sum()), "confidence": float(confidence[mask].mean()),
                         "accuracy": float(correct[mask].mean())})
    # Threshold at unique scores so ties cannot create an unrealizable acceptance policy.
    order = np.argsort(-confidence, kind="stable")
    boundaries = np.r_[np.flatnonzero(np.diff(confidence[order]) != 0), len(y) - 1]
    cumulative_errors = np.cumsum(1 - correct[order])
    risk = [{"coverage": float((i + 1) / len(y)), "risk": float(cumulative_errors[i] / (i + 1)),
             "threshold": float(confidence[order[i]])} for i in boundaries]
    return {"brier_multiclass_sum": float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1))),
            "log_loss": float(log_loss(y, p, labels=list(range(p.shape[1])))),
            "ece_10_bins": float(sum(b["count"] * abs(b["accuracy"] - b["confidence"]) for b in bins) / len(y)),
            "reliability": bins, "risk_coverage": risk}


def evaluate(records, labels, bootstrap_samples=300):
    valid = [r for r in records if r["error"] is None and r["probabilities"] is not None]
    truth = [r["true_label"] for r in records]
    predicted = [r["predicted_label"] if r["error"] is None and r["probabilities"] is not None else "__REQUEST_FAILED__" for r in records]
    correct = np.array([a == b for a, b in zip(truth, predicted)], dtype=float)
    latency = [r["latency_seconds"] * 1000 for r in records]
    rng = np.random.default_rng(2026)
    boot = [float(rng.choice(correct, len(correct), replace=True).mean()) for _ in range(bootstrap_samples)]
    result = {"n": len(records), "n_valid": len(valid), "failures": len(records) - len(valid),
        "accuracy_all": float(correct.mean()),
        "accuracy_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "macro_f1_all": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "recall_by_class": dict(zip(labels, recall_score(truth, predicted, labels=labels, average=None, zero_division=0).tolist())),
        "confusion_labels": labels + ["__REQUEST_FAILED__"],
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels + ["__REQUEST_FAILED__"]).tolist(),
        "p50_ms": float(np.percentile(latency, 50)), "p95_ms": float(np.percentile(latency, 95))}
    if valid:
        y = [labels.index(r["true_label"]) for r in valid]
        result["accuracy_valid_only"] = float(accuracy_score([r["true_label"] for r in valid], [r["predicted_label"] for r in valid]))
        result["probability_metrics_valid_only"] = probability_diagnostics(y, [r["probabilities"] for r in valid])
    return result
