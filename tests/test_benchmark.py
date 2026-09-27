import json
from pathlib import Path
import httpx
import numpy as np
import pytest

from jev_benchmark.data import prepare, load_bundle, training_subset, validate_bundle
from jev_benchmark.jev import classify, parse_answer
from jev_benchmark.metrics import probability_diagnostics, evaluate
from jev_benchmark.models import fit_best, predict_local

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle(tmp_path):
    config = json.loads((ROOT / "configs/demo.json").read_text())
    prepare(config, tmp_path / "data")
    rows, labels, _ = load_bundle(tmp_path / "data")
    return config, rows, sorted(labels)


def test_split_integrity_and_nested_samples(bundle):
    config, rows, labels = bundle
    small = training_subset(rows, labels, 6, 11)
    large = training_subset(rows, labels, 12, 11)
    assert {r["id"] for r in small} <= {r["id"] for r in large}
    assert all(r["split"] == "train" for r in large)
    assert len({r["id"] for r in rows}) == len(rows)


def test_reject_group_leakage(bundle):
    _, rows, labels = bundle
    train = next(r for r in rows if r["split"] == "train")
    test = next(r for r in rows if r["split"] == "test")
    train["group_id"] = test["group_id"] = "same-thread"
    with pytest.raises(ValueError, match="group_id"):
        validate_bundle(rows, labels)


def test_dataset_tampering(tmp_path):
    config = json.loads((ROOT / "configs/demo.json").read_text())
    prepare(config, tmp_path / "data")
    with (tmp_path / "data/examples.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="changed"):
        load_bundle(tmp_path / "data")


@pytest.mark.parametrize("name", ["dummy", "logreg", "naive_bayes", "linear_svm", "xgboost_tfidf", "xgboost_svd"])
def test_model_probability_contract(name, bundle):
    config, rows, labels = bundle
    train = training_subset(rows, labels, 6, 11)
    val = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]
    model, details = fit_best(name, train, val, labels, config, 11)
    predictions = predict_local(model, test, labels)
    assert len(predictions) == len(test)
    for p in predictions:
        assert len(p["probabilities"]) == len(labels)
        assert sum(p["probabilities"]) == pytest.approx(1, abs=1e-5)
        assert p["predicted_label"] in labels
    assert details["validation_trials"]


def test_calibration_perfect_and_ties():
    diag = probability_diagnostics([0, 1], [[1, 0], [0, 1]])
    assert diag["brier_multiclass_sum"] == 0
    assert diag["ece_10_bins"] == 0
    assert diag["risk_coverage"] == [{"coverage": 1, "risk": 0, "threshold": 1}]
    diag = probability_diagnostics([0, 1], [[0.5, 0.5], [0.5, 0.5]])
    assert diag["risk_coverage"][0]["coverage"] == 1
    assert diag["risk_coverage"][0]["risk"] == 0.5


def body():
    return {"model": "jev-pinned-test", "answers": {"category": {"type": "choice", "choice": "a",
        "probabilities": {"a": 0.8, "b": 0.2}, "confidence": 0.6}},
        "usage": {"input_tokens": 42, "output_tokens": 5}}


def test_http_retry_and_label_order():
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json=body())
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        r = classify(client, {"id": "one", "text": "example", "label": "a", "split": "test"},
            ["b", "a"], {"a": "alpha", "b": "beta"}, {"model": "jev-test", "max_attempts": 3}, sleep=lambda _: None)
    assert r["attempts"] == 2
    assert r["probabilities"] == [0.2, 0.8]
    assert r["confidence"] == 0.6
    assert "label" not in requests[0]["state"]
    assert requests[0]["state"] == "example"


@pytest.mark.parametrize("probabilities", [{"a": 0.8}, {"a": -0.2, "b": 1.2}, {"a": float("nan"), "b": 0.2}])
def test_malformed_probability_rejected(probabilities):
    b = body()
    b["answers"]["category"]["probabilities"] = probabilities
    with pytest.raises(ValueError):
        parse_answer(b, ["a", "b"])


def test_failure_counts_as_wrong():
    records = [
        {"true_label": "a", "predicted_label": "a", "probabilities": [0.8, 0.2], "error": None, "latency_seconds": 0.1},
        {"true_label": "b", "predicted_label": None, "probabilities": None, "error": "HTTP 500", "latency_seconds": 1},
    ]
    result = evaluate(records, ["a", "b"], 50)
    assert result["accuracy_all"] == 0.5
    assert result["accuracy_valid_only"] == 1
    assert result["recall_by_class"]["b"] == 0
    assert result["failures"] == 1


def test_auth_failure_stops_immediately():
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401))) as client:
        with pytest.raises(PermissionError):
            classify(client, {"id": "x", "text": "x", "label": "a", "split": "test"},
                     ["a", "b"], {"a": "alpha", "b": "beta"}, {"model": "jev-test", "max_attempts": 3})


def test_exhausted_retries_record_failure():
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
        r = classify(client, {"id": "x", "text": "x", "label": "a", "split": "test"},
                     ["a", "b"], {"a": "alpha", "b": "beta"}, {"model": "jev-test", "max_attempts": 3}, sleep=lambda _: None)
    assert r["attempts"] == 3
    assert r["error"] == "HTTP 503"
    assert r["probabilities"] is None


def test_probability_ties_use_returned_choice():
    diag = probability_diagnostics([1], [[0.5, 0.5]], predicted=[1])
    assert diag["reliability"][0]["accuracy"] == 1


def test_cli_live_path_with_mock_transport(tmp_path, monkeypatch):
    from argparse import Namespace
    import jev_benchmark.cli as cli
    config = json.loads((ROOT / "configs/demo.json").read_text())
    config.update(models=["dummy"], train_per_class=[6], bootstrap_samples=20)
    prepare(config, tmp_path / "data")
    labels = sorted(json.loads((tmp_path / "data/labels.json").read_text()))
    def handler(request):
        return httpx.Response(200, json={"model": "mock-only", "answers": {"category": {
            "type": "choice", "choice": labels[0], "probabilities": dict.fromkeys(labels, 0.25), "confidence": 0}},
            "usage": {"input_tokens": 100, "output_tokens": 10}})
    client_class = httpx.Client
    monkeypatch.setattr(cli.httpx, "Client", lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setenv("TYPESAFE_API_KEY", "mock-secret-not-for-logging")
    args = Namespace(data=str(tmp_path / "data"), out=str(tmp_path / "run"), with_jev=True, input_price=1.0, output_price=2.0)
    cli.run(args, config)
    results = json.loads((tmp_path / "run/results.json").read_text())
    assert results[-1]["model"] == "jev"
    assert results[-1]["billing"]["known_response_token_cost_usd"] == pytest.approx(32 * 120 / 1e6)
    assert (tmp_path / "run/calibration.png").exists()
    assert "mock-secret" not in (tmp_path / "run/metadata.json").read_text()
