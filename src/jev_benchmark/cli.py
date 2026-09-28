import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import httpx
import numpy as np

from .data import dump, load_bundle, prepare, training_subset
from .jev import ENDPOINT, INSTRUCTIONS, classify
from .metrics import evaluate
from .public import DATASETS as PUBLIC_DATASETS
from .models import fit_best, predict_local
from .report import make_report

MODELS = {"dummy", "logreg", "naive_bayes", "linear_svm", "xgboost_tfidf", "xgboost_svd"}


def load_config(path):
    c = json.loads(Path(path).read_text())
    if not c["models"] or not set(c["models"]) <= MODELS:
        raise ValueError("Unknown or empty model list")
    if not c["train_per_class"] or min(c["train_per_class"]) < 3:
        raise ValueError("Need at least 3 training examples/class for SVM calibration")
    if not c["training_seeds"]:
        raise ValueError("Need at least one training seed")
    for key in ("max_chars", "max_features", "svd_components", "xgb_estimators", "threads", "bootstrap_samples", "validation_per_class", "test_per_class"):
        if c[key] <= 0:
            raise ValueError(f"{key} must be positive")
    cap = c.get("max_train_pool_per_class")
    if cap is not None and cap < max(c["train_per_class"]):
        raise ValueError("max_train_pool_per_class must be at least the largest training budget")
    if not 1 <= c["jev"]["max_attempts"] <= 10 or c["jev"]["timeout_seconds"] <= 0:
        raise ValueError("Invalid Jev retry/timeout settings")
    return c


def append_predictions(path, run_id, records):
    with Path(path).open("a") as f:
        for r in records:
            f.write(json.dumps({"run_id": run_id, **r}, ensure_ascii=False, allow_nan=False) + "\n")


def run(args, config):
    rows, descriptions, dataset_meta = load_bundle(args.data)
    if config["max_chars"] != dataset_meta["max_chars"]:
        raise ValueError("max_chars differs from the prepared dataset; prepare a new bundle")
    labels = sorted(descriptions)
    validation = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]
    # Same deterministic shuffled test order for all methods.
    rng = np.random.default_rng(config["split_seed"])
    rng.shuffle(test)
    for seed in config["training_seeds"]:
        training_subset(rows, labels, max(config["train_per_class"]), seed)
    if args.with_jev and not os.environ.get("TYPESAFE_API_KEY"):
        raise ValueError("Set TYPESAFE_API_KEY to run Jev; omit --with-jev for local models")
    if (args.input_price is None) != (args.output_price is None):
        raise ValueError("Supply both --input-price and --output-price, or neither")
    if any(x is not None and (not np.isfinite(x) or x < 0) for x in (args.input_price, args.output_price)):
        raise ValueError("Token prices must be finite and nonnegative")
    out = Path(args.out)
    if out.exists():
        raise ValueError("Output directory exists; choose a new name to avoid overwriting a run")
    out.mkdir(parents=True)
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "scikit-learn", "matplotlib", "httpx", "xgboost"]}
    source_hash = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(file.name.encode() + file.read_bytes())
    meta = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "config": config, "dataset": dataset_meta, "labels": descriptions,
        "synthetic": dataset_meta["synthetic"], "versions": versions,
        "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
        "source_sha256": source_hash.hexdigest(), "validation_label_count": len(validation),
        "test_ids": [r["id"] for r in test], "with_jev": args.with_jev,
        "jev_endpoint": ENDPOINT, "jev_instructions": config["jev"].get("instructions", INSTRUCTIONS),
        "input_price_per_million": args.input_price, "output_price_per_million": args.output_price,
        "inference_concurrency": 1}
    dump(out / "metadata.json", meta)
    results = []
    try:
        for seed in config["training_seeds"]:
            for size in sorted(set(config["train_per_class"])):
                train = training_subset(rows, labels, size, seed)
                names = list(config["models"])
                np.random.default_rng(seed + size).shuffle(names)
                for name in names:
                    run_id = f"{name}-n{size}-seed{seed}"
                    print(f"Running {run_id}", flush=True)
                    model, details = fit_best(name, train, validation, labels, config, seed)
                    start = perf_counter()
                    predictions = predict_local(model, test, labels)
                    elapsed = perf_counter() - start
                    append_predictions(out / "predictions.jsonl", run_id, predictions)
                    result = {"run_id": run_id, "model": name, "seed": seed,
                        "n_train": len(train), "train_per_class": size,
                        "train_ids": [r["id"] for r in train], **details,
                        "test_inference_wall_seconds": elapsed,
                        "test_examples_per_second": len(test) / elapsed,
                        "metrics": evaluate(predictions, labels, config["bootstrap_samples"])}
                    results.append(result)
                    dump(out / "results.json", results)
        if args.with_jev:
            predictions = []
            start = perf_counter()
            with httpx.Client(headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"},
                              timeout=config["jev"]["timeout_seconds"], follow_redirects=False) as client:
                for i, row in enumerate(test):
                    prediction = classify(client, row, labels, descriptions, config["jev"])
                    append_predictions(out / "predictions.jsonl", "jev", [prediction])
                    predictions.append(prediction)
                    if (i + 1) % 10 == 0:
                        print(f"Jev {i + 1}/{len(test)}", flush=True)
            elapsed = perf_counter() - start
            known_cost, usage_missing = 0.0, 0
            for p in predictions:
                usage = p.get("usage") or {}
                if not all(isinstance(usage.get(k), int) and usage[k] >= 0 for k in ("input_tokens", "output_tokens")):
                    usage_missing += 1
                elif args.input_price is not None:
                    known_cost += (usage["input_tokens"] * args.input_price + usage["output_tokens"] * args.output_price) / 1e6
            results.append({"run_id": "jev", "model": "jev", "seed": 0, "n_train": 0,
                "resolved_models": sorted({p["resolved_model"] for p in predictions if p["resolved_model"]}),
                "test_inference_wall_seconds": elapsed, "test_examples_per_second": len(test) / elapsed,
                "billing": {"known_response_token_cost_usd": known_cost if args.input_price is not None else None,
                            "records_missing_usage": usage_missing,
                            "unaccounted_retry_attempts": sum(p["attempts"] - 1 for p in predictions),
                            "note": "Token-only estimate; unknown attempt charges and local compute excluded"},
                "metrics": evaluate(predictions, labels, config["bootstrap_samples"])})
            dump(out / "results.json", results)
        make_report(out, results, meta)
        meta["status"] = "complete"
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        dump(out / "metadata.json", meta)
    except BaseException as exc:
        meta["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        meta["error_type"] = type(exc).__name__
        dump(out / "metadata.json", meta)
        raise
    print(f"Report: {(out / 'report.md').resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Classical text classifiers versus Jev")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Freeze a dataset and its train/validation/test splits")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--input", help="Custom JSONL with id, text, label, split; optional group_id")
    p.add_argument("--labels", help="JSON label-to-description map for custom data")
    r = sub.add_parser("run", help="Train local models, optionally call Jev, produce a report")
    r.add_argument("--config", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--with-jev", action="store_true", help="Send held-out texts to TypeSafe (paid API)")
    r.add_argument("--input-price", type=float, help="Current USD per million input tokens; no default assumption")
    r.add_argument("--output-price", type=float, help="Current USD per million output tokens")
    sub.add_parser("list-datasets", help="Show the built-in public datasets")
    args = parser.parse_args()
    if args.command == "list-datasets":
        print("newsgroups   20 Newsgroups, 4 topics (scikit-learn download)")
        for name, spec in PUBLIC_DATASETS.items():
            print(f"{name:<12} {spec['title']} (Hugging Face: {spec['repos'][0]})")
        return
    try:
        config = load_config(args.config)
        if args.command == "prepare":
            rows = prepare(config, args.out, args.input, args.labels)
            print(f"Prepared {len(rows)} records in {args.out}")
        else:
            run(args, config)
    except (ValueError, PermissionError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
