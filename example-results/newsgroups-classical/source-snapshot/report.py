"""Portable Markdown report plus static PNG figures."""
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_report(destination, runs, meta):
    dest = Path(destination)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = defaultdict(list)
    for run in runs:
        groups[run["model"]].append(run)
    for name, group in groups.items():
        if name == "jev":
            ax.axhline(group[0]["metrics"]["macro_f1_all"], ls="--", color="black", label="Jev: zero-shot")
            continue
        by_n = defaultdict(list)
        for r in group:
            by_n[r["n_train"]].append(r["metrics"]["macro_f1_all"])
        xs = sorted(by_n)
        means = [np.mean(by_n[x]) for x in xs]
        stds = [np.std(by_n[x]) for x in xs]
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=name)
    ax.set(xlabel="Labeled training examples (validation labels additional)", ylabel="Test macro-F1",
           title="Classical learning curves vs Jev" + (" — SYNTHETIC DEMO" if meta["synthetic"] else ""), ylim=(0, 1.03))
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(dest / "learning_curves.png", dpi=170); plt.close(fig)
    # Largest training budget, first training seed; do not select the best test result.
    selected = []
    for name, group in groups.items():
        selected.append(sorted(group, key=lambda r: (-r["n_train"], r["seed"]))[0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    for run in selected:
        diag = run["metrics"].get("probability_metrics_valid_only")
        if not diag:
            continue
        bins, risk = diag["reliability"], diag["risk_coverage"]
        axes[0].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", label=run["model"])
        axes[1].plot([b["coverage"] for b in risk], [b["risk"] for b in risk], label=run["model"])
    axes[0].set(xlabel="Maximum class probability", ylabel="Observed accuracy", title="Reliability (valid responses)", xlim=(0, 1), ylim=(0, 1.03))
    axes[1].set(xlabel="Coverage among valid responses", ylabel="Error among accepted predictions", title="Risk–coverage (descriptive test curves)", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(dest / "calibration.png", dpi=170); plt.close(fig)
    lines = ["# Classifier benchmark report", "",
        "**Synthetic demonstration only — not evidence of real-world model quality.**" if meta["synthetic"] else "Public/custom dataset benchmark; inspect the caveats before publishing.", "",
        "| Model | Train labels | Seed | Test accuracy | Macro-F1 | Failed | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        m = r["metrics"]
        lines.append(f"| {r['model']} | {r['n_train']} | {r['seed']} | {m['accuracy_all']:.3f} | {m['macro_f1_all']:.3f} | {m['failures']} | {m['p50_ms']:.2f} | {m['p95_ms']:.2f} |")
    lines += ["", "![Learning curves](learning_curves.png)", "", "![Calibration](calibration.png)", "",
        "## How to read these results", "",
        "Classical models use training labels plus the fixed validation labels to choose hyperparameters. Jev receives category descriptions only. Training budgets exclude validation labels. Learning-curve bars show standard deviation across training-subset seeds, not confidence intervals over independent datasets.", "",
        "The calibration plots use the largest training budget and the first seed for each model. Brier score (sum across classes), log loss, 10-bin ECE, per-class recall, confusion matrices, and per-run accuracy bootstrap intervals are in results.json. Bootstrap intervals are conditional on this fitted model and test sample; they do not capture training or dataset uncertainty.", "",
        "Failures count as wrong in headline accuracy and macro-F1. Probability diagnostics use valid responses only; check failure counts before comparing those diagnostics. Risk–coverage curves describe the held-out test set; they do not select deployable thresholds. Jev confidence is logged separately and is not treated as a correctness probability.", "",
        "Latency is sequential single-example wall time. Local models include vectorization after one warm-up; Jev includes network time, its first request, retries and backoff. These are deployment timings, not an equal-hardware architecture comparison. Runs are not cached. Training/tuning time is reported separately in results.json.", "",
        "## Cost and limits", "",
        "Local compute, labeling and engineering costs are not assumed to be zero. Jev token charges are estimated only if you supply current prices. Retry billing and missing usage can make that estimate a lower bound. No aggregate cost-per-correct claim is made from incomplete billing data.", "",
        "Inspect metadata.json for dataset hashes, configuration, software versions and machine information. Predictions are in predictions.jsonl, including errors and Jev's raw response/usage when available. The public dataset is old and may overlap foundation-model pretraining; exact duplicates are removed but near-duplicate threads may remain. Use a private or newly collected labeled holdout before making broad claims."]
    if "jev" not in groups:
        lines += ["", "**Jev has not been run in this report. No Jev comparison result is available yet.**"]
    else:
        run = groups["jev"][0]
        lines += ["", "Jev cost metadata: `" + str(run["billing"]) + "`."]
    (dest / "report.md").write_text("\n".join(lines) + "\n")
