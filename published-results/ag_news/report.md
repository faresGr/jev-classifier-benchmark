# Classifier benchmark report

Dataset: **AG News (4 news topics)** — `fancyzhx/ag_news` at Hub revision `eb185aade064a813bc0b7f42de02595523103ca4`.

Public/custom dataset benchmark; inspect the caveats before publishing.

| Model | Train labels | Seed | Test accuracy | Macro-F1 | Failed | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dummy | 40 | 11 | 0.250 | 0.100 | 0 | 0.12 | 0.14 |
| logreg | 40 | 11 | 0.495 | 0.487 | 0 | 0.16 | 0.18 |
| xgboost_tfidf | 40 | 11 | 0.275 | 0.253 | 0 | 0.23 | 0.26 |
| linear_svm | 40 | 11 | 0.445 | 0.433 | 0 | 1.06 | 1.11 |
| xgboost_svd | 40 | 11 | 0.343 | 0.313 | 0 | 0.27 | 0.30 |
| naive_bayes | 40 | 11 | 0.505 | 0.498 | 0 | 0.18 | 0.20 |
| xgboost_tfidf | 200 | 11 | 0.415 | 0.413 | 0 | 0.23 | 0.26 |
| linear_svm | 200 | 11 | 0.703 | 0.701 | 0 | 1.06 | 1.11 |
| dummy | 200 | 11 | 0.250 | 0.100 | 0 | 0.13 | 0.14 |
| naive_bayes | 200 | 11 | 0.728 | 0.726 | 0 | 0.19 | 0.21 |
| logreg | 200 | 11 | 0.723 | 0.720 | 0 | 0.16 | 0.18 |
| xgboost_svd | 200 | 11 | 0.560 | 0.564 | 0 | 0.35 | 0.39 |
| xgboost_tfidf | 800 | 11 | 0.657 | 0.661 | 0 | 0.23 | 0.27 |
| xgboost_svd | 800 | 11 | 0.740 | 0.737 | 0 | 0.46 | 0.53 |
| dummy | 800 | 11 | 0.250 | 0.100 | 0 | 0.13 | 0.14 |
| naive_bayes | 800 | 11 | 0.815 | 0.813 | 0 | 0.22 | 0.23 |
| logreg | 800 | 11 | 0.792 | 0.791 | 0 | 0.16 | 0.18 |
| linear_svm | 800 | 11 | 0.802 | 0.802 | 0 | 1.07 | 1.13 |
| linear_svm | 4000 | 11 | 0.873 | 0.872 | 0 | 1.07 | 1.12 |
| xgboost_tfidf | 4000 | 11 | 0.802 | 0.802 | 0 | 0.25 | 0.31 |
| xgboost_svd | 4000 | 11 | 0.825 | 0.824 | 0 | 0.45 | 0.49 |
| dummy | 4000 | 11 | 0.250 | 0.100 | 0 | 0.13 | 0.14 |
| logreg | 4000 | 11 | 0.865 | 0.865 | 0 | 0.17 | 0.18 |
| naive_bayes | 4000 | 11 | 0.860 | 0.860 | 0 | 0.23 | 0.24 |
| xgboost_svd | 40 | 22 | 0.393 | 0.385 | 0 | 0.26 | 0.29 |
| naive_bayes | 40 | 22 | 0.525 | 0.518 | 0 | 0.18 | 0.20 |
| dummy | 40 | 22 | 0.250 | 0.100 | 0 | 0.13 | 0.14 |
| logreg | 40 | 22 | 0.530 | 0.531 | 0 | 0.16 | 0.18 |
| linear_svm | 40 | 22 | 0.547 | 0.540 | 0 | 1.05 | 1.11 |
| xgboost_tfidf | 40 | 22 | 0.295 | 0.272 | 0 | 0.23 | 0.25 |
| linear_svm | 200 | 22 | 0.685 | 0.685 | 0 | 1.07 | 1.13 |
| xgboost_tfidf | 200 | 22 | 0.470 | 0.450 | 0 | 0.22 | 0.25 |
| xgboost_svd | 200 | 22 | 0.578 | 0.572 | 0 | 0.34 | 0.37 |
| logreg | 200 | 22 | 0.705 | 0.705 | 0 | 0.16 | 0.18 |
| naive_bayes | 200 | 22 | 0.723 | 0.724 | 0 | 0.20 | 0.21 |
| dummy | 200 | 22 | 0.250 | 0.100 | 0 | 0.13 | 0.14 |
| xgboost_tfidf | 800 | 22 | 0.670 | 0.667 | 0 | 0.23 | 0.28 |
| xgboost_svd | 800 | 22 | 0.755 | 0.755 | 0 | 0.49 | 0.61 |
| dummy | 800 | 22 | 0.250 | 0.100 | 0 | 0.13 | 0.15 |
| linear_svm | 800 | 22 | 0.825 | 0.824 | 0 | 1.11 | 1.19 |
| logreg | 800 | 22 | 0.810 | 0.809 | 0 | 0.17 | 0.18 |
| naive_bayes | 800 | 22 | 0.812 | 0.812 | 0 | 0.24 | 0.30 |
| dummy | 4000 | 22 | 0.250 | 0.100 | 0 | 0.13 | 0.14 |
| linear_svm | 4000 | 22 | 0.873 | 0.873 | 0 | 1.10 | 1.16 |
| logreg | 4000 | 22 | 0.868 | 0.868 | 0 | 0.17 | 0.19 |
| naive_bayes | 4000 | 22 | 0.860 | 0.860 | 0 | 0.23 | 0.25 |
| xgboost_svd | 4000 | 22 | 0.823 | 0.822 | 0 | 0.46 | 0.50 |
| xgboost_tfidf | 4000 | 22 | 0.812 | 0.812 | 0 | 0.24 | 0.27 |
| xgboost_svd | 40 | 33 | 0.315 | 0.256 | 0 | 0.26 | 0.29 |
| logreg | 40 | 33 | 0.490 | 0.488 | 0 | 0.17 | 0.19 |
| xgboost_tfidf | 40 | 33 | 0.355 | 0.321 | 0 | 0.23 | 0.25 |
| naive_bayes | 40 | 33 | 0.482 | 0.481 | 0 | 0.18 | 0.20 |
| dummy | 40 | 33 | 0.250 | 0.100 | 0 | 0.13 | 0.14 |
| linear_svm | 40 | 33 | 0.295 | 0.278 | 0 | 1.10 | 1.15 |
| xgboost_svd | 200 | 33 | 0.525 | 0.525 | 0 | 0.35 | 0.38 |
| dummy | 200 | 33 | 0.250 | 0.100 | 0 | 0.13 | 0.15 |
| logreg | 200 | 33 | 0.682 | 0.678 | 0 | 0.17 | 0.18 |
| xgboost_tfidf | 200 | 33 | 0.403 | 0.384 | 0 | 0.23 | 0.28 |
| linear_svm | 200 | 33 | 0.685 | 0.680 | 0 | 1.11 | 1.15 |
| naive_bayes | 200 | 33 | 0.713 | 0.708 | 0 | 0.20 | 0.21 |
| naive_bayes | 800 | 33 | 0.820 | 0.818 | 0 | 0.23 | 0.24 |
| xgboost_tfidf | 800 | 33 | 0.667 | 0.663 | 0 | 0.23 | 0.26 |
| dummy | 800 | 33 | 0.250 | 0.100 | 0 | 0.13 | 0.14 |
| logreg | 800 | 33 | 0.807 | 0.806 | 0 | 0.17 | 0.18 |
| linear_svm | 800 | 33 | 0.802 | 0.801 | 0 | 1.14 | 1.19 |
| xgboost_svd | 800 | 33 | 0.730 | 0.729 | 0 | 0.47 | 0.52 |
| logreg | 4000 | 33 | 0.858 | 0.857 | 0 | 0.18 | 0.20 |
| dummy | 4000 | 33 | 0.250 | 0.100 | 0 | 0.13 | 0.14 |
| xgboost_svd | 4000 | 33 | 0.825 | 0.825 | 0 | 0.46 | 0.50 |
| linear_svm | 4000 | 33 | 0.853 | 0.852 | 0 | 1.13 | 1.17 |
| xgboost_tfidf | 4000 | 33 | 0.815 | 0.815 | 0 | 0.24 | 0.28 |
| naive_bayes | 4000 | 33 | 0.873 | 0.872 | 0 | 0.23 | 0.26 |
| jev | 0 | 0 | 0.890 | 0.890 | 0 | 249.62 | 369.07 |

![Learning curves](learning_curves.png)

![Calibration](calibration.png)

## How to read these results

Classical models use training labels plus the fixed validation labels to choose hyperparameters. Jev receives category descriptions only. Training budgets exclude validation labels. Learning-curve bars show standard deviation across training-subset seeds, not confidence intervals over independent datasets.

The calibration plots use the largest training budget and the first seed for each model. Brier score (sum across classes), log loss, 10-bin ECE, per-class recall, confusion matrices, and per-run accuracy bootstrap intervals are in results.json. Bootstrap intervals are conditional on this fitted model and test sample; they do not capture training or dataset uncertainty.

Failures count as wrong in headline accuracy and macro-F1. Probability diagnostics use valid responses only; check failure counts before comparing those diagnostics. Risk–coverage curves describe the held-out test set; they do not select deployable thresholds. Jev confidence is logged separately and is not treated as a correctness probability.

Latency is sequential single-example wall time. Local models include vectorization after one warm-up; Jev includes network time, its first request, retries and backoff. These are deployment timings, not an equal-hardware architecture comparison. Runs are not cached. Training/tuning time is reported separately in results.json.

## Cost and limits

Local compute, labeling and engineering costs are not assumed to be zero. Jev token charges are estimated only if you supply current prices. Retry billing and missing usage can make that estimate a lower bound. No aggregate cost-per-correct claim is made from incomplete billing data.

Inspect metadata.json for dataset hashes, configuration, software versions and machine information. Predictions are in predictions.jsonl, including errors and Jev's raw response/usage when available. Well-known public datasets may overlap foundation-model pretraining, which can favour Jev; exact duplicates are removed but near-duplicates may remain. Use a private or newly collected labeled holdout before making broad claims.

Jev cost metadata: `{'known_response_token_cost_usd': 0.008032835999999996, 'records_missing_usage': 0, 'unaccounted_retry_attempts': 0, 'note': 'Token-only estimate; unknown attempt charges and local compute excluded'}`.
