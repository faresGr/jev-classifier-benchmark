# Classifier benchmark report

**Synthetic demonstration only — not evidence of real-world model quality.**

| Model | Train labels | Seed | Test accuracy | Macro-F1 | Failed | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| logreg | 24 | 11 | 0.594 | 0.600 | 0 | 0.16 | 0.23 |
| dummy | 24 | 11 | 0.250 | 0.100 | 0 | 0.12 | 0.15 |
| xgboost_tfidf | 24 | 11 | 0.438 | 0.414 | 0 | 0.24 | 0.32 |
| naive_bayes | 24 | 11 | 0.531 | 0.539 | 0 | 0.17 | 0.20 |
| xgboost_svd | 24 | 11 | 0.375 | 0.377 | 0 | 0.26 | 0.31 |
| linear_svm | 24 | 11 | 0.562 | 0.564 | 0 | 1.10 | 1.22 |
| xgboost_svd | 48 | 11 | 0.688 | 0.683 | 0 | 0.27 | 0.35 |
| naive_bayes | 48 | 11 | 0.938 | 0.935 | 0 | 0.17 | 0.18 |
| linear_svm | 48 | 11 | 0.906 | 0.900 | 0 | 1.10 | 1.26 |
| dummy | 48 | 11 | 0.250 | 0.100 | 0 | 0.12 | 0.13 |
| logreg | 48 | 11 | 0.906 | 0.900 | 0 | 0.16 | 0.17 |
| xgboost_tfidf | 48 | 11 | 0.406 | 0.424 | 0 | 0.24 | 0.31 |

![Learning curves](learning_curves.png)

![Calibration](calibration.png)

## How to read these results

Classical models use training labels plus the fixed validation labels to choose hyperparameters. Jev receives category descriptions only. Training budgets exclude validation labels. Learning-curve bars show standard deviation across training-subset seeds, not confidence intervals over independent datasets.

The calibration plots use the largest training budget and the first seed for each model. Brier score (sum across classes), log loss, 10-bin ECE, per-class recall, confusion matrices, and per-run accuracy bootstrap intervals are in results.json. Bootstrap intervals are conditional on this fitted model and test sample; they do not capture training or dataset uncertainty.

Failures count as wrong in headline accuracy and macro-F1. Probability diagnostics use valid responses only; check failure counts before comparing those diagnostics. Risk–coverage curves describe the held-out test set; they do not select deployable thresholds. Jev confidence is logged separately and is not treated as a correctness probability.

Latency is sequential single-example wall time. Local models include vectorization after one warm-up; Jev includes network time, its first request, retries and backoff. These are deployment timings, not an equal-hardware architecture comparison. Runs are not cached. Training/tuning time is reported separately in results.json.

## Cost and limits

Local compute, labeling and engineering costs are not assumed to be zero. Jev token charges are estimated only if you supply current prices. Retry billing and missing usage can make that estimate a lower bound. No aggregate cost-per-correct claim is made from incomplete billing data.

Inspect metadata.json for dataset hashes, configuration, software versions and machine information. Predictions are in predictions.jsonl, including errors and Jev's raw response/usage when available. The public dataset is old and may overlap foundation-model pretraining; exact duplicates are removed but near-duplicate threads may remain. Use a private or newly collected labeled holdout before making broad claims.

**Jev has not been run in this report. No Jev comparison result is available yet.**
