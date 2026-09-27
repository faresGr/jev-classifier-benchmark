# Classifier benchmark report

Public/custom dataset benchmark; inspect the caveats before publishing.

| Model | Train labels | Seed | Test accuracy | Macro-F1 | Failed | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dummy | 40 | 11 | 0.250 | 0.100 | 0 | 0.15 | 0.27 |
| logreg | 40 | 11 | 0.485 | 0.483 | 0 | 0.18 | 0.31 |
| xgboost_tfidf | 40 | 11 | 0.282 | 0.280 | 0 | 0.26 | 0.40 |
| linear_svm | 40 | 11 | 0.507 | 0.508 | 0 | 1.20 | 1.68 |
| xgboost_svd | 40 | 11 | 0.335 | 0.277 | 0 | 0.32 | 0.50 |
| naive_bayes | 40 | 11 | 0.520 | 0.517 | 0 | 0.22 | 0.36 |
| xgboost_tfidf | 200 | 11 | 0.568 | 0.569 | 0 | 0.26 | 0.43 |
| linear_svm | 200 | 11 | 0.740 | 0.741 | 0 | 1.15 | 1.62 |
| dummy | 200 | 11 | 0.250 | 0.100 | 0 | 0.15 | 0.29 |
| naive_bayes | 200 | 11 | 0.738 | 0.740 | 0 | 0.24 | 0.40 |
| logreg | 200 | 11 | 0.720 | 0.723 | 0 | 0.19 | 0.35 |
| xgboost_svd | 200 | 11 | 0.645 | 0.645 | 0 | 0.49 | 0.66 |
| xgboost_tfidf | 800 | 11 | 0.713 | 0.712 | 0 | 0.28 | 0.44 |
| xgboost_svd | 800 | 11 | 0.770 | 0.771 | 0 | 0.49 | 0.69 |
| dummy | 800 | 11 | 0.250 | 0.100 | 0 | 0.15 | 0.31 |
| naive_bayes | 800 | 11 | 0.848 | 0.848 | 0 | 0.25 | 0.42 |
| logreg | 800 | 11 | 0.833 | 0.832 | 0 | 0.19 | 0.34 |
| linear_svm | 800 | 11 | 0.833 | 0.832 | 0 | 1.18 | 1.73 |
| xgboost_svd | 40 | 22 | 0.323 | 0.261 | 0 | 0.30 | 0.45 |
| naive_bayes | 40 | 22 | 0.502 | 0.489 | 0 | 0.21 | 0.34 |
| dummy | 40 | 22 | 0.250 | 0.100 | 0 | 0.15 | 0.29 |
| logreg | 40 | 22 | 0.497 | 0.487 | 0 | 0.18 | 0.31 |
| linear_svm | 40 | 22 | 0.520 | 0.515 | 0 | 1.15 | 1.53 |
| xgboost_tfidf | 40 | 22 | 0.270 | 0.249 | 0 | 0.25 | 0.39 |
| linear_svm | 200 | 22 | 0.730 | 0.728 | 0 | 1.17 | 1.68 |
| xgboost_tfidf | 200 | 22 | 0.497 | 0.497 | 0 | 0.26 | 0.43 |
| xgboost_svd | 200 | 22 | 0.637 | 0.618 | 0 | 0.48 | 0.67 |
| logreg | 200 | 22 | 0.725 | 0.728 | 0 | 0.19 | 0.33 |
| naive_bayes | 200 | 22 | 0.743 | 0.743 | 0 | 0.25 | 0.41 |
| dummy | 200 | 22 | 0.250 | 0.100 | 0 | 0.15 | 0.30 |
| xgboost_tfidf | 800 | 22 | 0.710 | 0.710 | 0 | 0.26 | 0.45 |
| xgboost_svd | 800 | 22 | 0.782 | 0.783 | 0 | 0.50 | 0.69 |
| dummy | 800 | 22 | 0.250 | 0.100 | 0 | 0.16 | 0.31 |
| linear_svm | 800 | 22 | 0.833 | 0.832 | 0 | 1.21 | 1.82 |
| logreg | 800 | 22 | 0.830 | 0.831 | 0 | 0.20 | 0.39 |
| naive_bayes | 800 | 22 | 0.858 | 0.858 | 0 | 0.27 | 0.43 |
| xgboost_svd | 40 | 33 | 0.355 | 0.292 | 0 | 0.33 | 0.51 |
| logreg | 40 | 33 | 0.677 | 0.674 | 0 | 0.20 | 0.40 |
| xgboost_tfidf | 40 | 33 | 0.315 | 0.281 | 0 | 0.25 | 0.40 |
| naive_bayes | 40 | 33 | 0.695 | 0.692 | 0 | 0.21 | 0.35 |
| dummy | 40 | 33 | 0.250 | 0.100 | 0 | 0.15 | 0.30 |
| linear_svm | 40 | 33 | 0.615 | 0.612 | 0 | 1.16 | 1.59 |
| xgboost_svd | 200 | 33 | 0.652 | 0.657 | 0 | 0.48 | 0.67 |
| dummy | 200 | 33 | 0.250 | 0.100 | 0 | 0.15 | 0.30 |
| logreg | 200 | 33 | 0.775 | 0.776 | 0 | 0.19 | 0.36 |
| xgboost_tfidf | 200 | 33 | 0.550 | 0.540 | 0 | 0.26 | 0.43 |
| linear_svm | 200 | 33 | 0.765 | 0.766 | 0 | 1.19 | 1.69 |
| naive_bayes | 200 | 33 | 0.772 | 0.775 | 0 | 0.25 | 0.41 |
| naive_bayes | 800 | 33 | 0.812 | 0.816 | 0 | 0.25 | 0.40 |
| xgboost_tfidf | 800 | 33 | 0.705 | 0.705 | 0 | 0.26 | 0.44 |
| dummy | 800 | 33 | 0.250 | 0.100 | 0 | 0.16 | 0.32 |
| logreg | 800 | 33 | 0.828 | 0.828 | 0 | 0.19 | 0.35 |
| linear_svm | 800 | 33 | 0.838 | 0.838 | 0 | 1.21 | 1.74 |
| xgboost_svd | 800 | 33 | 0.775 | 0.776 | 0 | 0.49 | 0.71 |

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
