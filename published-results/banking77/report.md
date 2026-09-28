# Classifier benchmark report

Dataset: **Banking77 (77 fine-grained customer intents)** — `mteb/banking77` at Hub revision `18072d2685ea682290f7b8924d94c62acc19c0b2`.

Public/custom dataset benchmark; inspect the caveats before publishing.

| Model | Train labels | Seed | Test accuracy | Macro-F1 | Failed | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dummy | 385 | 11 | 0.013 | 0.000 | 0 | 0.12 | 0.13 |
| linear_svm | 385 | 11 | 0.508 | 0.497 | 0 | 3.47 | 3.54 |
| logreg | 385 | 11 | 0.547 | 0.543 | 0 | 0.15 | 0.16 |
| xgboost_tfidf | 385 | 11 | 0.329 | 0.324 | 0 | 0.49 | 0.53 |
| xgboost_svd | 385 | 11 | 0.312 | 0.310 | 0 | 0.49 | 0.53 |
| naive_bayes | 385 | 11 | 0.522 | 0.517 | 0 | 0.23 | 0.26 |
| dummy | 770 | 11 | 0.013 | 0.000 | 0 | 0.11 | 0.12 |
| logreg | 770 | 11 | 0.692 | 0.689 | 0 | 0.15 | 0.16 |
| xgboost_tfidf | 770 | 11 | 0.462 | 0.457 | 0 | 0.51 | 0.55 |
| linear_svm | 770 | 11 | 0.664 | 0.654 | 0 | 3.48 | 3.56 |
| xgboost_svd | 770 | 11 | 0.447 | 0.442 | 0 | 0.52 | 0.56 |
| naive_bayes | 770 | 11 | 0.647 | 0.640 | 0 | 0.28 | 0.30 |
| xgboost_tfidf | 1540 | 11 | 0.601 | 0.595 | 0 | 0.51 | 0.55 |
| naive_bayes | 1540 | 11 | 0.752 | 0.749 | 0 | 0.35 | 0.37 |
| logreg | 1540 | 11 | 0.775 | 0.773 | 0 | 0.15 | 0.16 |
| xgboost_svd | 1540 | 11 | 0.597 | 0.594 | 0 | 0.59 | 0.62 |
| linear_svm | 1540 | 11 | 0.778 | 0.774 | 0 | 3.64 | 3.78 |
| dummy | 1540 | 11 | 0.013 | 0.000 | 0 | 0.12 | 0.13 |
| logreg | 385 | 22 | 0.549 | 0.543 | 0 | 0.16 | 0.17 |
| dummy | 385 | 22 | 0.013 | 0.000 | 0 | 0.12 | 0.13 |
| naive_bayes | 385 | 22 | 0.519 | 0.510 | 0 | 0.24 | 0.25 |
| linear_svm | 385 | 22 | 0.469 | 0.451 | 0 | 3.61 | 3.72 |
| xgboost_tfidf | 385 | 22 | 0.258 | 0.244 | 0 | 0.51 | 0.55 |
| xgboost_svd | 385 | 22 | 0.339 | 0.330 | 0 | 0.50 | 0.52 |
| xgboost_svd | 770 | 22 | 0.478 | 0.476 | 0 | 0.54 | 0.57 |
| naive_bayes | 770 | 22 | 0.664 | 0.656 | 0 | 0.29 | 0.31 |
| dummy | 770 | 22 | 0.013 | 0.000 | 0 | 0.12 | 0.13 |
| logreg | 770 | 22 | 0.697 | 0.693 | 0 | 0.16 | 0.17 |
| linear_svm | 770 | 22 | 0.664 | 0.657 | 0 | 3.66 | 3.72 |
| xgboost_tfidf | 770 | 22 | 0.484 | 0.483 | 0 | 0.55 | 0.59 |
| linear_svm | 1540 | 22 | 0.769 | 0.768 | 0 | 3.66 | 3.72 |
| naive_bayes | 1540 | 22 | 0.766 | 0.764 | 0 | 0.37 | 0.39 |
| xgboost_svd | 1540 | 22 | 0.609 | 0.611 | 0 | 0.59 | 0.62 |
| xgboost_tfidf | 1540 | 22 | 0.614 | 0.611 | 0 | 0.52 | 0.56 |
| logreg | 1540 | 22 | 0.788 | 0.789 | 0 | 0.16 | 0.17 |
| dummy | 1540 | 22 | 0.013 | 0.000 | 0 | 0.12 | 0.13 |
| xgboost_svd | 385 | 33 | 0.322 | 0.319 | 0 | 0.50 | 0.52 |
| dummy | 385 | 33 | 0.013 | 0.000 | 0 | 0.12 | 0.13 |
| naive_bayes | 385 | 33 | 0.549 | 0.539 | 0 | 0.24 | 0.25 |
| linear_svm | 385 | 33 | 0.556 | 0.543 | 0 | 3.67 | 3.76 |
| xgboost_tfidf | 385 | 33 | 0.282 | 0.276 | 0 | 0.51 | 0.55 |
| logreg | 385 | 33 | 0.582 | 0.569 | 0 | 0.16 | 0.17 |
| xgboost_svd | 770 | 33 | 0.462 | 0.457 | 0 | 0.54 | 0.57 |
| logreg | 770 | 33 | 0.673 | 0.672 | 0 | 0.16 | 0.17 |
| xgboost_tfidf | 770 | 33 | 0.440 | 0.432 | 0 | 0.53 | 0.57 |
| naive_bayes | 770 | 33 | 0.649 | 0.650 | 0 | 0.29 | 0.30 |
| dummy | 770 | 33 | 0.013 | 0.000 | 0 | 0.12 | 0.13 |
| linear_svm | 770 | 33 | 0.656 | 0.650 | 0 | 3.65 | 3.73 |
| xgboost_tfidf | 1540 | 33 | 0.617 | 0.619 | 0 | 0.52 | 0.59 |
| logreg | 1540 | 33 | 0.768 | 0.768 | 0 | 0.16 | 0.17 |
| xgboost_svd | 1540 | 33 | 0.622 | 0.619 | 0 | 0.59 | 0.64 |
| dummy | 1540 | 33 | 0.013 | 0.000 | 0 | 0.12 | 0.14 |
| naive_bayes | 1540 | 33 | 0.743 | 0.744 | 0 | 0.36 | 0.39 |
| linear_svm | 1540 | 33 | 0.753 | 0.753 | 0 | 3.64 | 3.70 |
| jev | 0 | 0 | 0.818 | 0.809 | 0 | 261.04 | 369.01 |

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

Jev cost metadata: `{'known_response_token_cost_usd': 0.07990197600000001, 'records_missing_usage': 0, 'unaccounted_retry_attempts': 0, 'note': 'Token-only estimate; unknown attempt charges and local compute excluded'}`.
