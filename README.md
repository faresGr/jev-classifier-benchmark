# Jev versus classical text classifiers

A Python experiment for the question: **how much labeled data does a classical classifier need to match Jev on a text-classification task?** It ships with 20 Newsgroups plus five other well-known text-classification datasets (AG News, DBpedia-14, IMDB, Emotion, Banking77).

This project compares six classical configurations with an optional live Jev run, using identical held-out texts. It produces learning curves, calibration plots, a Markdown report, and auditable JSON/JSONL records. No Jev results are simulated.

## Published experiment

The [blog post](https://faresgr.github.io/blog/2026/how-many-labels-is-a-description-worth/) compares Jev with the classical baselines on AG News, Banking77 and Emotion. The [saved predictions and corrected reports](published-results/README.md) include all five nontrivial classical configurations, the dummy control and actual Jev outputs.

To reproduce the scoring, paired intervals and blog charts **without an API key**:

```sh
python scripts/publish_analysis.py
```

See `published-results/README.md` for exact label budgets, provenance, parser correction and interpretation limits. `scripts/prompt_followup.py` runs a separate, explicitly post-hoc prompt sensitivity check; it requires a key and makes paid API calls. Optional task instructions can also be supplied via `jev.instructions` in a normal run config, and are recorded in run metadata.

## Models

| Model | Input representation | Selection on validation data |
|---|---|---|
| Majority/prior control | TF-IDF (ignored by classifier) | No tuning |
| Logistic regression | Word unigram/bigram TF-IDF | C = 0.1, 1, 10 |
| Multinomial Naive Bayes | Same TF-IDF | alpha = 0.1, 1, 10 |
| Calibrated linear SVM | Same TF-IDF, fitted separately inside each calibration fold | C = 0.1, 1, 10; training-only 3-fold sigmoid calibration |
| XGBoost | Sparse TF-IDF | max_depth = 3, 6 |
| XGBoost + SVD | TF-IDF reduced with training-only TruncatedSVD | max_depth = 3, 6 |
| Jev (optional) | Text plus natural-language category descriptions | Fixed prompt; no task-specific training |

XGBoost is a strong additional baseline, not a guaranteed winner for sparse text. Its two representations help distinguish model choice from feature choice. The SVD variant is latent semantic analysis, **not pretrained neural embeddings**. The searches are deliberately small; this is a reproducible starting comparison, not a claim that every model has been exhaustively optimized.

## Install

Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate  # fish: source .venv/bin/activate.fish
python -m pip install -e '.[dev]'
```

On macOS, XGBoost also needs the OpenMP runtime. If loading XGBoost reports a missing `libomp.dylib`:

```bash
brew install libomp
```

`requirements-tested.txt` records the versions used to validate this project. For closer reproduction, install that file before the editable project. It captures Python packages, not native runtimes or a cross-platform lock.

## Quick offline demo

Run these commands from this project's directory:

```bash
jev-bench prepare --config configs/demo.json --out data/demo
jev-bench run --config configs/demo.json --data data/demo --out runs/demo
```

Open `runs/demo/report.md`. The tiny, deliberately easy synthetic dataset checks the pipeline; shared templates make it unsuitable for scientific claims. There is no network access in this demo after installing dependencies, and no API key is needed.

## Run the public-data experiment

```bash
jev-bench prepare --config configs/newsgroups.json --out data/newsgroups
jev-bench run --config configs/newsgroups.json --data data/newsgroups --out runs/classical
```

The first command downloads 20 Newsgroups using scikit-learn. It selects four topics: computer graphics, cars, baseball, and space. Headers, footers and quoted text are stripped using scikit-learn's heuristics. Text is capped at 8,000 characters **before deduplication**, identically for all methods. Empty texts and normalized exact duplicates are removed.

The original public test partition supplies the test examples. The original training partition supplies training and validation. The default experiment uses 50 validation and 100 test examples per class; classical training curves use 10, 50 and 200 examples per class, repeated with three training-subset seeds. Smaller training subsets are nested inside larger ones for each seed. The fixed validation and test sets are shared across all runs.

The full run has **6 models × 3 budgets × 3 seeds = 54 configurations**, each with its documented candidate search. Edit the config for a smaller pilot. The default uses two XGBoost threads and up to 20,000 TF-IDF features. Runtime depends on your machine.

20 Newsgroups is old and may have appeared in foundation-model pretraining. Exact deduplication does not remove all related threads or near-duplicates. Treat this as an accessible reference experiment; a fresh, independently labeled dataset would make the stronger blog result. Labels are topic categories, not an independently audited semantic truth for every stripped/truncated message.

## More public benchmarks

Five further well-known datasets cover different kinds of classification, so a single topic task does not decide the comparison:

| Config | Task | Classes | Train labels/class | Val/class | Test/class (Jev calls) | Why it is included |
|---|---|---:|---|---:|---:|---|
| `ag_news` | News topic | 4 | 10, 50, 200, 1000 | 50 | 100 (400) | Classic topic benchmark; TF-IDF is strong once data is plentiful |
| `dbpedia_14` | Wikipedia entity type | 14 | 10, 50, 200, 1000 | 50 | 50 (700) | More classes, short encyclopedic text |
| `imdb` | Review sentiment | 2 | 10, 50, 200, 1000 | 50 | 200 (400) | Long texts; sentiment rather than topic |
| `emotion` | Emotion in tweets | 6 | 10, 50, 200, 500 | 50 | 50 (300) | Short, noisy text; semantically close classes |
| `banking77` | Customer intent | 77 | 5, 10, 20 | 5 | 10 (770) | Fine-grained few-shot regime where label descriptions should matter most |

They are downloaded from the Hugging Face Hub, which needs one extra dependency:

```bash
python -m pip install -e '.[dev,public]'
jev-bench list-datasets
jev-bench prepare --config configs/ag_news.json --out data/ag_news
jev-bench run --config configs/ag_news.json --data data/ag_news --out runs/ag_news
```

Or all of them (classical only):

```bash
for d in ag_news dbpedia_14 imdb emotion banking77; do
  jev-bench prepare --config configs/$d.json --out data/$d
  jev-bench run --config configs/$d.json --data data/$d --out runs/$d
done
```

Add `--with-jev` to a run exactly as for 20 Newsgroups; the table above gives the number of paid requests per dataset.

How they are prepared:

- Test examples come only from each dataset's official test split. Everything else (including Emotion's official validation split) forms the pool for validation and training. IMDB's unlabeled split is ignored.
- Exact normalized duplicates are removed with training-side texts kept first, so a test text that also appears in training is dropped from test.
- `max_train_pool_per_class` stores only a shuffled subset of each class's training pool, keeping huge corpora such as DBpedia (560k training texts) small on disk. Every training budget is still sampled, nested, from that stored pool.
- The Hub commit that was downloaded is recorded in the bundle's `metadata.json` (`provenance.hub_revision`) and in the report. Set `hub_revision` in the config to that value to rebuild the same data later.
- Banking77 is read from `PolyAI/banking77`, falling back to the `mteb/banking77` mirror if the original repository cannot be loaded by your `datasets` version. The repository actually used is recorded.
- DBpedia text is `title. content`. IMDB's `<br />` tags become line breaks.

Category descriptions for Jev are in `src/jev_benchmark/public.py`. For AG News, DBpedia, IMDB and Emotion they are short hand-written definitions. Banking77's 77 descriptions are generated mechanically from the label names (for example `card_arrival` becomes "…whose intent is: card arrival."), so they add no extra hints; better descriptions may improve Jev and would be a legitimate part of its cost.

Caveats: all five are widely used and very likely appear in foundation-model pretraining data, which can favour Jev. Budgets are balanced per class, so these results do not show behaviour under the original class imbalance. Banking77's 10 test examples per class give noisy per-class recall; raise `test_per_class` (the official test split has 40 per class) if the cost is acceptable. XGBoost is the slowest model on many-class data: Banking77 took about 9 minutes for the full classical grid on two threads.

## Add Jev

Set `TYPESAFE_API_KEY` in your shell using your normal secret-management method. Credentials are read from the environment and are not written to reports. Then:

```bash
jev-bench run --config configs/newsgroups.json --data data/newsgroups \
  --out runs/with-jev --with-jev
```

This runs the local comparison and then sends the same **400 test texts** to TypeSafe, one request at a time. Jev is evaluated once, not once per classical training budget. Category descriptions are in the dataset's `labels.json`; the prompt is in `jev.py` and is copied into run metadata. The request contains the text and descriptions, never the example's true label.

The configuration uses `jev-latest` for convenience. Before a publishable run, replace it with a currently available pinned model identifier. The actual model identifier returned by every response is logged; if it changes during a run, report that limitation. Check your account's available models in the [TypeSafe documentation](https://docs.typesafe.ai/models).

No token price is assumed. To estimate charges, pass both current prices in USD per million tokens:

```bash
jev-bench run --config configs/newsgroups.json --data data/newsgroups \
  --out runs/with-jev-priced --with-jev \
  --input-price "$JEV_INPUT_USD_PER_MILLION" \
  --output-price "$JEV_OUTPUT_USD_PER_MILLION"
```

These environment variables must contain the prices you have verified for your account. Charges for failed/retried attempts may be unknown; the report explicitly records unaccounted attempts and missing usage. Local compute, training, labeling and engineering costs are not zero and are not estimated automatically.

Requests use bounded retries for rate limits, overload and transient server/network errors. Latency includes retries and backoff. Authentication failures stop the run. Other failed responses remain in the prediction log and count as incorrect in headline metrics. This runner does not cache or resume API calls; use a new output directory for each run and remember that repeating a live run can incur new charges. Partial logs survive interruptions.

## Use your own labeled data

Create `labels.json`:

```json
{
  "billing": "Payments, invoices, billing disputes and refunds.",
  "technical": "Software failures, outages and integration issues."
}
```

Create JSONL with one record per line:

```json
{"id":"ticket-001","text":"I was charged twice.","label":"billing","split":"train","group_id":"conversation-001"}
{"id":"ticket-002","text":"The application crashes when opening a file.","label":"technical","split":"test","group_id":"conversation-002"}
```

Supply enough records of **every class in every split** (`train`, `validation`, `test`). Include a `group_id` for related messages or paraphrases; the importer rejects a group spanning multiple splits. Make the splits yourself so they reflect your intended evaluation, such as a temporal holdout. The two lines above only illustrate the format, not a complete dataset.

```bash
jev-bench prepare --config configs/newsgroups.json --input my-data.jsonl \
  --labels labels.json --out data/custom
jev-bench run --config configs/newsgroups.json --data data/custom --out runs/custom
```

For custom data, all supplied records are retained in their supplied split; the config's validation/test counts do not subsample them. Adjust training budgets to fit your data. Normalized duplicate text, overlapping group IDs, missing categories, and changed dataset contents are rejected. Truncation still follows `max_chars`; inspect examples where the deciding evidence may have been removed. This project accepts JSONL, not CSV.

## Outputs and interpretation

Each new run directory contains:

- `report.md`: comparison table, plots and caveats.
- `learning_curves.png`: test macro-F1 against number of training labels; variability across training-subset seeds.
- `calibration.png`: reliability and risk–coverage plots for each model at its largest training budget and first seed, plus Jev if run.
- `results.json`: accuracy, macro-F1, per-class recall, confusion matrices, Brier score, log loss, ECE, conditional accuracy bootstrap intervals, timing, tuning choices and training IDs.
- `predictions.jsonl`: per-example distributions, predicted/true labels, latency, failures, attempts and Jev responses/usage.
- `metadata.json`: exact configuration, dataset hash, label order/descriptions, software versions, source hash, machine details, prompt and run status.

Hyperparameters are selected by **validation macro-F1**, never test performance. The selected model is not refit on validation data, preserving the stated training budget. Validation labels are an additional labeled-data cost, disclosed separately. All preprocessing is fitted on training data; the SVM fits TF-IDF separately within each calibration fold.

Macro-F1 and accuracy include failed requests as errors. Calibration diagnostics use valid probability vectors only. Multiclass Brier score is the mean of the **sum** of squared class-probability errors (range 0–2); log loss uses scikit-learn's numerical clipping. ECE uses ten equal-width bins and is sample/bin dependent. Bootstrap accuracy intervals condition on one trained model; they are not uncertainty over the entire experiment. Risk–coverage curves group tied probabilities and are descriptive, not a way to tune thresholds on test data.

Jev's `confidence` is stored separately from maximum class probability. Calibration is computed from the **option probabilities**. A distribution summary is not automatically a probability of correctness.

Inference latency is measured one example at a time. Local timings include feature transformation after one warm-up; Jev timings include network latency and its first request. Throughput reflects this sequential setup. Do not present it as an equal-hardware comparison or extrapolate it to saturated batch throughput. Model execution order is shuffled deterministically; Jev runs after local models. Repeat runs at different times if latency is central to your article.

## A good first article from this experiment

Use the learning curve to ask where supervised models catch up, then inspect disagreements. Show at least one case where each approach fails and report per-class recall rather than only an aggregate. Keep the synthetic demo out of the conclusions. Do not claim a statistically established winner from a small score difference or select the best test seed.

This initial project focuses on the classifier comparison. It does not yet implement taxonomy changes, neural embeddings, a generative LLM comparator, or uncertainty perturbation experiments.

## Included example results

`example-results/newsgroups-classical/report.md` contains a completed 54-configuration classical run on 400 test examples. `example-results/demo/report.md` shows the offline smoke test. Neither includes a live Jev run. The source snapshots match the benchmark source hashes in their metadata; reports were subsequently regenerated with clearer plot titles, recorded separately as `report_source_sha256`. The current implementation also handles ties in Jev’s returned choice explicitly.

## Tests

```bash
python -m pytest -q
```

Tests exercise all six model paths, preparation of every public dataset through a mocked Hub loader, dataset integrity, group leakage, nested training subsets, calibration ties, failed-request accounting, and the Jev HTTP contract/retry behavior through a mock transport. Mocked API tests do not validate live service quality or availability.

## Sources

- [TypeSafe HTTP API](https://docs.typesafe.ai/api)
- [TypeSafe confidence semantics](https://docs.typesafe.ai/confidence)
- [scikit-learn: 20 Newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)
- [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) and [DBpedia-14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) — Zhang, Zhao & LeCun, [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)
- [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) — Maas et al., [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015/)
- [Emotion](https://huggingface.co/datasets/dair-ai/emotion) — Saravia et al., [CARER](https://aclanthology.org/D18-1404/)
- [Banking77](https://huggingface.co/datasets/PolyAI/banking77) — Casanueva et al., [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807)
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [Guo et al.: On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
