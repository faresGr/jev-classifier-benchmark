# Results accompanying the Jev comparison

[Read the article](https://faresgr.github.io/blog/2026/how-many-labels-is-a-description-worth/).

These are the actual saved outputs for AG News, Banking77 and Emotion, not simulated examples. Each dataset directory contains:

- `predictions.jsonl.gz`: losslessly compressed original predictions for every model, budget and training seed. Includes Jev's raw API response; does not include input texts or API credentials.
- `metadata.json`: original run configuration, exact category descriptions, test IDs, frozen dataset hashes, resolved Hub revision, prompt, software versions and timings.
- `original-results.json`: original scores, retained for audit. **Banking77 and Emotion contain obsolete parser failures here. Use `results.json` for the corrected analysis.**
- `results.json`, `report.md` and figures: corrected analysis. Original raw responses with small probability-sum rounding discrepancies are normalized; no API calls are made for rescoring.

`summary.json` contains the blog figures, confidence-threshold counts and paired comparisons. Recreate them from the repository root:

```sh
python -m pip install -e .
python scripts/publish_analysis.py
```

The script produces all corrected reports and the blog figures without downloading datasets, fitting models or contacting Jev. The compressed records remain immutable. To export a new set of local runs, use `--export-local` deliberately; this replaces published artifacts.

## Intervals and label accounting

Paired macro-F1 intervals use 5,000 class-stratified bootstrap resamples (seed 123). The same resampled test IDs are used for Jev and each of the three fitted classical seeds; their F1 scores are averaged, not their predictions. Comparators are selected by highest observed mean test F1 at the largest budget. These intervals are descriptive and conditional on those fitted/selected models: they do not adjust for model selection or represent training-sample uncertainty. Three training seeds do not constitute three independent datasets.

Training labels at maximum budget, per fitted model: AG News 4,000 (+200 validation); Banking77 1,540 (+385); Emotion 3,000 (+300). Test labels are a separate evaluation expense. Training subsets can overlap across seeds. Balanced sampling and public-data pretraining overlap limit generalization.

Confidence figures use normalized option probabilities, not Jev's separate `confidence` field. Values reported as 0.00 are finite-precision API outputs. ECE uses 10 equally spaced bins. A 0.9 threshold is a descriptive slice of the test results, not a validated operational rule.

## Repeating inference

To reconstruct the source dataset, use the `config` in that dataset's metadata and set `hub_revision` to `dataset.provenance.hub_revision` before running `jev-bench prepare`. Compare the generated dataset/label SHA256 hashes against metadata. Keep original class descriptions and source selection unchanged. Current unpinned configs are convenient for new experiments, not a guarantee of identical future downloads.

`python scripts/prompt_followup.py --out runs/prompt-followup --env-file .env` runs one predeclared wording comparison on the existing frozen `data/{ag_news,banking77,emotion}` bundles. It refuses changed data or an existing output directory. It uses exactly the original category descriptions and test IDs, changes only the task instruction for banking/emotion, and repeats AG News unchanged as a control. This makes 1,470 additional paid requests. The script does not load `.env` unless that path is explicitly passed.

This follow-up reuses a test set already inspected during the original analysis: it is post-hoc sensitivity analysis, not fresh held-out confirmation. An unchanged-prompt repeat helps reveal API variation, but does not fully separate prompt effects from timing/service variation. Do not repeatedly tune prompts on these test sets.
