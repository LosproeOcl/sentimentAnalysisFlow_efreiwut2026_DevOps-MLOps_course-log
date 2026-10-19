# Experiment Tracking (MLflow)

All training runs are tracked with MLflow. By default this uses the local
`mlruns/` backend; when `DAGSHUB_REPO_OWNER` / `DAGSHUB_REPO_NAME` (plus
`MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`) are set, runs and models
go to DagsHub's MLflow server (see `15-MLFlow.pdf`).

## What is logged per run

- **Code version**: `git_sha`.
- **Dataset version**: `data_version` and `train_checksum` (MD5).
- **Hyperparameters**: `mode` (baseline/tuned/search), vectorizer/classifier
  settings, and each aspect's best parameters from the search.
- **Metrics**: per-aspect `macro_f1` and `accuracy`, plus per-class (positive/negative) F1.
- **Models**: each of the four aspect classifiers is logged with the `sklearn`
  flavor and **registered** in the MLflow Model Registry.

The sentiment task is **binary (positive vs negative)** per aspect: the murky
"neutral" class is dropped at training time, which both improves macro-F1 and
yields more trustworthy confidence. At inference, predictions below a confidence
threshold are reported as "neutral" (uncertain) rather than a forced polarity.

## Experiments

| Run | Model | Data | Description |
|-----|-------|------|-------------|
| `baseline-v2` | TF-IDF (unigrams) + LogisticRegression | v2 | Simple baseline. |
| `tuned-v2` | TF-IDF (1-2 grams, sublinear) + LogisticRegression (balanced class weights, tuned `C`) | v2 | Improved model. |
| `search-v2` | Per-aspect randomized CV search around the tuned configuration | v2 | Recommended when refreshing the model. |

### Results (macro-F1 per aspect, v2, binary positive/negative)

| Aspect | baseline | tuned |
|--------|----------|-------|
| food | 0.4219 | 0.6609 |
| service | 0.5577 | 0.7039 |
| price | 0.5333 | 0.8518 |
| ambience | 0.6065 | 0.7164 |

The tuned model improves macro-F1 on every aspect.

### Per-aspect search

`--mode search` keeps the tuned pipeline as the starting point and runs a small
`RandomizedSearchCV` **per aspect** (stratified, macro-F1 scoring). The search
space covers character n-gram range, `min_df`, `max_df`, regularization strength
(`C`), and class weighting:

```bash
python -m ml.train --data-version v2 --mode search --search-iter 8 --cv-folds 3
```

Useful options:

- `--cv-folds` — stratified folds (automatically reduced when a class has fewer
  samples).
- `--search-iter` — number of sampled candidates per aspect (default 8).
- `--n-jobs` — parallel search workers (default `-1`).
- `--seed` — reproducibility seed for the split and sampler.

The selected parameters and the mean CV macro-F1 are logged per aspect, and the
best pipeline is refitted on all train+val rows before the held-out test
evaluation. The tuned default is also cross-validated, so the search only
replaces it when CV macro-F1 improves; this prevents a sampled grid from
regressing against `tuned`. The new mode replaces the previous
one-size-fits-all `C=0.5` choice with a per-aspect choice while keeping
`baseline` and `tuned` reproducible for comparison. With the v2 dataset, run the
command above to refresh the `search-v2` row in the results table.

## Metric we optimize for

**Macro-F1** (over `positive`/`negative`). The classes are imbalanced, so raw
accuracy is misleading; macro-F1 rewards the model for getting the minority class
right. Dropping the murky `neutral` class (binary task) measurably improved both
macro-F1 and the reliability of the reported confidence.

## Registered models (MLflow Model Registry)

- `absa_restaurant` — the **unified** TF-IDF + LR model (all four aspects in one
  pyfunc model).
- `absa_restaurant_lstm` — the BiLSTM model (see below).

## BiLSTM experiment

A bidirectional LSTM (embedding 128, hidden 128) was trained as a deep-learning
replacement. On this small dataset it **underperforms** the tuned TF-IDF + LR:

| Aspect | TF-IDF+LR (tuned) | BiLSTM |
|--------|-------------------|--------|
| food | 0.6609 | 0.4706 |
| service | 0.7039 | 0.6199 |
| price | 0.8518 | 0.8036 |
| ambience | 0.7164 | 0.5830 |

This is expected: deep models need far more training data. The BiLSTM is kept as
a registered alternative (`absa_restaurant_lstm`); the TF-IDF + LR remains the
production model until a larger dataset (e.g. the full Chinese MAMS corpus) is
available.

## Production-worthy run

The **tuned** run is the current production baseline: it improves macro-F1 on
all four aspects. Its combined model is saved to `models/model.joblib`
(referenced in `ml/registry/current_model.md`) and served by the app.

For a new dataset or a model refresh, use `--mode search` and compare the logged
`*_cv_macro_f1` and held-out `*_macro_f1` values against `tuned-v2`. Keep the
new artifact only when the held-out macro-F1 is better on every aspect.
