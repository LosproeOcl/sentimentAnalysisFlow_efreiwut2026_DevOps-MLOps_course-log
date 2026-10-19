# Aspect Model Training Optimization

This branch (`training-optimize`) focuses only on improving the aspect-sentiment
training pipeline. Application code, deployment and the full project README stay
on `main`.

## What is optimized

The production model is four binary TF-IDF + LogisticRegression pipelines
(positive vs negative per aspect; the neutral class is dropped at training
time). This branch optimizes the training and model-selection loop:

- **Per-aspect hyperparameter search** (`--mode search`): stratified
  `RandomizedSearchCV` scored by macro-F1, run independently for food / service
  / price / ambience.
- **Search space**: character n-gram range, `min_df`, `max_df`, regularization
  strength (`C`), and `class_weight`.
- **No-regression guard**: the existing `tuned` configuration is also
  cross-validated; the search result is kept only when its mean CV macro-F1 is
  higher.
- **Reproducibility**: `--seed`, `--cv-folds`, `--search-iter`, `--n-jobs`.
- **Configurable paths**: `--data-dir`, `--model-path`, `--demo-path`.
- **Better MLflow logging**: per-aspect best parameters, CV macro-F1,
  train/eval sizes, held-out metrics, and per-class F1.

## Run

```bash
dvc pull data
python -m ml.train --data-version v2 --mode search --search-iter 8 --cv-folds 3
```

Compare against the fixed modes:

```bash
python -m ml.train --data-version v2 --mode baseline
python -m ml.train --data-version v2 --mode tuned
```

The selected pipelines are saved to `models/model.joblib`; each aspect is logged
and registered through MLflow. Use `mlflow ui` or DagsHub to compare runs.

## Metric

Per-aspect macro-F1 over positive/negative. Accuracy and per-class F1 are logged
for inspection, but macro-F1 is the selection metric because the classes are
imbalanced.

## Validation

- `ruff check app ml tests scripts` passes.
- `pytest`: 39 tests pass, including new training/search helper tests.
- End-to-end training with MLflow was smoke-tested on synthetic data.

## Current limitation

The real data is DVC-tracked in `efreiwut-projdvc`. This environment currently
gets `AuthorizationPermissionMismatch` (403) from that remote, so the committed
`models/model.joblib` is still the previous `tuned` artifact. After access is
granted, run the search command above and commit the refreshed model only if
held-out macro-F1 improves on every aspect.

Details: `ml/README_experiments.md` and `ml/registry/current_model.md`.
