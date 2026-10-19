# Aspect Model Training Optimization

This branch (`training-optimize`) focuses only on the aspect-sentiment training
pipeline. Application code, deployment and the full project README stay on
`main`.

## Optimization

The production model is four binary TF-IDF + LogisticRegression pipelines
(positive vs negative per aspect; neutral is dropped at training time). This
branch optimizes the training and model-selection loop:

- `--mode search`: stratified `RandomizedSearchCV` per aspect, scored by
  macro-F1.
- Search space: character n-gram range, `min_df`, `max_df`, regularization
  strength (`C`), and `class_weight`.
- No-regression guard: the existing `tuned` configuration is also
  cross-validated; the search result replaces it only when mean CV macro-F1 is
  higher.
- Reproducibility: `--seed`, `--cv-folds`, `--search-iter`, `--n-jobs`.
- Configurable paths: `--data-dir`, `--model-path`, `--demo-path`.
- MLflow logs: best parameters, CV macro-F1, train/eval sizes, held-out metrics,
  and per-class F1.

```bash
dvc pull data
python -m ml.train --data-version v2 --mode search --search-iter 8 --cv-folds 3
```

Compare with the fixed modes:

```bash
python -m ml.train --data-version v2 --mode baseline
python -m ml.train --data-version v2 --mode tuned
```

## Performance

Per-aspect macro-F1 on the v2 test split (positive/negative):

| Aspect | Baseline (before) | Tuned (previous) | Search (this branch) |
|---|---:|---:|---:|
| food | 0.4219 | 0.6609 | pending* |
| service | 0.5577 | 0.7039 | pending* |
| price | 0.5333 | 0.8518 | pending* |
| ambience | 0.6065 | 0.7164 | pending* |

\* The `search` column needs the DVC-tracked dataset. The current Azure account
gets `AuthorizationPermissionMismatch` (403) from `efreiwut-projdvc`, so the
committed `models/model.joblib` is still the `tuned` artifact. After access is
granted, run the search command and fill in the held-out results. Do not commit
a new artifact unless every aspect improves.

Baseline/tuned numbers come from `ml/README_experiments.md`. The later root
README run reported 0.738 / 0.784 / 0.786 / 0.760 for tuned; the table above
uses the baseline/tuned pair from the same experiment log for a fair
before/after comparison.

Validation: `ruff check app ml tests scripts` passes; `pytest` 39 tests pass;
end-to-end MLflow training was smoke-tested on synthetic data.
