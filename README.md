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

Per-aspect macro-F1 on the v2 test split (positive/negative). This verification
used the official ASAP train/dev/test split (public mirror: `nd-ball/DSPN`),
processed with `ml/data_pipeline.py`; all three modes used the same split and
seed. The tuned run reproduces the project's published 0.738 / 0.784 / 0.786 /
0.760, confirming the split is comparable.

| Aspect | Baseline (before) | Tuned (previous) | Search (this branch) | Delta vs tuned |
|---|---:|---:|---:|---:|
| food | 0.6758 | 0.7384 | 0.7654 | +0.0270 |
| service | 0.7413 | 0.7840 | 0.8005 | +0.0165 |
| price | 0.7148 | 0.7864 | 0.8141 | +0.0277 |
| ambience | 0.6851 | 0.7598 | 0.7852 | +0.0254 |
| **average** | **0.7043** | **0.7672** | **0.7913** | **+0.0242** |

The search improves macro-F1 on all four aspects: +0.0242 on average
(about +3.1% relative over tuned, +12.4% over baseline). Verification budget:
`--search-iter 4 --cv-folds 2 --n-jobs 1` (the default command uses 8/3).

The search artifact was produced in a temporary directory for this verification;
the branch keeps the training code and measured results, and the committed
`models/model.joblib` is still the `tuned` model. To refresh the artifact, run
the search command and commit the resulting `models/model.joblib` (the source
data is DVC-tracked; this run used the public ASAP mirror because the Azure
remote returned 403).

Validation: `ruff check app ml tests scripts` passes; `pytest` 39 tests pass;
full baseline / tuned / search runs completed on the ASAP split and were logged
to MLflow.
