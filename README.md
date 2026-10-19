# Training Optimization (experimental)

This branch is a focused experiment; the full project documentation lives on
`main`.

- `--mode search`: per-aspect TF-IDF + LR hyperparameter search with stratified
  macro-F1 CV.
- Default search space now uses character `ngram_range=(1,2)` only; `(1,3)`
  produced too many features and was too slow on CPU.
- The tuned configuration is always cross-validated; search only replaces it
  when CV macro-F1 improves.
- Changes: `ml/train.py`, `tests/unit/test_train.py`.
- Run: `python -m ml.train --data-version v2 --mode search --search-iter 8 --cv-folds 3`

| Metric (official ASAP split) | Tuned | Search | Delta |
|---|---:|---:|---:|
| Average macro-F1 | 0.7672 | 0.7824 | +0.0152 (~+2.0%) |

All four aspects were non-regressing, but the gain is marginal and below the
merge threshold. The experiment is not merged; production remains `tuned`.

Validation: `ruff` passes; `pytest` 39 tests pass.
