# Current production models

- **Run**: `tuned-v2` (see `ml/README_experiments.md`)
- **Data version**: `v2` (`data/processed/train_v2.csv`)
- **Model artifact**: `models/model.joblib` (dict of 4 TF-IDF + LR pipelines)
- **MLflow registry**:
  - `absa_restaurant` — unified TF-IDF + LR model (production)
  - `absa_restaurant_lstm` — BiLSTM alternative (underperforms on current small data)
- **Optimized metric**: macro-F1 (binary positive/negative)
