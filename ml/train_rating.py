"""Train a star-rating predictor (1-5) on ASAP reviews (CPU, TF-IDF + LR).

The ASAP dataset includes a `star` column (1-5) that the aspect model ignores.
This script trains a separate rating classifier on it, logs to MLflow, and
saves `models/rating.joblib` for the app.

Usage:
    python -m ml.train_rating
"""
from __future__ import annotations

import argparse
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline

from .aspects import tokenize_zh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
MODEL_PATH = os.path.join(ROOT, "models", "rating.joblib")
EXPERIMENT = "absa_restaurant_aspects"
MODEL_NAME = "absa_restaurant_rating"

SPLITS = [("train", "train.csv"), ("val", "dev.csv"), ("test", "test.csv")]


def load_rating_data() -> pd.DataFrame:
    frames = []
    for split, fname in SPLITS:
        df = pd.read_csv(os.path.join(RAW_DIR, fname))
        sub = df[["review", "star"]].copy()
        sub["star"] = sub["star"].astype(int)
        sub["split"] = "val" if split == "val" else split
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    owner = os.environ.get("DAGSHUB_REPO_OWNER")
    name = os.environ.get("DAGSHUB_REPO_NAME")
    if owner and name:
        import dagshub

        dagshub.init(repo_owner=owner, repo_name=name, mlflow=True)
    elif args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    df = load_rating_data()
    train = df[df["split"].isin(["train", "val"])]
    test = df[df["split"] == "test"]

    X_train = [tokenize_zh(t) for t in train["review"]]
    y_train = train["star"]
    X_test = [tokenize_zh(t) for t in test["review"]]
    y_test = test["star"]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )

    with mlflow.start_run(run_name="rating-tfidf"):
        mlflow.log_param("task", "star_rating")
        mlflow.log_param("model", "TF-IDF + LogisticRegression")
        mlflow.log_param("train_size", len(train))
        mlflow.log_param("test_size", len(test))

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        mlflow.log_metric("rating_accuracy", acc)
        mlflow.log_metric("rating_mae", mae)

        info = mlflow.sklearn.log_model(pipeline, name=MODEL_NAME)
        mlflow.register_model(info.model_uri, MODEL_NAME)
        joblib.dump(pipeline, MODEL_PATH)
        print(f"[rating] accuracy={acc:.4f} mae={mae:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
