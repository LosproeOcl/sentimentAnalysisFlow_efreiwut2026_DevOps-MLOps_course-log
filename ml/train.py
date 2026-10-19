"""Train four aspect-category sentiment classifiers and log to MLflow.

One classifier per aspect (food/service/price/ambience). Each is a
TF-IDF + LogisticRegression pipeline trained on reviews that mention that
aspect, then applied to any input review at inference time.

Usage:
    python ml/train.py --data-version v2 --mode baseline
    python ml/train.py --data-version v2 --mode tuned

Produces:
    models/model.joblib            (dict {aspect: pipeline}, loaded by the app)
    data/demo.csv                  (reserved demo reviews, excluded from evaluation)
    mlruns/...                     (MLflow run history)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from .aspects import ASPECTS, BINARY_LABELS, LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")
MODEL_PATH = os.path.join(ROOT, "models", "model.joblib")
DEMO_PATH = os.path.join(ROOT, "data", "demo.csv")

EXPERIMENT_NAME = "absa_restaurant_aspects"


def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def data_checksum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def build_pipeline(mode: str) -> Pipeline:
    if mode == "baseline":
        vectorizer = TfidfVectorizer(min_df=2, token_pattern=r"\S+")
        classifier = LogisticRegression(max_iter=1000, random_state=42)
    elif mode == "tuned":
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+")
        classifier = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def reserve_demo(test: pd.DataFrame, per_class: int = 8, seed: int = 42) -> pd.DataFrame:
    """Reserve a balanced set of whole reviews (by text) for the interactive demo."""
    texts = test["text"].drop_duplicates()
    samples = []
    for label in LABELS:
        candidate = texts[texts.isin(test[test["label"] == label]["text"])].drop_duplicates()
        n = min(per_class, len(candidate))
        samples.append(candidate.sample(n=n, random_state=seed))
    return pd.Series(pd.concat(samples).unique()).rename("text")


def build_aspect_labels(test: pd.DataFrame) -> pd.DataFrame:
    """Pivot exploded (text, aspect, label) rows into one row per text."""
    piv = test.pivot_table(index="text", columns="aspect", values="label", aggfunc="first")
    return piv.reindex(columns=ASPECTS).reset_index()


def train_one_aspect(train: pd.DataFrame, eval_set: pd.DataFrame, aspect: str, mode: str):
    """Train and evaluate a single aspect classifier (binary pos/neg); log metrics."""
    sub = train[(train["aspect"] == aspect) & (train["label"] != "neutral")]
    ev = eval_set[(eval_set["aspect"] == aspect) & (eval_set["label"] != "neutral")]
    if sub.empty or ev.empty:
        return None

    pipeline = build_pipeline(mode)
    pipeline.fit(sub["text"], sub["label"])

    y_true = ev["label"]
    y_pred = pipeline.predict(ev["text"])
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=BINARY_LABELS)
    acc = accuracy_score(y_true, y_pred)

    mlflow.log_metric(f"{aspect}_macro_f1", macro_f1)
    mlflow.log_metric(f"{aspect}_accuracy", acc)
    report = classification_report(y_true, y_pred, labels=BINARY_LABELS, output_dict=True)
    for label in BINARY_LABELS:
        mlflow.log_metric(f"{aspect}_{label}_f1", report[label]["f1-score"])

    print(f"  [{aspect}] {mode}: macro_f1={macro_f1:.4f} accuracy={acc:.4f} (n={len(ev)})")
    return pipeline


def configure_tracking(args) -> None:
    """Point MLflow at DagsHub (via env vars) or a custom/local tracking URI.

    To track to DagsHub, set:
        DAGSHUB_REPO_OWNER / DAGSHUB_REPO_NAME
        MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
    """
    owner = os.environ.get("DAGSHUB_REPO_OWNER")
    name = os.environ.get("DAGSHUB_REPO_NAME")
    if owner and name:
        import dagshub

        dagshub.init(repo_owner=owner, repo_name=name, mlflow=True)
    elif args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--mode", default="baseline", choices=["baseline", "tuned"])
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    configure_tracking(args)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train = pd.read_csv(os.path.join(DATA_DIR, f"train_{args.data_version}.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, f"val_{args.data_version}.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, f"test_{args.data_version}.csv"))
    train = pd.concat([train, val], ignore_index=True)

    demo_texts = reserve_demo(test)
    demo = build_aspect_labels(test[test["text"].isin(demo_texts)])
    demo.to_csv(DEMO_PATH, index=False)
    eval_set = test[~test["text"].isin(demo_texts)].reset_index(drop=True)

    with mlflow.start_run(run_name=f"{args.mode}-{args.data_version}"):
        mlflow.log_param("mode", args.mode)
        mlflow.log_param("data_version", args.data_version)
        mlflow.log_param("git_sha", git_sha())
        mlflow.log_param(
            "train_checksum",
            data_checksum(os.path.join(DATA_DIR, f"train_{args.data_version}.csv")),
        )
        mlflow.log_param("train_size", len(train))
        mlflow.log_param("eval_size", len(eval_set))
        mlflow.log_param("demo_size", len(demo))

        models: dict[str, Pipeline] = {}
        for aspect in ASPECTS:
            pipe = train_one_aspect(train, eval_set, aspect, args.mode)
            if pipe is not None:
                models[aspect] = pipe

        joblib.dump(models, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH, artifact_path="models")

        # Log each aspect model with its native flavor and register it in the
        # MLflow Model Registry so DagsHub recognizes it under "Models".
        for aspect in ASPECTS:
            if aspect not in models:
                continue
            info = mlflow.sklearn.log_model(models[aspect], name=f"absa_restaurant_{aspect}")
            try:
                mlflow.register_model(info.model_uri, f"absa_restaurant_{aspect}")
                print(f"  [register] {aspect} -> absa_restaurant_{aspect}")
            except Exception as exc:  # noqa: BLE001 - registry is best-effort
                print(f"  [register] {aspect} failed: {exc}")

    print(f"[train] {args.mode}-{args.data_version}: saved {len(models)} aspect models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
