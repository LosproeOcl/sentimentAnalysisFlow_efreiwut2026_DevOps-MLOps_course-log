"""Train four aspect-category sentiment classifiers and log to MLflow.

One classifier per aspect (food/service/price/ambience). Each is a
TF-IDF + LogisticRegression pipeline trained on reviews that mention that
aspect, then applied to any input review at inference time.

Modes:
    baseline : fixed unigram TF-IDF + LogisticRegression
    tuned    : fixed character 1-2 gram TF-IDF + balanced LogisticRegression
    search   : per-aspect randomized cross-validation search around the tuned
               configuration; best pipeline is selected on macro-F1

Usage:
    python -m ml.train --data-version v2 --mode baseline
    python -m ml.train --data-version v2 --mode tuned
    python -m ml.train --data-version v2 --mode search --search-iter 8 --cv-folds 3

Produces:
    models/model.joblib            (dict {aspect: pipeline}, loaded by the app)
    data/demo.csv                  (reserved demo reviews, excluded from evaluation)
    mlruns/...                     (MLflow run history)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .aspects import ASPECTS, BINARY_LABELS, LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "processed")
DEFAULT_MODEL_PATH = os.path.join(ROOT, "models", "model.joblib")
DEFAULT_DEMO_PATH = os.path.join(ROOT, "data", "demo.csv")
EXPERIMENT_NAME = "absa_restaurant_aspects"

MODEL_MODES = ("baseline", "tuned", "search")

# Search space for --mode search. Keep it small enough to run on CPU.
# The number of candidates is 2 * 3 * 2 * 4 * 2 = 96; --search-iter samples it.
SEARCH_SPACE: dict[str, list[Any]] = {
    "tfidf__ngram_range": [(1, 2), (1, 3)],
    "tfidf__min_df": [2, 3, 5],
    "tfidf__max_df": [1.0, 0.95],
    "clf__C": [0.25, 0.5, 1.0, 2.0],
    "clf__class_weight": ["balanced", None],
}


@dataclass
class AspectTrainingResult:
    """Fitted pipeline plus the metrics needed for MLflow and reports."""

    aspect: str
    pipeline: Pipeline
    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]
    best_params: dict[str, Any] = field(default_factory=dict)
    cv_macro_f1: float | None = None
    n_train: int = 0
    n_eval: int = 0


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


def build_pipeline(mode: str, seed: int = 42) -> Pipeline:
    """Build the (untrained) TF-IDF + LogisticRegression pipeline."""
    if mode == "baseline":
        vectorizer = TfidfVectorizer(min_df=2, token_pattern=r"\S+")
        classifier = LogisticRegression(max_iter=1000, random_state=seed)
    elif mode in ("tuned", "search"):
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+"
        )
        classifier = LogisticRegression(
            max_iter=2000, C=0.5, class_weight="balanced", random_state=seed
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def _pipeline_params(pipeline: Pipeline) -> dict[str, Any]:
    """Return the effective hyperparameters of a fitted baseline/tuned pipeline."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    return {
        "tfidf__ngram_range": tfidf.ngram_range,
        "tfidf__min_df": tfidf.min_df,
        "tfidf__max_df": tfidf.max_df,
        "clf__C": clf.C,
        "clf__class_weight": clf.class_weight,
    }


def _effective_folds(labels: pd.Series, requested: int) -> int:
    """Reduce the fold count when a class has fewer samples than requested."""
    counts = labels.value_counts()
    if len(counts) < 2:
        return 0
    return max(0, min(requested, int(counts.min())))


def tune_pipeline(
    train_sub: pd.DataFrame,
    seed: int = 42,
    cv_folds: int = 3,
    search_iter: int = 8,
    n_jobs: int = -1,
    search_space: dict[str, list[Any]] | None = None,
) -> tuple[Pipeline, dict[str, Any], float | None]:
    """Select per-aspect hyperparameters with randomized cross-validation.

    Returns the refitted best pipeline, its parameters, and the mean CV
    macro-F1. The tuned default is also cross-validated, so the search only
    replaces it when it improves CV macro-F1. When there are too few samples
    for CV, falls back to the tuned defaults and returns a ``None`` CV score.
    """
    base = build_pipeline("tuned", seed)
    folds = _effective_folds(train_sub["label"], cv_folds)
    if folds < 2:
        base.fit(train_sub["text"], train_sub["label"])
        return base, _pipeline_params(base), None

    space = search_space or SEARCH_SPACE
    n_candidates = 1
    for values in space.values():
        n_candidates *= max(1, len(values))
    n_iter = max(1, min(search_iter, n_candidates))

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=space,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        random_state=seed,
        refit=True,
        error_score="raise",
    )
    search.fit(train_sub["text"], train_sub["label"])

    # Guard against CV sampling missing the tuned default: keep the search
    # result only when it actually beats the tuned configuration on CV.
    default_pipeline = build_pipeline("tuned", seed)
    default_scores = cross_val_score(
        default_pipeline,
        train_sub["text"],
        train_sub["label"],
        cv=cv,
        scoring="f1_macro",
        n_jobs=n_jobs,
    )
    default_score = float(default_scores.mean())
    if default_score > float(search.best_score_) + 1e-9:
        default_pipeline.fit(train_sub["text"], train_sub["label"])
        return default_pipeline, _pipeline_params(default_pipeline), default_score

    return search.best_estimator_, dict(search.best_params_), float(search.best_score_)


def evaluate_pipeline(
    pipeline: Pipeline, eval_set: pd.DataFrame
) -> tuple[float, float, dict[str, float]]:
    """Evaluate on binary positive/negative rows and return macro-F1, accuracy, per-class F1."""
    y_true = eval_set["label"]
    y_pred = pipeline.predict(eval_set["text"])
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", labels=BINARY_LABELS, zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=BINARY_LABELS, output_dict=True, zero_division=0
    )
    per_class = {label: float(report[label]["f1-score"]) for label in BINARY_LABELS}
    return float(macro_f1), float(accuracy), per_class


def reserve_demo(test: pd.DataFrame, per_class: int = 8, seed: int = 42) -> pd.Series:
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


def train_one_aspect(
    train: pd.DataFrame,
    eval_set: pd.DataFrame,
    aspect: str,
    mode: str,
    seed: int = 42,
    cv_folds: int = 3,
    search_iter: int = 8,
    n_jobs: int = -1,
    search_space: dict[str, list[Any]] | None = None,
) -> AspectTrainingResult | None:
    """Train and evaluate a single aspect classifier (binary pos/neg)."""
    sub = train[(train["aspect"] == aspect) & (train["label"] != "neutral")]
    ev = eval_set[(eval_set["aspect"] == aspect) & (eval_set["label"] != "neutral")]
    if sub.empty or ev.empty or sub["label"].nunique() < 2:
        return None

    best_params: dict[str, Any] = {}
    cv_macro_f1: float | None = None
    if mode == "search":
        pipeline, best_params, cv_macro_f1 = tune_pipeline(
            sub,
            seed=seed,
            cv_folds=cv_folds,
            search_iter=search_iter,
            n_jobs=n_jobs,
            search_space=search_space,
        )
    else:
        pipeline = build_pipeline(mode, seed)
        pipeline.fit(sub["text"], sub["label"])
        best_params = _pipeline_params(pipeline)

    macro_f1, accuracy, per_class = evaluate_pipeline(pipeline, ev)
    return AspectTrainingResult(
        aspect=aspect,
        pipeline=pipeline,
        macro_f1=macro_f1,
        accuracy=accuracy,
        per_class_f1=per_class,
        best_params=best_params,
        cv_macro_f1=cv_macro_f1,
        n_train=len(sub),
        n_eval=len(ev),
    )


def _param_value(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return "-".join(str(v) for v in value)
    if value is None:
        return "none"
    return str(value)


def log_aspect_result(result: AspectTrainingResult) -> None:
    """Log one aspect's metrics and best hyperparameters to the active MLflow run."""
    import mlflow

    mlflow.log_metric(f"{result.aspect}_macro_f1", result.macro_f1)
    mlflow.log_metric(f"{result.aspect}_accuracy", result.accuracy)
    for label, value in result.per_class_f1.items():
        mlflow.log_metric(f"{result.aspect}_{label}_f1", value)
    if result.cv_macro_f1 is not None:
        mlflow.log_metric(f"{result.aspect}_cv_macro_f1", result.cv_macro_f1)
    for key, value in result.best_params.items():
        mlflow.log_param(f"{result.aspect}_best_{key.replace('.', '_')}", _param_value(value))


def log_aspect_sizes(result: AspectTrainingResult) -> None:
    """Log the evaluated train/eval row counts for one aspect."""
    import mlflow

    mlflow.log_param(f"{result.aspect}_n_train", result.n_train)
    mlflow.log_param(f"{result.aspect}_n_eval", result.n_eval)


def configure_tracking(args: argparse.Namespace) -> None:
    """Point MLflow at DagsHub (via env vars) or a custom/local tracking URI.

    To track to DagsHub, set:
        DAGSHUB_REPO_OWNER / DAGSHUB_REPO_NAME
        MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
    """
    import mlflow

    owner = os.environ.get("DAGSHUB_REPO_OWNER")
    name = os.environ.get("DAGSHUB_REPO_NAME")
    if owner and name:
        import dagshub

        dagshub.init(repo_owner=owner, repo_name=name, mlflow=True)
    elif args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)


def main(argv: list[str] | None = None) -> int:
    import mlflow
    import mlflow.sklearn

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--mode", default="baseline", choices=MODEL_MODES)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--demo-path", default=DEFAULT_DEMO_PATH)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--search-iter", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)

    configure_tracking(args)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(ROOT, args.data_dir)
    model_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(ROOT, args.model_path)
    demo_path = args.demo_path if os.path.isabs(args.demo_path) else os.path.join(ROOT, args.demo_path)

    train = pd.read_csv(os.path.join(data_dir, f"train_{args.data_version}.csv"))
    val = pd.read_csv(os.path.join(data_dir, f"val_{args.data_version}.csv"))
    test = pd.read_csv(os.path.join(data_dir, f"test_{args.data_version}.csv"))
    train = pd.concat([train, val], ignore_index=True)

    demo_texts = reserve_demo(test, seed=args.seed)
    demo = build_aspect_labels(test[test["text"].isin(demo_texts)])
    os.makedirs(os.path.dirname(os.path.abspath(demo_path)), exist_ok=True)
    demo.to_csv(demo_path, index=False)
    eval_set = test[~test["text"].isin(demo_texts)].reset_index(drop=True)

    with mlflow.start_run(run_name=f"{args.mode}-{args.data_version}"):
        mlflow.log_param("mode", args.mode)
        mlflow.log_param("data_version", args.data_version)
        mlflow.log_param("git_sha", git_sha())
        mlflow.log_param(
            "train_checksum",
            data_checksum(os.path.join(data_dir, f"train_{args.data_version}.csv")),
        )
        mlflow.log_param("train_size", len(train))
        mlflow.log_param("eval_size", len(eval_set))
        mlflow.log_param("demo_size", len(demo))
        mlflow.log_param("seed", args.seed)
        if args.mode == "search":
            mlflow.log_param("cv_folds", args.cv_folds)
            mlflow.log_param("search_iter", args.search_iter)
            mlflow.log_param(
                "search_space",
                json.dumps({key: [str(v) for v in values] for key, values in SEARCH_SPACE.items()}),
            )

        models: dict[str, Pipeline] = {}
        for aspect in ASPECTS:
            result = train_one_aspect(
                train,
                eval_set,
                aspect,
                args.mode,
                seed=args.seed,
                cv_folds=args.cv_folds,
                search_iter=args.search_iter,
                n_jobs=args.n_jobs,
            )
            if result is None:
                print(f"  [{aspect}] skipped: not enough positive/negative training rows")
                continue
            log_aspect_result(result)
            log_aspect_sizes(result)
            models[aspect] = result.pipeline
            cv_note = f" cv_macro_f1={result.cv_macro_f1:.4f}" if result.cv_macro_f1 is not None else ""
            print(
                f"  [{aspect}] {args.mode}: macro_f1={result.macro_f1:.4f} "
                f"accuracy={result.accuracy:.4f} (n={result.n_eval}){cv_note}"
            )
            if args.mode == "search":
                print(f"    best params: {result.best_params}")

        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        joblib.dump(models, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

        # Log each aspect model with its native flavor and register it in the
        # MLflow Model Registry so DagsHub recognizes it under "Models".
        for aspect in ASPECTS:
            if aspect not in models:
                continue
            info = mlflow.sklearn.log_model(models[aspect], f"models/{aspect}")
            try:
                mlflow.register_model(info.model_uri, f"absa_restaurant_{aspect}")
                print(f"  [register] {aspect} -> absa_restaurant_{aspect}")
            except Exception as exc:  # noqa: BLE001 - registry is best-effort
                print(f"  [register] {aspect} failed: {exc}")

    print(f"[train] {args.mode}-{args.data_version}: saved {len(models)} aspect models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
