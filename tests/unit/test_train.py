import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml import train as train_mod


def _toy_frame(aspect: str = "food", n_per_class: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(n_per_class):
        rows.append({"text": f"好 吃 新 鲜 {i}", "aspect": aspect, "label": "positive"})
        rows.append({"text": f"难 吃 冷 淡 {i}", "aspect": aspect, "label": "negative"})
    return pd.DataFrame(rows)


def test_build_pipeline_modes():
    baseline = train_mod.build_pipeline("baseline")
    tuned = train_mod.build_pipeline("tuned")
    search = train_mod.build_pipeline("search")

    assert isinstance(baseline, Pipeline)
    assert isinstance(tuned, Pipeline)
    assert isinstance(search, Pipeline)
    assert baseline.named_steps["tfidf"].ngram_range == (1, 1)
    assert tuned.named_steps["tfidf"].ngram_range == (1, 2)
    assert tuned.named_steps["clf"].class_weight == "balanced"

    with pytest.raises(ValueError, match="unknown mode"):
        train_mod.build_pipeline("nope")


def test_evaluate_pipeline_returns_binary_metrics():
    frame = _toy_frame()
    pipeline = train_mod.build_pipeline("baseline")
    pipeline.fit(frame["text"], frame["label"])

    macro_f1, accuracy, per_class = train_mod.evaluate_pipeline(pipeline, frame)

    assert 0.0 <= macro_f1 <= 1.0
    assert 0.0 <= accuracy <= 1.0
    assert set(per_class) == {"negative", "positive"}
    assert all(0.0 <= value <= 1.0 for value in per_class.values())


def test_train_one_aspect_baseline_returns_metrics():
    frame = _toy_frame()
    result = train_mod.train_one_aspect(frame, frame, "food", "baseline")

    assert result is not None
    assert result.aspect == "food"
    assert result.n_train == 24
    assert result.n_eval == 24
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.cv_macro_f1 is None
    assert result.best_params["clf__C"] == 1.0


def test_train_one_aspect_search_tunes_hyperparameters():
    frame = _toy_frame()
    space = {"clf__C": [0.5, 1.0], "tfidf__ngram_range": [(1, 2)]}

    result = train_mod.train_one_aspect(
        frame,
        frame,
        "food",
        "search",
        seed=42,
        cv_folds=2,
        search_iter=2,
        n_jobs=1,
        search_space=space,
    )

    assert result is not None
    assert result.cv_macro_f1 is not None
    assert 0.0 <= result.cv_macro_f1 <= 1.0
    assert "clf__C" in result.best_params
    assert result.best_params["clf__C"] in (0.5, 1.0)


def test_train_one_aspect_skips_single_class():
    frame = _toy_frame().assign(label="positive")
    assert train_mod.train_one_aspect(frame, frame, "food", "baseline") is None


def test_effective_folds_is_capped_by_smallest_class():
    labels = pd.Series(["positive"] * 2 + ["negative"] * 8)
    assert train_mod._effective_folds(labels, 5) == 2
    assert train_mod._effective_folds(pd.Series(["positive"] * 8), 5) == 0
