"""Training, evaluation and inference for classical complaint classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from .language import detect_language
from .routing import route_department
from .text import normalize_text


def build_pipeline() -> Pipeline:
    """Combine word and character TF-IDF; character n-grams handle dialect/spelling."""
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2), min_df=1, max_df=0.98, sublinear_tf=True)),
            ("char", TfidfVectorizer(preprocessor=normalize_text, analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ]
    )
    return Pipeline(
        [
            ("tfidf", features),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear", random_state=42)),
        ]
    )


@dataclass
class ModelBundle:
    models: dict[str, Pipeline]
    metrics: dict[str, dict[str, Any]]
    trained_rows: int

    def predict(self, text: str) -> dict[str, Any]:
        if not str(text).strip():
            raise ValueError("Complaint text cannot be empty.")
        result: dict[str, Any] = {"language": detect_language(text)}
        for target, model in self.models.items():
            probabilities = model.predict_proba([text])[0]
            index = int(probabilities.argmax())
            result[target] = str(model.classes_[index])
            result[f"{target}_confidence"] = float(probabilities[index])
        result["department"] = route_department(result["topic"], result["urgency"])
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        return joblib.load(path)


def train_models(df: pd.DataFrame, test_size: float = 0.25) -> ModelBundle:
    required = {"text", "topic", "sentiment", "urgency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")

    train_idx, test_idx = train_test_split(
        range(len(df)), test_size=test_size, random_state=42, stratify=df["topic"]
    )
    models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, Any]] = {}

    for target in ("topic", "sentiment", "urgency"):
        model = build_pipeline()
        model.fit(df.iloc[train_idx]["text"], df.iloc[train_idx][target])
        predicted = model.predict(df.iloc[test_idx]["text"])
        labels = sorted(df[target].unique().tolist())
        report = classification_report(
            df.iloc[test_idx][target], predicted, labels=labels, output_dict=True, zero_division=0
        )
        metrics[target] = {
            "accuracy": float(accuracy_score(df.iloc[test_idx][target], predicted)),
            "macro_f1": float(f1_score(df.iloc[test_idx][target], predicted, average="macro")),
            "labels": labels,
            "confusion_matrix": confusion_matrix(df.iloc[test_idx][target], predicted, labels=labels).tolist(),
            "report": report,
            "test_rows": len(test_idx),
        }
        models[target] = model
    return ModelBundle(models=models, metrics=metrics, trained_rows=len(train_idx))

