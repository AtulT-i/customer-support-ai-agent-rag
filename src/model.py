from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.text import normalize_for_matching


def build_intent_pipeline(random_seed: int) -> Pipeline:
    features = FeatureUnion(
        (
            (
                "word",
                TfidfVectorizer(
                    preprocessor=normalize_for_matching,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=50_000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    preprocessor=normalize_for_matching,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=50_000,
                ),
            ),
        )
    )
    classifier = LogisticRegression(
        max_iter=1_000,
        class_weight="balanced",
        random_state=random_seed,
    )
    return Pipeline((("features", features), ("classifier", classifier)))


def train_intent_model(
    labels_csv: Path,
    output_path: Path,
    random_seed: int,
    allow_weak_labels: bool = False,
) -> Pipeline:
    labels = pd.read_csv(labels_csv, dtype=str, keep_default_na=False)
    text_column = "text" if "text" in labels.columns else "customer_text"
    required = {text_column, "intent"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")

    if not allow_weak_labels:
        if "reviewed" not in labels.columns or not labels["reviewed"].str.strip().str.casefold().eq("yes").all():
            raise ValueError("Every training row must be explicitly reviewed=yes.")
        if "label_source" in labels.columns and not labels["label_source"].eq("human").all():
            raise ValueError("Reviewed training cannot contain non-human labels.")

    normalized = labels[text_column].map(normalize_for_matching)
    if normalized.eq("").any() or labels["intent"].str.strip().eq("").any():
        raise ValueError("Training text and intents must not be blank.")
    if not allow_weak_labels and normalized.duplicated().any():
        raise ValueError("Deduplicate normalized training text before training.")
    if len(labels) < 100:
        raise ValueError(
            "At least 100 reviewed labels are required. Use --allow-weak only "
            "for a provisional demo, never for headline results."
        )
    if labels["intent"].nunique() < 2:
        raise ValueError("Training data must contain at least two intents.")

    pipeline = build_intent_pipeline(random_seed)
    pipeline.fit(labels[text_column], labels["intent"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "label_source": (
                "weak_provisional" if allow_weak_labels else "human_reviewed"
            ),
            "training_examples": len(labels),
            "training_data_sha256": hashlib.sha256(labels_csv.read_bytes()).hexdigest(),
        },
        output_path,
    )
    return pipeline


def load_intent_model(path: Path) -> tuple[Pipeline, dict[str, object]]:
    bundle = joblib.load(path)
    return bundle["pipeline"], {
        "label_source": bundle["label_source"],
        "training_examples": bundle["training_examples"],
        "training_data_sha256": bundle.get("training_data_sha256"),
    }


def predict_intent(model: Pipeline, text: str) -> tuple[str, float, float]:
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_
    ranked = probabilities.argsort()[::-1]
    top = int(ranked[0])
    second = int(ranked[1]) if len(ranked) > 1 else top
    return str(classes[top]), float(probabilities[top]), float(probabilities[second])
