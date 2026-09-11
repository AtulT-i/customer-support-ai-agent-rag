from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.agent import SupportAgent
from src.baselines import keyword_prediction, majority_label, trivial_prediction
from src.text import normalize_for_matching


DECISIONS = {"auto_handle", "escalate"}


def _require_fields(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    for field in sorted(required):
        values = frame[field]
        if (values.isna() | values.astype(str).str.strip().eq("")).any():
            raise ValueError(f"{name} contains blank required field {field!r}.")


def _normalized_texts(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    values = frame[column].map(normalize_for_matching)
    if values.eq("").any():
        raise ValueError(f"{name} contains empty normalized text.")
    return values


def _validate_golden(
    golden: pd.DataFrame, intents: tuple[str, ...] | None = None
) -> None:
    _require_fields(
        golden,
        {"example_id", "text", "intent", "decision", "thread_id", "reviewed"},
        "Golden set",
    )
    if not 150 <= len(golden) <= 250:
        raise ValueError(
            f"Golden set must contain 150-250 labelled rows; found {len(golden)}."
        )
    if not golden["reviewed"].astype(str).str.strip().str.casefold().eq("yes").all():
        raise ValueError("Every golden-set row must be marked reviewed=yes.")
    if not golden["decision"].isin(DECISIONS).all():
        raise ValueError("Golden set contains invalid decisions.")
    if intents is not None and not golden["intent"].isin(intents).all():
        raise ValueError("Golden set contains intents outside the configured intents.")
    if golden["example_id"].astype(str).str.strip().duplicated().any():
        raise ValueError("Golden set contains duplicate example_id values.")
    if _normalized_texts(golden, "text", "Golden set").duplicated().any():
        raise ValueError("Golden set contains duplicate normalized texts.")
    if "tweet_id" in golden.columns:
        _require_fields(golden, {"tweet_id"}, "Golden set")


def _reject_overlap(
    left: pd.Series, right: pd.Series, name: str, field: str
) -> None:
    if set(left.astype(str).str.strip()) & set(right.astype(str).str.strip()):
        raise ValueError(f"{name} leakage: {field} overlaps the golden set.")


def _validate_artifacts_and_training(
    golden: pd.DataFrame,
    training: pd.DataFrame,
    training_labels_csv: Path,
    golden_csv: Path,
    agent: SupportAgent,
) -> None:
    metadata = getattr(agent, "model_metadata", None)
    if not isinstance(metadata, dict) or metadata.get("label_source") != "human_reviewed":
        raise ValueError(
            "Evaluation requires model_metadata label_source=human_reviewed; "
            "weak or old model artifacts are not eligible. Retrain on reviewed labels."
        )
    training_hash = hashlib.sha256(training_labels_csv.read_bytes()).hexdigest()
    if metadata.get("training_data_sha256") != training_hash:
        raise ValueError(
            "Model training_data_sha256 is missing or does not match the training file. "
            "Old model artifacts are not eligible; retrain on this reviewed file."
        )
    _require_fields(training, {"text", "intent", "thread_id", "reviewed"}, "Training set")
    if training.empty:
        raise ValueError("Training set must not be empty.")
    if not training["reviewed"].astype(str).str.strip().str.casefold().eq("yes").all():
        raise ValueError("Every training row must be marked reviewed=yes.")
    if "label_source" in training.columns and not training["label_source"].eq("human").all():
        raise ValueError("Every training label_source must be human.")
    if not training["intent"].isin(agent.config.intents).all():
        raise ValueError("Training set contains intents outside the configured intents.")
    if "decision" in training.columns and not training["decision"].isin(DECISIONS).all():
        raise ValueError("Training set contains invalid decisions.")
    golden_texts = _normalized_texts(golden, "text", "Golden set")
    training_texts = _normalized_texts(training, "text", "Training set")
    if training_texts.duplicated().any():
        raise ValueError("Training set contains duplicate normalized texts.")
    _reject_overlap(training_texts, golden_texts, "Training", "normalized text")
    _reject_overlap(training["thread_id"], golden["thread_id"], "Training", "thread_id")
    if "example_id" in training.columns:
        _require_fields(training, {"example_id"}, "Training set")
        _reject_overlap(training["example_id"], golden["example_id"], "Training", "example_id")
    for field in ("tweet_id", "customer_tweet_id"):
        if field in training.columns:
            _require_fields(training, {field}, "Training set")
            if "tweet_id" in golden.columns:
                _reject_overlap(training[field], golden["tweet_id"], "Training", field)

    bundle = getattr(agent, "retrieval", None)
    if not isinstance(bundle, dict) or bundle.get("version") != 2:
        raise ValueError(
            "Evaluation requires retrieval index version=2; old indexes are not eligible. "
            "Rebuild with golden exclusions before vectorizer fitting."
        )
    golden_hash = hashlib.sha256(golden_csv.read_bytes()).hexdigest()
    if bundle.get("excluded_golden_sha256") != golden_hash:
        raise ValueError(
            "Retrieval excluded_golden_sha256 is missing or does not match the golden file. "
            "Rebuild the index excluding this golden set before vectorizer fitting."
        )
    pairs = bundle.get("pairs")
    if not isinstance(pairs, pd.DataFrame):
        raise ValueError("Retrieval index must contain a pairs DataFrame; rebuild the index.")
    _require_fields(pairs, {"thread_id", "customer_tweet_id", "customer_text"}, "Retrieval index")
    _reject_overlap(pairs["thread_id"], golden["thread_id"], "Retrieval", "thread_id")
    if "tweet_id" in golden.columns:
        _reject_overlap(pairs["customer_tweet_id"], golden["tweet_id"], "Retrieval", "customer_tweet_id")
    _reject_overlap(
        _normalized_texts(pairs, "customer_text", "Retrieval index"),
        golden_texts,
        "Retrieval",
        "normalized customer text",
    )


def _system_metrics(
    name: str,
    golden: pd.DataFrame,
    predictions: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    if len(predictions) != len(golden):
        raise ValueError("There must be exactly one prediction per golden example.")
    if any(row["decision"] not in DECISIONS for row in predictions):
        raise ValueError("Predictions contain invalid decisions.")
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_intents = golden["intent"].astype(str).to_numpy()
    predicted_intents = [str(row["intent"]) for row in predictions]
    actual_escalation = golden["decision"].eq("escalate").to_numpy()
    predicted_escalation = pd.Series(
        [row["decision"] == "escalate" for row in predictions], dtype=bool
    ).to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_escalation,
        predicted_escalation,
        average="binary",
        zero_division=0,
    )
    report = classification_report(
        actual_intents,
        predicted_intents,
        output_dict=True,
        zero_division=0,
    )
    (output_dir / f"per_intent_report_{name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    labels = sorted(set(actual_intents) | set(predicted_intents))
    matrix = confusion_matrix(actual_intents, predicted_intents, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / f"confusion_{name}.csv"
    )
    false_auto = int(((~predicted_escalation) & actual_escalation).sum())
    must_escalate_count = int(actual_escalation.sum())
    auto_handle_count = int((~predicted_escalation).sum())
    return {
        "system": name,
        "accuracy": accuracy_score(actual_intents, predicted_intents),
        "macro_f1": report["macro avg"]["f1-score"],
        "escalation_precision": precision,
        "must_escalate_recall": recall,
        "escalation_f1": f1,
        "false_auto_handle_count": false_auto,
        "must_escalate_count": must_escalate_count,
        "auto_handle_count": auto_handle_count,
        "false_auto_handle_rate": (
            float(false_auto / must_escalate_count)
            if must_escalate_count
            else 0.0
        ),
        "false_auto_handle_fraction_among_auto_handled": (
            float(false_auto / auto_handle_count) if auto_handle_count else 0.0
        ),
        "auto_handle_coverage": float((~predicted_escalation).mean()),
    }


def evaluate_all(
    golden_csv: Path,
    training_labels_csv: Path,
    agent: SupportAgent,
    output_dir: Path,
) -> pd.DataFrame:
    # Read identifiers as strings (including leading zeros); never discard malformed rows.
    golden = pd.read_csv(golden_csv, dtype=str, keep_default_na=False)
    _validate_golden(golden, agent.config.intents)
    training = pd.read_csv(training_labels_csv, dtype=str, keep_default_na=False)
    _validate_artifacts_and_training(
        golden, training, training_labels_csv, golden_csv, agent
    )
    majority = majority_label(training["intent"].astype(str).tolist())

    systems: dict[str, list[dict[str, object]]] = {
        "trivial": [trivial_prediction(majority) for _ in range(len(golden))],
        "keyword": [keyword_prediction(text) for text in golden["text"]],
        "main": [agent.respond(text) for text in golden["text"]],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    prediction_rows: list[dict[str, object]] = []
    for name, predictions in systems.items():
        metrics.append(_system_metrics(name, golden, predictions, output_dir))
        for (_, source), prediction in zip(golden.iterrows(), predictions):
            # Similarity is an inspection aid, NOT a human relevance label or recall@k.
            evidence = prediction.get("evidence", [])[:3]
            evidence_ids = [item["customer_tweet_id"] for item in evidence]
            similarities = [item["similarity"] for item in evidence]
            prediction_rows.append(
                {
                    "example_id": source["example_id"],
                    "system": name,
                    "actual_intent": source["intent"],
                    "actual_decision": source["decision"],
                    "predicted_intent": prediction["intent"],
                    "predicted_decision": prediction["decision"],
                    "draft_reply": prediction["draft_reply"],
                    "retrieval_top1_id": evidence_ids[0] if evidence_ids else None,
                    "retrieval_top1_similarity": similarities[0] if similarities else None,
                    "retrieval_top3_ids": json.dumps(evidence_ids, ensure_ascii=True),
                    "retrieval_top3_similarities": json.dumps(similarities),
                    "prediction_json": json.dumps(prediction, ensure_ascii=True),
                }
            )

    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "predictions.csv", index=False
    )
    results = pd.DataFrame(metrics)
    results.to_csv(output_dir / "headline_metrics.csv", index=False)
    # Bind exported failures/results to the evaluated dataset and generated predictions.
    (output_dir / "evaluation_manifest.json").write_text(json.dumps({
        "golden_sha256": hashlib.sha256(golden_csv.read_bytes()).hexdigest(),
        "training_sha256": hashlib.sha256(training_labels_csv.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256((output_dir / "predictions.csv").read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256((output_dir / "headline_metrics.csv").read_bytes()).hexdigest(),
        "systems": list(systems), "golden_examples": len(golden),
        "note": "Input/output provenance, not independent certification of annotation quality.",
    }, indent=2), encoding="utf-8")
    return results


def annotation_agreement(labels_csv: Path) -> dict[str, float]:
    labels = pd.read_csv(labels_csv).dropna(
        subset=["example_id", "annotator", "intent", "decision"]
    )
    duplicated = labels[labels.duplicated("example_id", keep=False)]
    paired = duplicated.groupby("example_id").filter(lambda group: len(group) == 2)
    if paired.empty:
        raise ValueError("No examples with exactly two annotations were found.")
    first = paired.groupby("example_id").nth(0)
    second = paired.groupby("example_id").nth(1)
    return {
        "intent_cohen_kappa": float(
            cohen_kappa_score(first["intent"], second["intent"])
        ),
        "decision_cohen_kappa": float(
            cohen_kappa_score(first["decision"], second["decision"])
        ),
        "double_annotated_examples": float(len(first)),
    }
