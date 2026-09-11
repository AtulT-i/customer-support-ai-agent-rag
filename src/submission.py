"""Submission helpers: explicit human work and measured failures, never synthetic truth."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.human_scoring import _atomic_save
from src.text import normalize_for_matching


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def annotation_progress(queue: pd.DataFrame, intents: tuple[str, ...], golden_size: int = 200) -> dict:
    if not 150 <= golden_size <= 250:
        raise ValueError("Golden size must be between 150 and 250.")
    required = {"example_id", "text", "thread_id", "intent", "decision", "reviewed"}
    if missing := required - set(queue.columns):
        raise ValueError(f"Annotation queue missing columns: {sorted(missing)}")
    reviewed = queue[queue["reviewed"].str.strip().str.casefold().eq("yes")].copy()
    valid = reviewed["intent"].isin(intents) & reviewed["decision"].isin(["auto_handle", "escalate"])
    for field in ("example_id", "thread_id", "text"):
        valid &= reviewed[field].str.strip().ne("")
    reviewed["_text"] = reviewed["text"].map(normalize_for_matching)
    valid &= reviewed["_text"].ne("")
    usable = reviewed[valid].drop_duplicates("_text").drop_duplicates("thread_id")
    counts = usable["intent"].value_counts().to_dict()
    rare = {key: value for key, value in counts.items() if value < 2}
    missing_intents = sorted(set(intents) - set(counts))
    return {
        "queue_rows": len(queue), "reviewed_rows": len(reviewed),
        "invalid_reviewed_rows": int((~valid).sum()),
        "usable_unique_thread_rows": len(usable), "intent_counts": counts,
        "golden_size": golden_size, "minimum_reviewed_rows": golden_size + 100,
        "additional_usable_rows_needed": max(0, golden_size + 100 - len(usable)),
        "intents_with_fewer_than_two_rows": rare, "unrepresented_intents": missing_intents,
        "split_precheck_passed": len(usable) >= golden_size + 100 and not rare and bool(valid.all()),
        "note": "Count precheck only; inspect label quality, sampling and class coverage before freezing.",
    }


def save_annotation(
    path: Path, example_id: str, expected_text: str, intent: str, decision: str,
    escalation_reason: str, required_reply_points: str, forbidden_claims: str,
    annotator: str, intents: tuple[str, ...],
) -> None:
    if intent not in intents or decision not in {"auto_handle", "escalate"}:
        raise ValueError("Select a valid intent and decision.")
    if not annotator.strip():
        raise ValueError("Annotator name is required.")
    if decision == "escalate" and not escalation_reason.strip():
        raise ValueError("Explain why this case needs human review.")
    if not required_reply_points.strip() or not forbidden_claims.strip():
        raise ValueError("Provide required reply points and forbidden claims (or explicitly write none).")
    queue = read_table(path)
    if queue["example_id"].duplicated().any():
        raise ValueError("Queue example IDs must be unique.")
    match = queue["example_id"].eq(example_id)
    if match.sum() != 1 or queue.loc[match, "text"].iloc[0] != expected_text:
        raise ValueError("Unknown or changed example; reload before saving.")
    if not normalize_for_matching(expected_text) or not queue.loc[match, "thread_id"].iloc[0].strip():
        raise ValueError("Example has unusable text or thread ID; repair it separately.")
    updates = dict(intent=intent, decision=decision, escalation_reason=escalation_reason,
                   required_reply_points=required_reply_points, forbidden_claims=forbidden_claims,
                   annotator=annotator.strip(), reviewed="yes")
    if "label_source" in queue.columns:
        updates["label_source"] = "human"
    for field, value in updates.items():
        queue.loc[match, field] = value
    _atomic_save(queue, path)


def submission_status(config: ProjectConfig, golden_size: int = 200) -> dict:
    result = {"annotation": annotation_progress(read_table(config.data["label_queue_csv"]), config.intents, golden_size)}
    blockers = []
    for name, key in (("training", "train_labels_csv"), ("golden", "golden_csv")):
        rows = len(read_table(config.data[key])) if config.data[key].exists() else 0
        result[f"{name}_rows"] = rows
    if result["training_rows"] < 100:
        blockers.append("At least 100 reviewed training rows are needed after splitting.")
    if not 150 <= result["golden_rows"] <= 250:
        blockers.append("Freeze 150–250 human-labelled golden examples.")
    directory = config.data["model_path"].parent / "evaluation"
    # Existence is only a checklist, not a certificate of valid current results.
    names = ("headline_metrics.csv", "predictions.csv", "evaluation_manifest.json",
             "judge_scores_current.csv", "judge_human_agreement.csv")
    result["evidence_files_present"] = {name: (directory / name).exists() for name in names}
    for name, exists in result["evidence_files_present"].items():
        if not exists:
            blockers.append(f"Missing evaluation evidence: {name}")
    human = read_table(config.data["human_scores_csv"])
    result["reviewed_human_score_rows"] = int(human["reviewed"].eq("yes").sum()) if "reviewed" in human else 0
    if not result["reviewed_human_score_rows"]:
        blockers.append("Complete independent human reply scoring; no reviewed worksheet rows found.")
    result["blockers"] = blockers
    result["final_submission_ready"] = False
    result["note"] = "Manual sign-off required: inspect provenance, paired agreement coverage, five real failures and final report. File presence alone never certifies readiness."
    return result


def export_failure_queue(predictions_csv: Path, golden_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Export all measured main-system intent/decision errors, prioritizing unsafe auto-handles."""
    if output_csv.resolve() in {predictions_csv.resolve(), golden_csv.resolve()}:
        raise ValueError("Output must not replace source data.")
    manifest_path = predictions_csv.parent / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, path in (("golden_sha256", golden_csv), ("predictions_sha256", predictions_csv)):
        if manifest.get(key) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Evaluation evidence is stale; evaluate again before exporting failures.")
    predictions = read_table(predictions_csv)
    golden = read_table(golden_csv)
    main = predictions[predictions["system"].eq("main")].copy()
    if main.empty or main["example_id"].duplicated().any():
        raise ValueError("Expected exactly one main prediction per example.")
    if set(main["example_id"]) != set(golden["example_id"]):
        raise ValueError("Main predictions must match golden examples.")
    rows = main.merge(golden[["example_id", "text"]], on="example_id", validate="one_to_one")
    rows["intent_error"] = rows["actual_intent"].ne(rows["predicted_intent"])
    rows["decision_error"] = rows["actual_decision"].ne(rows["predicted_decision"])
    rows["false_auto"] = rows["actual_decision"].eq("escalate") & rows["predicted_decision"].eq("auto_handle")
    rows = rows[rows["intent_error"] | rows["decision_error"]].sort_values(
        ["false_auto", "decision_error", "example_id"], ascending=[False, False, True], kind="stable",
    )
    for field in ("failure_mode", "likely_cause", "risk", "proposed_fix", "reviewer", "reviewed"):
        rows[field] = "no" if field == "reviewed" else ""
    if output_csv.exists() and len(read_table(output_csv)):
        raise ValueError("Failure worksheet already has rows; archive it before generating another run.")
    _atomic_save(rows, output_csv)
    return rows