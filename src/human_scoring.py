"""Blind human review of a specific, content-validated paired judge sample.

Pass the current sample returned by judge_predictions, not an append-only cache
containing multiple content/model versions. No scores are inferred or copied
from the judge. These APIs do not call Ollama.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pandas as pd

from src import judge


PAIR = ["example_id", "system"]
CONTENT = ["text", "draft_reply", "evidence_json"]
ANNOTATIONS = [*judge.RUBRIC_FIELDS, "annotator", "reviewed"]
WORKSHEET_COLUMNS = [*PAIR, *CONTENT, "cache_key", *ANNOTATIONS]


def _read_csv(path: Path, required: list[str], name: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(f"{name} must be a valid CSV with headers.") from error
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    return frame


def _nonblank(frame: pd.DataFrame, fields: list[str], name: str) -> None:
    for field in fields:
        if frame[field].str.strip().eq("").any():
            raise ValueError(f"{name} contains blank {field} values.")


def _unique_pairs(frame: pd.DataFrame, name: str) -> None:
    _nonblank(frame, PAIR, name)
    if frame.duplicated(PAIR).any():
        raise ValueError(f"{name} contains duplicate example_id/system pairs.")


def _json(value: str) -> object:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"Nonfinite JSON constant: {constant}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        result = json.loads(value, parse_constant=reject_constant, object_pairs_hook=unique_object)
        # Also reject floating-point overflow, e.g. 1e999, anywhere in the object.
        json.dumps(result, allow_nan=False)
        return result
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("Malformed JSON in prediction or evidence.") from error


def _worksheet(path: Path) -> pd.DataFrame:
    frame = _read_csv(path, WORKSHEET_COLUMNS, "Human worksheet")
    if set(frame.columns) != set(WORKSHEET_COLUMNS):
        raise ValueError("Human worksheet has unexpected columns; refusing to drop data.")
    _unique_pairs(frame, "Human worksheet")
    _nonblank(frame, ["text", "cache_key"], "Human worksheet")
    if not frame["cache_key"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Human worksheet contains an invalid cache_key.")
    if not frame["reviewed"].str.strip().str.casefold().isin(["yes", "no"]).all():
        raise ValueError("Human worksheet reviewed must be yes or no.")
    for evidence in frame["evidence_json"]:
        _json(evidence)
    reviewed = frame["reviewed"].str.strip().str.casefold().eq("yes")
    # Partial work is allowed, but never preserve an invalid score silently.
    for field in judge.RUBRIC_FIELDS:
        filled = frame[field].str.strip().ne("")
        # Spreadsheet/CSV round trips may spell an integer as 4.0. Validate
        # numerically without rewriting its original annotation representation.
        judge._numeric_scores(frame.loc[filled], "Human worksheet", (field,))
        if (reviewed & ~filled).any():
            raise ValueError(f"Reviewed rows require a {field} score.")
    _nonblank(frame.loc[reviewed], ["annotator"], "Reviewed rows")
    return frame


def _atomic_save(frame: pd.DataFrame, path: Path) -> None:
    """Back up every existing destination, then atomically replace it.

    Backups have unique names (<filename>.<uuid>.bak) and are never overwritten.
    Callers must finish validation first; concurrent editors require external
    coordination (this is an atomic replacement, not a multi-writer lock).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            shutil.copy2(path, path.with_name(f"{path.name}.{uuid.uuid4().hex}.bak"))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_human_scores(
    predictions_csv: Path, golden_csv: Path, judge_csv: Path, output_csv: Path,
) -> pd.DataFrame:
    """Export a blind worksheet for exactly the supplied paired judge sample.

    Every sampled example must contain all systems found in predictions (not a
    hard-coded three). Keys must match the current text, draft, evidence, model
    and rubric. Existing annotations are preserved verbatim for identical rows.
    Changed content is rejected, as is removal of reviewed or partially annotated
    rows. All validation precedes any write, including directory/backup creation.
    """
    for source in (predictions_csv, golden_csv, judge_csv):
        if output_csv.resolve() == source.resolve() or (
            output_csv.exists() and output_csv.samefile(source)
        ):
            raise ValueError("Worksheet output must not replace an input file.")
    predictions = _read_csv(
        predictions_csv, [*PAIR, "draft_reply", "prediction_json"], "Predictions",
    )
    golden = _read_csv(golden_csv, ["example_id", "text"], "Golden set")
    sample = _read_csv(
        judge_csv, [*PAIR, "model", "rubric_version", "cache_key"], "Judge sample",
    )
    if predictions.empty or golden.empty or sample.empty:
        raise ValueError("Predictions, golden set and judge sample must not be empty.")
    _unique_pairs(predictions, "Predictions")
    _unique_pairs(sample, "Judge sample (select a single current version)")
    _nonblank(golden, ["example_id", "text"], "Golden set")
    _nonblank(sample, ["model", "rubric_version", "cache_key"], "Judge sample")
    if golden["example_id"].duplicated().any():
        raise ValueError("Golden example_id values must be unique.")
    if not set(predictions["example_id"]) <= set(golden["example_id"]):
        raise ValueError("Predictions contain unknown golden example IDs.")
    if not sample["rubric_version"].eq(judge.RUBRIC_VERSION).all():
        raise ValueError("Judge sample has an incorrect rubric_version.")
    systems = set(predictions["system"])
    selected_ids = set(sample["example_id"])
    expected_pairs = {(example_id, system) for example_id in selected_ids for system in systems}
    actual_pairs = set(sample[PAIR].itertuples(index=False, name=None))
    current_pairs = set(predictions[PAIR].itertuples(index=False, name=None))
    if actual_pairs != expected_pairs or not expected_pairs <= current_pairs:
        raise ValueError("Paired sample has unknown or missing example_id/system pairs.")

    evidence_by_pair = {}
    for row in predictions.to_dict("records"):
        prediction = _json(row["prediction_json"])
        if not isinstance(prediction, dict):
            raise ValueError("prediction_json must contain an object.")
        evidence_by_pair[(row["example_id"], row["system"])] = json.dumps(
            prediction.get("evidence", []), sort_keys=True, ensure_ascii=True,
        )
    current = predictions.set_index(PAIR)
    texts = golden.set_index("example_id")["text"]
    rows = []
    for record in sample.to_dict("records"):
        pair = (record["example_id"], record["system"])
        draft = current.loc[pair, "draft_reply"]
        text = texts.loc[pair[0]]
        evidence = evidence_by_pair[pair]
        key = judge._cache_key(record["model"], text, draft, evidence)
        if record["cache_key"] != key:
            raise ValueError(f"Stale or invalid judge cache_key for {pair!r}.")
        rows.append({
            "example_id": pair[0], "system": pair[1], "text": text,
            "draft_reply": draft, "evidence_json": evidence, "cache_key": key,
            **dict.fromkeys(judge.RUBRIC_FIELDS, ""), "annotator": "", "reviewed": "no",
        })
    worksheet = pd.DataFrame(rows, columns=WORKSHEET_COLUMNS).set_index(PAIR)
    if output_csv.exists():
        # Migrate the distributed header-only legacy score template without
        # discarding any populated legacy annotations.
        prior = pd.read_csv(output_csv, dtype=str, keep_default_na=False)
        existing = (_worksheet(output_csv).set_index(PAIR) if not prior.empty
                    else pd.DataFrame(columns=WORKSHEET_COLUMNS).set_index(PAIR))
        for pair, old in existing.iterrows():
            if pair not in worksheet.index:
                has_work = old["reviewed"].strip().casefold() == "yes" or any(
                    old[field].strip() for field in [*judge.RUBRIC_FIELDS, "annotator"]
                )
                if has_work:
                    raise ValueError(f"Cannot remove existing reviewed or annotated row {pair!r}.")
                continue
            if any(old[field] != worksheet.loc[pair, field] for field in [*CONTENT, "cache_key"]):
                raise ValueError(f"Changed content/cache_key for existing row {pair!r}.")
            worksheet.loc[pair, ANNOTATIONS] = old[ANNOTATIONS]
    result = worksheet.reset_index()[WORKSHEET_COLUMNS]
    _atomic_save(result, output_csv)
    return result


def save_human_score(
    path: Path, example_id: str, system: str, cache_key: str,
    scores: dict[str, int], annotator: str,
) -> None:
    """Save explicit human scores for one exact worksheet row, with a backup.

    Content provenance is established by prepare_human_scores; this function
    checks the caller's key against that worksheet, not external predictions.
    """
    if not isinstance(scores, dict) or set(scores) != set(judge.RUBRIC_FIELDS):
        raise ValueError("scores must contain exactly the five rubric fields.")
    for field, value in scores.items():
        if type(value) is not int or not 1 <= value <= 5:
            raise ValueError(f"Invalid {field} score: expected a strict integer from 1 to 5.")
    if not isinstance(annotator, str) or not annotator.strip():
        raise ValueError("A nonblank annotator is required.")
    worksheet = _worksheet(path)
    match = (
        worksheet["example_id"].eq(example_id) & worksheet["system"].eq(system)
        & worksheet["cache_key"].eq(cache_key)
    )
    if match.sum() != 1:
        raise ValueError("Unknown row or stale cache_key; an exact worksheet row is required.")
    for field, value in scores.items():
        worksheet.loc[match, field] = str(value)
    worksheet.loc[match, "annotator"] = annotator
    worksheet.loc[match, "reviewed"] = "yes"
    _atomic_save(worksheet, path)