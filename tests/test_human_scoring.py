import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src import human_scoring, judge
from src.human_scoring import prepare_human_scores, save_human_score


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    monkeypatch.setattr(judge, "_ollama_score", Mock(side_effect=AssertionError("No live judge")))
    monkeypatch.setattr(judge.urllib.request, "urlopen", Mock(side_effect=AssertionError("No network")))


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_inputs(case):
    case.predictions.to_csv(case.predictions_csv, index=False)
    case.golden.to_csv(case.golden_csv, index=False)
    case.sample.to_csv(case.judge_csv, index=False)


def refresh_keys(case):
    for index, row in case.sample.iterrows():
        prediction = case.predictions.set_index(["example_id", "system"]).loc[
            (row["example_id"], row["system"])
        ]
        evidence = json.dumps(
            json.loads(prediction["prediction_json"]).get("evidence", []),
            sort_keys=True, ensure_ascii=True,
        )
        text = case.golden.set_index("example_id").loc[row["example_id"], "text"]
        case.sample.loc[index, "cache_key"] = judge._cache_key(
            row["model"], text, prediction["draft_reply"], evidence,
        )


@pytest.fixture
def case(tmp_path):
    predictions = pd.DataFrame([
        {
            "example_id": example_id, "system": system,
            "draft_reply": f"Draft {example_id} {system}\nTry again, please.",
            "prediction_json": json.dumps({"evidence": [{"reply": "café", "id": "001"}]}),
        }
        for example_id in ("001", "002", "003")
        for system in ("main", "keyword", "trivial")
    ])
    predictions.loc[0, "draft_reply"] = ""  # Abstentions are valid drafts.
    sample = predictions.iloc[:6][["example_id", "system"]].copy()
    sample["model"] = "local:test"
    sample["rubric_version"] = judge.RUBRIC_VERSION
    sample["cache_key"] = ""
    for field in judge.RUBRIC_FIELDS:
        sample[field] = 5
    sample["rationale"] = "SECRET AUTOMATED RATIONALE: never show the reviewer"
    result = SimpleNamespace(
        predictions=predictions,
        golden=pd.DataFrame([
            {"example_id": example_id, "text": f"Customer {example_id}, café\nHelp me"}
            for example_id in ("001", "002", "003")
        ]),
        sample=sample, predictions_csv=tmp_path / "predictions.csv",
        golden_csv=tmp_path / "golden.csv", judge_csv=tmp_path / "judge.csv",
        output_csv=tmp_path / "reviews" / "human.csv",
    )
    refresh_keys(result)
    write_inputs(result)
    return result


def prepare(case):
    return prepare_human_scores(
        case.predictions_csv, case.golden_csv, case.judge_csv, case.output_csv,
    )


def score_first(case, **kwargs):
    row = read(case.output_csv).iloc[0]
    args = dict(
        path=case.output_csv, example_id=row["example_id"], system=row["system"],
        cache_key=row["cache_key"], scores=dict.fromkeys(judge.RUBRIC_FIELDS, 4),
        annotator="  Reviewer One  ",
    )
    save_human_score(**{**args, **kwargs})


def snapshot(directory):
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_prepare_blind_exact_sample_and_no_automated_score_leakage(case):
    before = {path: path.read_bytes() for path in (case.predictions_csv, case.golden_csv, case.judge_csv)}
    result = prepare(case)
    assert list(result.columns) == human_scoring.WORKSHEET_COLUMNS
    assert len(result) == 6
    assert list(result[["example_id", "system"]].itertuples(index=False, name=None)) == list(
        case.sample[["example_id", "system"]].itertuples(index=False, name=None)
    )
    assert result.groupby("example_id")["system"].apply(set).tolist() == [{"main", "keyword", "trivial"}] * 2
    assert result[[*judge.RUBRIC_FIELDS, "annotator"]].eq("").all().all()
    assert result["reviewed"].eq("no").all()
    assert result["cache_key"].tolist() == case.sample["cache_key"].tolist()
    assert result.iloc[0]["draft_reply"] == ""
    assert "\\u00e9" in result.iloc[0]["evidence_json"]
    assert "SECRET AUTOMATED" not in case.output_csv.read_text(encoding="utf-8")
    assert not {"rationale", "model", "rubric_version"} & set(result.columns)
    pd.testing.assert_frame_equal(result, read(case.output_csv))
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("systems", [("main", "keyword"), ("main", "keyword", "trivial", "fourth")])
def test_pairing_uses_all_prediction_systems_not_hardcoded_three(case, systems):
    extra = case.predictions[case.predictions["system"].eq("main")].copy()
    extra["system"] = "fourth"
    case.predictions = pd.concat([case.predictions, extra], ignore_index=True)
    case.predictions = case.predictions[case.predictions["system"].isin(systems)]
    case.sample = case.predictions[case.predictions["example_id"].eq("001")][["example_id", "system"]].copy()
    case.sample["model"] = "local:test"
    case.sample["rubric_version"] = judge.RUBRIC_VERSION
    refresh_keys(case)
    write_inputs(case)
    result = prepare(case)
    assert len(result) == len(systems)
    assert set(result["system"]) == set(systems)


def test_save_and_prepare_preserve_exact_scores_partial_work_and_backups(case):
    prepare(case)
    initial = case.output_csv.read_bytes()
    score_first(case, scores=dict(zip(judge.RUBRIC_FIELDS, range(1, 6))))
    backups = list(case.output_csv.parent.glob("human.csv.*.bak"))
    assert len(backups) == 1 and backups[0].read_bytes() == initial
    saved = read(case.output_csv)
    assert saved.iloc[0][list(judge.RUBRIC_FIELDS)].tolist() == ["1", "2", "3", "4", "5"]
    assert saved.iloc[0]["annotator"] == "  Reviewer One  "
    assert saved.iloc[0]["reviewed"] == "yes"
    assert saved.iloc[1:]["reviewed"].eq("no").all()
    saved.loc[1, "relevance"] = "3"
    saved.loc[1, "annotator"] = " Partial work "
    saved.to_csv(case.output_csv, index=False)
    before = case.output_csv.read_bytes()
    # Neither new automated scores nor input ordering can replace human work.
    case.sample[list(judge.RUBRIC_FIELDS)] = 1
    case.sample = case.sample.iloc[::-1]
    case.predictions = case.predictions.iloc[::-1]
    write_inputs(case)
    result = prepare(case)
    pd.testing.assert_frame_equal(
        saved.sort_values(["example_id", "system"]).reset_index(drop=True),
        result.sort_values(["example_id", "system"]).reset_index(drop=True),
    )
    assert any(path.read_bytes() == before for path in case.output_csv.parent.glob("human.csv.*.bak"))
    assert len(list(case.output_csv.parent.glob("human.csv.*.bak"))) == 2


@pytest.mark.parametrize("problem", [
    "stale_key", "model", "rubric", "duplicate_judge", "duplicate_predictions",
    "duplicate_golden", "missing_judge_pair", "missing_prediction_pair",
    "unknown_judge_id", "unknown_judge_system", "unknown_prediction_id",
    "blank_text", "blank_system", "blank_model", "missing_column", "empty_sample",
])
def test_invalid_inputs_never_create_output_or_backups(case, problem):
    if problem == "stale_key":
        case.sample.loc[0, "cache_key"] = "0" * 64
    elif problem == "model":
        case.sample.loc[0, "model"] = "changed:model"
    elif problem == "rubric":
        case.sample.loc[0, "rubric_version"] = "obsolete"
    elif problem.startswith("duplicate_"):
        name = {"duplicate_judge": "sample", "duplicate_predictions": "predictions", "duplicate_golden": "golden"}[problem]
        frame = getattr(case, name)
        setattr(case, name, pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    elif problem == "missing_judge_pair":
        case.sample = case.sample.iloc[1:]
    elif problem == "missing_prediction_pair":
        case.predictions = case.predictions.iloc[1:]
    elif problem == "unknown_judge_id":
        case.sample["example_id"] = case.sample["example_id"].replace("001", "unknown")
    elif problem == "unknown_judge_system":
        case.sample.loc[0, "system"] = "unknown"
    elif problem == "unknown_prediction_id":
        case.predictions.loc[0, "example_id"] = "unknown"
    elif problem == "blank_text":
        case.golden.loc[0, "text"] = " \t"
    elif problem == "blank_system":
        case.predictions.loc[0, "system"] = " "
    elif problem == "blank_model":
        case.sample.loc[0, "model"] = " "
    elif problem == "missing_column":
        case.sample = case.sample.drop(columns="rubric_version")
    else:
        case.sample = case.sample.iloc[:0]
    write_inputs(case)
    before = snapshot(case.predictions_csv.parent)
    with pytest.raises(ValueError):
        prepare(case)
    assert snapshot(case.predictions_csv.parent) == before
    assert not case.output_csv.parent.exists()


@pytest.mark.parametrize("payload", [
    "{", "[]", "null", '{"evidence": NaN}', '{"evidence": Infinity}',
    '{"evidence": [1e999]}', '{"evidence": [], "evidence": [1]}',
    '{"evidence": [{"x": 1, "x": 2}]}',
])
@pytest.mark.parametrize("index", [0, 8])
def test_malformed_prediction_json_even_outside_sample_is_rejected(case, payload, index):
    case.predictions.loc[index, "prediction_json"] = payload
    write_inputs(case)
    with pytest.raises(ValueError, match="JSON|object"):
        prepare(case)
    assert not case.output_csv.parent.exists()


@pytest.mark.parametrize("changed", ["text", "draft", "evidence", "model"])
def test_changed_current_content_rejected_even_with_fresh_judge_key(case, changed):
    prepare(case)
    score_first(case)
    before = snapshot(case.output_csv.parent)
    if changed == "text":
        case.golden.loc[0, "text"] = "New customer text"
    elif changed == "draft":
        case.predictions.loc[0, "draft_reply"] = "New draft"
    elif changed == "evidence":
        case.predictions.loc[0, "prediction_json"] = '{"evidence": ["New evidence"]}'
    else:
        case.sample["model"] = "new:model"
    refresh_keys(case)
    write_inputs(case)
    with pytest.raises(ValueError, match="Changed content/cache_key"):
        prepare(case)
    assert snapshot(case.output_csv.parent) == before


@pytest.mark.parametrize("field,value", [
    ("text", "Tampered text"), ("draft_reply", "Tampered draft"),
    ("evidence_json", "[123]"), ("cache_key", "f" * 64),
])
def test_same_identity_cannot_hide_edited_worksheet_content(case, field, value):
    worksheet = prepare(case)
    worksheet.loc[0, field] = value
    worksheet.to_csv(case.output_csv, index=False)
    before = snapshot(case.output_csv.parent)
    with pytest.raises(ValueError, match="Changed content/cache_key"):
        prepare(case)
    assert snapshot(case.output_csv.parent) == before


@pytest.mark.parametrize("work", ["reviewed", "partial", "none"])
def test_removing_sample_preserves_reviewed_and_partial_work(case, work):
    frame = prepare(case)
    if work == "reviewed":
        score_first(case)
    elif work == "partial":
        frame.loc[0, "relevance"] = "2"
        frame.to_csv(case.output_csv, index=False)
    case.sample = case.sample[case.sample["example_id"].eq("002")]
    write_inputs(case)
    before = snapshot(case.output_csv.parent)
    if work == "none":
        assert len(prepare(case)) == 3
    else:
        with pytest.raises(ValueError, match="Cannot remove"):
            prepare(case)
        assert snapshot(case.output_csv.parent) == before


@pytest.mark.parametrize("score", [True, False, 3.0, 1.5, "4", None, 0, 6, float("nan"), float("inf")])
@pytest.mark.parametrize("field", judge.RUBRIC_FIELDS)
def test_save_rejects_non_strict_scores_without_writes(case, score, field):
    prepare(case)
    before = snapshot(case.output_csv.parent)
    scores = dict.fromkeys(judge.RUBRIC_FIELDS, 4)
    scores[field] = score
    with pytest.raises(ValueError, match="strict integer"):
        score_first(case, scores=scores)
    assert snapshot(case.output_csv.parent) == before


@pytest.mark.parametrize("annotator", ["", " \t\n", None, 7, True])
def test_save_requires_nonblank_annotator(case, annotator):
    prepare(case)
    before = snapshot(case.output_csv.parent)
    with pytest.raises(ValueError, match="annotator"):
        score_first(case, annotator=annotator)
    assert snapshot(case.output_csv.parent) == before


@pytest.mark.parametrize("scores", [{}, {"relevance": 4}, {**dict.fromkeys(judge.RUBRIC_FIELDS, 4), "rationale": "no"}, None])
def test_save_requires_exact_score_fields(case, scores):
    prepare(case)
    with pytest.raises(ValueError, match="five rubric fields"):
        score_first(case, scores=scores)
    assert not list(case.output_csv.parent.glob("*.bak"))


@pytest.mark.parametrize("kwargs", [{"example_id": "unknown"}, {"system": "unknown"}, {"cache_key": "0" * 64}])
def test_save_requires_known_exact_identity(case, kwargs):
    prepare(case)
    before = snapshot(case.output_csv.parent)
    with pytest.raises(ValueError, match="Unknown row or stale"):
        score_first(case, **kwargs)
    assert snapshot(case.output_csv.parent) == before


@pytest.mark.parametrize("operation", ["prepare", "save"])
@pytest.mark.parametrize("problem", ["duplicate", "bad_json", "bad_score", "blank_annotator", "missing_score", "extra_column"])
def test_existing_worksheet_validation_precedes_writes(case, operation, problem):
    worksheet = prepare(case)
    if problem == "duplicate":
        worksheet = pd.concat([worksheet, worksheet.iloc[[0]]])
    elif problem == "bad_json":
        worksheet.loc[1, "evidence_json"] = "{"
    elif problem == "bad_score":
        worksheet.loc[1, "safety"] = "True"
    elif problem == "extra_column":
        worksheet["unexpected_notes"] = "do not discard"
    else:
        worksheet.loc[1, "reviewed"] = "yes"
        worksheet.loc[1, list(judge.RUBRIC_FIELDS)] = "4"
        if problem == "missing_score":
            worksheet.loc[1, "annotator"] = "Reviewer"
            worksheet.loc[1, "tone"] = ""
    worksheet.to_csv(case.output_csv, index=False)
    before = snapshot(case.output_csv.parent)
    with pytest.raises(ValueError):
        prepare(case) if operation == "prepare" else score_first(case)
    assert snapshot(case.output_csv.parent) == before


def test_prepare_rejects_input_as_output(case):
    before = snapshot(case.predictions_csv.parent)
    with pytest.raises(ValueError, match="must not replace an input"):
        prepare_human_scores(case.predictions_csv, case.golden_csv, case.judge_csv, case.judge_csv)
    assert snapshot(case.predictions_csv.parent) == before


@pytest.mark.parametrize("failure", ["backup", "replace"])
def test_atomic_save_failure_keeps_original_and_cleans_temporary_file(case, monkeypatch, failure):
    prepare(case)
    before = case.output_csv.read_bytes()
    target, method = (human_scoring.shutil, "copy2") if failure == "backup" else (human_scoring.os, "replace")
    monkeypatch.setattr(target, method, Mock(side_effect=OSError("Simulated failure")))
    with pytest.raises(OSError, match="Simulated failure"):
        score_first(case)
    assert case.output_csv.read_bytes() == before
    assert not list(case.output_csv.parent.glob("*.tmp"))
    if failure == "replace":
        backups = list(case.output_csv.parent.glob("*.bak"))
        assert len(backups) == 1 and backups[0].read_bytes() == before


def test_prepare_preserves_integral_scores_after_spreadsheet_roundtrip(case):
    prepare(case)
    worksheet = pd.read_csv(case.output_csv, dtype={"example_id": str, "annotator": str})
    worksheet.loc[0, list(judge.RUBRIC_FIELDS)] = 4
    worksheet.loc[0, "annotator"] = "Reviewer"
    worksheet.loc[0, "reviewed"] = "yes"
    worksheet.to_csv(case.output_csv, index=False)
    expected = read(case.output_csv)
    assert expected.iloc[0]["relevance"] == "4.0"
    result = prepare(case)
    pd.testing.assert_frame_equal(result, expected)


def test_prepare_save_and_partial_agreement_end_to_end(case):
    prepare(case)
    with pytest.raises(ValueError, match="No matching"):
        judge.judge_human_agreement(case.judge_csv, case.output_csv)
    score_first(case)
    result = judge.judge_human_agreement(case.judge_csv, case.output_csv)
    assert result["examples"].eq(1).all()
    assert result["exact_agreement"].eq(0).all()
    assert result["within_one_agreement"].eq(1).all()
    assert result["spearman"].isna().all()