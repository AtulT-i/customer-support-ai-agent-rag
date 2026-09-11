"""Submission bookkeeping uses temporary evidence, never models or live services."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.config import load_config
from src.submission import (
    annotation_progress,
    export_failure_queue,
    read_table,
    save_annotation,
    submission_status,
)


INTENTS = ("account_access", "billing_subscription", "other_unclear")
QUEUE_COLUMNS = [
    "example_id", "text", "thread_id", "intent", "decision", "reviewed",
    "escalation_reason", "required_reply_points", "forbidden_claims", "annotator",
    "historical_reply", "label_source",
]


@pytest.fixture(autouse=True)
def forbid_network_and_model_loads(monkeypatch):
    guards = []
    for target in (
        "urllib.request.urlopen", "urllib.request.OpenerDirector.open",
        "socket.socket.connect", "socket.socket.connect_ex", "joblib.load",
        "src.judge._ollama_score",
    ):
        guard = Mock(side_effect=AssertionError(f"Forbidden external call: {target}"))
        monkeypatch.setattr(target, guard)
        guards.append(guard)
    yield
    for guard in guards:
        guard.assert_not_called()


def queue_row(index, **overrides):
    return {
        "example_id": f"{index:03d}", "text": f"Customer needs login help case {index}",
        "thread_id": f"thread-{index}", "intent": "account_access",
        "decision": "auto_handle", "reviewed": "yes", "escalation_reason": "",
        "required_reply_points": "Ask about the device", "forbidden_claims": "No account changes",
        "annotator": "Existing reviewer", "historical_reply": "Context, not a label\nCafé",
        "label_source": "weak_provisional", **overrides,
    }


def queue_frame(rows=()):
    return pd.DataFrame(rows, columns=QUEUE_COLUMNS)


def snapshot(directory):
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("size", [299, 300, 301])
def test_annotation_progress_default_requires_300_not_500(size):
    queue = queue_frame([
        queue_row(index, intent=INTENTS[index % len(INTENTS)]) for index in range(size)
    ])
    result = annotation_progress(queue, INTENTS)
    assert result["golden_size"] == 200
    assert result["minimum_reviewed_rows"] == 300
    assert result["usable_unique_thread_rows"] == size
    assert result["additional_usable_rows_needed"] == max(0, 300 - size)
    assert result["split_precheck_passed"] == (size >= 300)


@pytest.mark.parametrize("golden_size", [150, 200, 250])
def test_annotation_progress_header_only_queue(golden_size):
    result = annotation_progress(queue_frame(), INTENTS, golden_size)
    for key in ("queue_rows", "reviewed_rows", "invalid_reviewed_rows", "usable_unique_thread_rows"):
        assert result[key] == 0
    assert result["intent_counts"] == {}
    assert result["intents_with_fewer_than_two_rows"] == {}
    assert result["unrepresented_intents"] == sorted(INTENTS)
    assert result["minimum_reviewed_rows"] == golden_size + 100
    assert result["additional_usable_rows_needed"] == golden_size + 100
    assert not result["split_precheck_passed"]


@pytest.mark.parametrize("golden_size", [149, 251])
def test_annotation_progress_rejects_invalid_golden_size(golden_size):
    with pytest.raises(ValueError, match="between 150 and 250"):
        annotation_progress(queue_frame(), INTENTS, golden_size)


@pytest.mark.parametrize("field", ["example_id", "text", "thread_id", "intent", "decision", "reviewed"])
def test_annotation_progress_requires_queue_schema(field):
    with pytest.raises(ValueError, match="missing columns"):
        annotation_progress(queue_frame().drop(columns=field), INTENTS)


def test_annotation_progress_counts_duplicates_invalid_empty_and_rare_without_mutation():
    queue = queue_frame([
        queue_row(0, text="Cannot log in"),
        queue_row(1, text="Login page is blank"),
        queue_row(2, intent="billing_subscription"),
        queue_row(3, text="@someone CANNOT   log in https://example.invalid/help"),
        queue_row(4, thread_id="thread-1"),
        queue_row(5, intent="invented_intent"),
        queue_row(6, decision="invented_action"),
        queue_row(7, example_id=" \t"),
        queue_row(8, thread_id=" \t"),
        queue_row(9, text=" \t\n"),
        queue_row(10, text="@someone https://example.invalid/help 123456"),
        queue_row(11, reviewed="no", intent="", decision=""),
        queue_row(12, reviewed=" YES "),
    ])
    before = queue.copy(deep=True)
    result = annotation_progress(queue, INTENTS)
    assert result["queue_rows"] == 13
    assert result["reviewed_rows"] == 12
    assert result["invalid_reviewed_rows"] == 6
    assert result["usable_unique_thread_rows"] == 4
    assert result["intent_counts"] == {"account_access": 3, "billing_subscription": 1}
    assert result["intents_with_fewer_than_two_rows"] == {"billing_subscription": 1}
    assert result["unrepresented_intents"] == ["other_unclear"]
    assert result["additional_usable_rows_needed"] == 296
    assert not result["split_precheck_passed"]
    pd.testing.assert_frame_equal(queue, before)


@pytest.mark.parametrize("problem", ["rare", "invalid"])
def test_annotation_precheck_rejects_rare_or_invalid_even_with_300_usable(problem):
    queue = queue_frame([queue_row(index, intent=INTENTS[index % 2]) for index in range(300)])
    extra = queue_row(300, intent="other_unclear" if problem == "rare" else "invalid")
    result = annotation_progress(pd.concat([queue, queue_frame([extra])], ignore_index=True), INTENTS)
    assert result["additional_usable_rows_needed"] == 0
    assert result["invalid_reviewed_rows"] == (1 if problem == "invalid" else 0)
    assert result["intents_with_fewer_than_two_rows"] == ({"other_unclear": 1} if problem == "rare" else {})
    assert not result["split_precheck_passed"]


@pytest.fixture
def annotation_case(tmp_path):
    path = tmp_path / "queue.csv"
    queue_frame([queue_row(1, reviewed="no"), queue_row(2)]).to_csv(path, index=False)
    return SimpleNamespace(path=path, root=tmp_path, args=dict(
        path=path, example_id="001", expected_text=queue_row(1)["text"],
        intent="billing_subscription", decision="escalate",
        escalation_reason="Account-specific review needed", required_reply_points="Explain next steps",
        forbidden_claims="Do not promise a refund", annotator="  Reviewer One  ", intents=INTENTS,
    ))


@pytest.mark.parametrize("with_label_source", [True, False])
def test_save_annotation_explicit_human_values_preserves_other_rows_and_unique_backups(annotation_case, with_label_source):
    case = annotation_case
    original = read_table(case.path)
    if not with_label_source:
        original = original.drop(columns="label_source")
        original.to_csv(case.path, index=False)
    preimages = []
    for changes in ({}, {"decision": "auto_handle", "escalation_reason": "", "forbidden_claims": "none"}):
        preimages.append(case.path.read_bytes())
        args = {**case.args, **changes}
        save_annotation(**args)
        saved = read_table(case.path)
        expected = original.copy(deep=True)
        for field in ("intent", "decision", "escalation_reason", "required_reply_points", "forbidden_claims"):
            expected.loc[0, field] = args[field]
        expected.loc[0, "reviewed"] = "yes"
        expected.loc[0, "annotator"] = "Reviewer One"
        if with_label_source:
            expected.loc[0, "label_source"] = "human"
        pd.testing.assert_frame_equal(saved, expected)
        backups = list(case.root.glob("queue.csv.*.bak"))
        assert len(backups) == len(preimages)
        assert {backup.read_bytes() for backup in backups} == set(preimages)
        assert not list(case.root.glob("*.tmp"))


@pytest.mark.parametrize("changes,message", [
    ({"intent": "Choose an intent"}, "valid intent and decision"),
    ({"decision": "Choose a decision"}, "valid intent and decision"),
    ({"annotator": " \t\n"}, "Annotator name"),
    ({"escalation_reason": " \t"}, "Explain why"),
    ({"required_reply_points": ""}, "required reply points and forbidden claims"),
    ({"forbidden_claims": " \t"}, "required reply points and forbidden claims"),
    ({"expected_text": "Changed message"}, "Unknown or changed example"),
    ({"example_id": "unknown"}, "Unknown or changed example"),
])
def test_save_annotation_invalid_or_stale_requests_do_not_write(annotation_case, changes, message):
    before = snapshot(annotation_case.root)
    with pytest.raises(ValueError, match=message):
        save_annotation(**{**annotation_case.args, **changes})
    assert snapshot(annotation_case.root) == before


def test_save_annotation_refuses_text_changed_on_disk(annotation_case):
    case = annotation_case
    queue = read_table(case.path)
    queue.loc[0, "text"] = "Another editor changed this customer message"
    queue.to_csv(case.path, index=False)
    before = snapshot(case.root)
    with pytest.raises(ValueError, match="Unknown or changed example"):
        save_annotation(**case.args)
    assert snapshot(case.root) == before


@pytest.mark.parametrize("index", [0, 1])
def test_save_annotation_refuses_duplicate_ids_even_on_other_rows(annotation_case, index):
    case = annotation_case
    queue = read_table(case.path)
    pd.concat([queue, queue.iloc[[index]]], ignore_index=True).to_csv(case.path, index=False)
    before = snapshot(case.root)
    with pytest.raises(ValueError, match="IDs must be unique"):
        save_annotation(**case.args)
    assert snapshot(case.root) == before


@pytest.mark.parametrize("field,value", [
    ("text", " \t"), ("text", "@customer https://example.invalid 123456"), ("thread_id", " \t"),
])
def test_save_annotation_refuses_unusable_source_rows(annotation_case, field, value):
    case = annotation_case
    queue = read_table(case.path)
    queue.loc[0, field] = value
    queue.to_csv(case.path, index=False)
    args = {**case.args, "expected_text": queue.loc[0, "text"]}
    before = snapshot(case.root)
    with pytest.raises(ValueError, match="unusable text or thread ID"):
        save_annotation(**args)
    assert snapshot(case.root) == before


@pytest.fixture
def empty_config(tmp_path):
    base = load_config()
    config = replace(base, data={key: tmp_path / path.name for key, path in base.data.items()})
    queue_frame().to_csv(config.data["label_queue_csv"], index=False)
    for key in ("train_labels_csv", "golden_csv"):
        queue_frame().to_csv(config.data[key], index=False)
    pd.DataFrame(columns=["example_id", "system", "reviewed"]).to_csv(config.data["human_scores_csv"], index=False)
    return config


@pytest.mark.parametrize("missing_split_files", [False, True])
def test_submission_status_empty_sources_returns_blockers_without_external_calls(empty_config, tmp_path, missing_split_files):
    if missing_split_files:
        for key in ("train_labels_csv", "golden_csv"):
            empty_config.data[key].unlink()
    before = snapshot(tmp_path)
    result = submission_status(empty_config)
    assert result["training_rows"] == result["golden_rows"] == result["reviewed_human_score_rows"] == 0
    assert result["annotation"]["minimum_reviewed_rows"] == 300
    assert result["annotation"]["additional_usable_rows_needed"] == 300
    evidence = (
        "headline_metrics.csv", "predictions.csv", "evaluation_manifest.json",
        "judge_scores_current.csv", "judge_human_agreement.csv",
    )
    assert result["evidence_files_present"] == dict.fromkeys(evidence, False)
    assert result["blockers"] == [
        "At least 100 reviewed training rows are needed after splitting.",
        "Freeze 150–250 human-labelled golden examples.",
        *(f"Missing evaluation evidence: {name}" for name in evidence),
        "Complete independent human reply scoring; no reviewed worksheet rows found.",
    ]
    assert result["final_submission_ready"] is False
    assert snapshot(tmp_path) == before


def test_submission_status_file_presence_is_not_certification(empty_config, tmp_path):
    for key, count in (("train_labels_csv", 100), ("golden_csv", 200)):
        queue_frame([queue_row(index) for index in range(count)]).to_csv(empty_config.data[key], index=False)
    pd.DataFrame({"reviewed": ["no", "yes", "no"]}).to_csv(empty_config.data["human_scores_csv"], index=False)
    directory = empty_config.data["model_path"].parent / "evaluation"
    directory.mkdir()
    for name in ("headline_metrics.csv", "predictions.csv", "evaluation_manifest.json", "judge_scores_current.csv", "judge_human_agreement.csv"):
        (directory / name).write_text("Not validated evidence", encoding="utf-8")
    before = snapshot(tmp_path)
    result = submission_status(empty_config)
    assert result["blockers"] == []
    assert result["reviewed_human_score_rows"] == 1
    assert all(result["evidence_files_present"].values())
    assert result["final_submission_ready"] is False
    assert "Manual sign-off" in result["note"]
    assert snapshot(tmp_path) == before


def write_failure_inputs(case):
    case.predictions.to_csv(case.predictions_csv, index=False)
    case.golden.to_csv(case.golden_csv, index=False)
    case.manifest_path.write_text(json.dumps({
        "golden_sha256": hashlib.sha256(case.golden_csv.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(case.predictions_csv.read_bytes()).hexdigest(),
    }), encoding="utf-8")


@pytest.fixture
def failure_case(tmp_path):
    rows = []
    for example_id, actual, predicted, intent_error in (
        ("030", "auto_handle", "auto_handle", True),
        ("020", "auto_handle", "escalate", False),
        ("090", "escalate", "auto_handle", False),
        ("010", "auto_handle", "auto_handle", False),
        ("080", "escalate", "auto_handle", True),
        ("005", "escalate", "escalate", True),
    ):
        rows.append(dict(
            example_id=example_id, system="main", actual_intent="account_access",
            predicted_intent="other_unclear" if intent_error else "account_access",
            actual_decision=actual, predicted_decision=predicted, draft_reply="Try again, please.\nCafé",
        ))
    case = SimpleNamespace(
        predictions=pd.DataFrame([*rows, {**rows[2], "system": "keyword"}]),
        golden=pd.DataFrame([{"example_id": row["example_id"], "text": f"Customer {row['example_id']}\nCafé"} for row in reversed(rows)]),
        predictions_csv=tmp_path / "predictions.csv", golden_csv=tmp_path / "golden.csv",
        manifest_path=tmp_path / "evaluation_manifest.json", output_csv=tmp_path / "reviews" / "failures.csv",
        root=tmp_path,
    )
    write_failure_inputs(case)
    return case


def export(case):
    return export_failure_queue(case.predictions_csv, case.golden_csv, case.output_csv)


def test_export_failure_queue_uses_hashed_evidence_and_prioritizes_false_auto(failure_case):
    case = failure_case
    before = snapshot(case.root)
    result = export(case)
    assert result["example_id"].tolist() == ["080", "090", "020", "005", "030"]
    assert result["system"].eq("main").all()
    assert result["false_auto"].tolist() == [True, True, False, False, False]
    assert result["decision_error"].tolist() == [True, True, True, False, False]
    assert result["intent_error"].tolist() == [True, False, False, True, True]
    texts = case.golden.set_index("example_id")["text"]
    assert result["text"].tolist() == [texts[item] for item in result["example_id"]]
    assert result[["failure_mode", "likely_cause", "risk", "proposed_fix", "reviewer"]].eq("").all().all()
    assert result["reviewed"].eq("no").all()
    pd.testing.assert_frame_equal(read_table(case.output_csv), result.astype(str).reset_index(drop=True))
    assert all((case.root / path).read_bytes() == contents for path, contents in before.items())
    assert not list(case.root.rglob("*.bak"))


@pytest.mark.parametrize("source", ["golden", "predictions"])
@pytest.mark.parametrize("problem", ["changed_bytes", "missing_hash"])
def test_export_failure_queue_rejects_stale_or_unhashed_evidence_without_writes(failure_case, source, problem):
    case = failure_case
    if problem == "changed_bytes":
        path = getattr(case, f"{source}_csv")
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        manifest = json.loads(case.manifest_path.read_text(encoding="utf-8"))
        del manifest[f"{source}_sha256"]
        case.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = snapshot(case.root)
    with pytest.raises(ValueError, match="evidence is stale"):
        export(case)
    assert snapshot(case.root) == before
    assert not case.output_csv.parent.exists()


@pytest.mark.parametrize("work", ["unreviewed", "partial", "reviewed"])
def test_export_failure_queue_refuses_overwriting_any_existing_rows(failure_case, work):
    case = failure_case
    frame = export(case)
    if work != "unreviewed":
        frame.loc[frame.index[0], "likely_cause"] = "Independent reviewer finding"
    if work == "reviewed":
        frame.loc[frame.index[0], "reviewed"] = "yes"
    frame.to_csv(case.output_csv, index=False)
    before = snapshot(case.root)
    with pytest.raises(ValueError, match="already has rows; archive"):
        export(case)
    assert snapshot(case.root) == before


@pytest.mark.parametrize("source", ["golden_csv", "predictions_csv"])
def test_export_failure_queue_refuses_replacing_source(failure_case, source):
    case = failure_case
    before = snapshot(case.root)
    with pytest.raises(ValueError, match="must not replace source"):
        export_failure_queue(case.predictions_csv, case.golden_csv, getattr(case, source))
    assert snapshot(case.root) == before


@pytest.mark.parametrize("problem,message", [
    ("no_main", "one main prediction"), ("duplicate_main", "one main prediction"),
    ("missing_example", "must match golden"), ("unknown_example", "must match golden"),
])
def test_export_failure_queue_rejects_invalid_main_predictions(failure_case, problem, message):
    case = failure_case
    if problem == "no_main":
        case.predictions = case.predictions[case.predictions["system"].ne("main")]
    elif problem == "duplicate_main":
        case.predictions = pd.concat([case.predictions, case.predictions.iloc[[0]]], ignore_index=True)
    elif problem == "missing_example":
        case.predictions = case.predictions.iloc[1:]
    else:
        case.predictions.loc[0, "example_id"] = "unknown"
    write_failure_inputs(case)
    before = snapshot(case.root)
    with pytest.raises(ValueError, match=message):
        export(case)
    assert snapshot(case.root) == before
    assert not case.output_csv.parent.exists()