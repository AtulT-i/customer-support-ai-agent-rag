import json
from unittest.mock import Mock

import pandas as pd
import pytest

from src import judge


def rubric():
    return {**dict.fromkeys(judge.RUBRIC_FIELDS, 4), "rationale": "Supported by supplied evidence."}


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    monkeypatch.setattr(judge.urllib.request, "urlopen", Mock(side_effect=AssertionError("Unexpected network call")))


@pytest.fixture
def judge_case(tmp_path, monkeypatch):
    predictions = pd.DataFrame([
        {
            "example_id": f"00{index}", "system": system,
            "draft_reply": f"{system} draft for example {index}",
            "prediction_json": json.dumps({"evidence": [{"customer_tweet_id": "historical", "similarity": 0.5}]}),
        }
        for index in range(6)
        for system in ("main", "keyword", "trivial")
    ])
    golden = pd.DataFrame([
        {"example_id": f"00{index}", "text": f"Customer question {index}"}
        for index in range(6)
    ])
    predictions_csv = tmp_path / "predictions.csv"
    golden_csv = tmp_path / "golden.csv"
    predictions.to_csv(predictions_csv, index=False)
    golden.to_csv(golden_csv, index=False)
    scorer = Mock(side_effect=lambda **kwargs: rubric())
    monkeypatch.setattr(judge, "_ollama_score", scorer)
    return {
        "predictions_csv": predictions_csv, "golden_csv": golden_csv,
        "output_csv": tmp_path / "nested" / "scores.csv", "model": "local-model:tag",
        "sample_size": 3, "random_seed": 42,
    }, predictions, golden, scorer


def mock_ollama(monkeypatch, payload):
    response = Mock()
    response.read.return_value = json.dumps({"response": json.dumps(payload)}).encode()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    urlopen = Mock(return_value=response)
    monkeypatch.setattr(judge.urllib.request, "urlopen", urlopen)
    return urlopen


def test_ollama_temperature_zero_and_valid_json(monkeypatch):
    urlopen = mock_ollama(monkeypatch, rubric())
    assert judge._ollama_score("local", "question", "draft", "[]") == rubric()
    request = urlopen.call_args.args[0]
    body = json.loads(request.data)
    assert body["options"]["temperature"] == 0
    assert body["model"] == "local"
    assert body["format"] == "json"
    assert body["stream"] is False


@pytest.mark.parametrize("score", [True, False, 1.5, 3.0, "4", None, 0, 6])
@pytest.mark.parametrize("field", judge.RUBRIC_FIELDS)
def test_ollama_rejects_noninteger_or_out_of_range_scores(monkeypatch, field, score):
    payload = rubric()
    payload[field] = score
    mock_ollama(monkeypatch, payload)
    with pytest.raises(ValueError, match=f"invalid {field}"):
        judge._ollama_score("local", "question", "draft", "[]")


@pytest.mark.parametrize("payload", [[], {}, {"relevance": 4}])
def test_ollama_rejects_missing_scores_and_nonobjects(monkeypatch, payload):
    mock_ollama(monkeypatch, payload)
    with pytest.raises(ValueError):
        judge._ollama_score("local", "question", "draft", "[]")


def test_paired_sampling_and_deterministic_shuffled_order(judge_case):
    kwargs, predictions, golden, scorer = judge_case
    first = judge.judge_predictions(**kwargs)
    assert len(first) == 9
    assert first["example_id"].nunique() == 3
    assert first.groupby("example_id")["system"].apply(set).tolist() == [{"main", "keyword", "trivial"}] * 3
    assert set(first["model"]) == {kwargs["model"]}
    assert first["cache_key"].str.len().eq(64).all()
    assert set(first["rubric_version"]) == {judge.RUBRIC_VERSION}
    assert scorer.call_count == 9
    assert any(group["system"].tolist() != sorted(group["system"]) for _, group in first.groupby("example_id"))
    predictions.iloc[::-1].to_csv(kwargs["predictions_csv"], index=False)
    golden.iloc[::-1].to_csv(kwargs["golden_csv"], index=False)
    scorer.reset_mock()
    second = judge.judge_predictions(**{**kwargs, "output_csv": kwargs["output_csv"].with_name("second.csv")})
    pd.testing.assert_frame_equal(first, second)
    assert scorer.call_count == 9


def test_identical_run_uses_cache_without_calls(judge_case):
    kwargs, _, _, scorer = judge_case
    first = judge.judge_predictions(**kwargs)
    scorer.reset_mock()
    second = judge.judge_predictions(**kwargs)
    scorer.assert_not_called()
    pd.testing.assert_frame_equal(first, second)
    assert len(pd.read_csv(kwargs["output_csv"])) == 9


@pytest.mark.parametrize("changed", ["model", "rubric", "text", "draft", "evidence"])
def test_cache_invalidation_preserves_unrelated_entries(judge_case, monkeypatch, changed):
    kwargs, predictions, golden, scorer = judge_case
    first = judge.judge_predictions(**kwargs)
    selected_id = first.iloc[0]["example_id"]
    selected_system = first.iloc[0]["system"]
    expected_calls = 9
    if changed == "model":
        kwargs["model"] = "different-model:tag"
    elif changed == "rubric":
        monkeypatch.setattr(judge, "RUBRIC_VERSION", "new-rubric-version")
    elif changed == "text":
        golden.loc[golden["example_id"].eq(selected_id), "text"] = "Changed customer question"
        golden.to_csv(kwargs["golden_csv"], index=False)
        expected_calls = 3
    else:
        mask = predictions["example_id"].eq(selected_id) & predictions["system"].eq(selected_system)
        if changed == "draft":
            predictions.loc[mask, "draft_reply"] = "Changed draft"
        else:
            predictions.loc[mask, "prediction_json"] = json.dumps({"evidence": [{"customer_text": "Changed evidence"}]})
        predictions.to_csv(kwargs["predictions_csv"], index=False)
        expected_calls = 1
    scorer.reset_mock()
    second = judge.judge_predictions(**kwargs)
    assert scorer.call_count == expected_calls
    retained = pd.read_csv(kwargs["output_csv"])
    assert len(retained) == 9 + expected_calls
    assert set(first["cache_key"]) <= set(retained["cache_key"])
    assert len(second) == 9
    scorer.reset_mock()
    judge.judge_predictions(**kwargs)
    scorer.assert_not_called()


def test_cache_preserves_other_samples(judge_case):
    kwargs, _, _, scorer = judge_case
    all_rows = judge.judge_predictions(**{**kwargs, "sample_size": 100})
    assert len(all_rows) == 18
    scorer.reset_mock()
    subset = judge.judge_predictions(**{**kwargs, "sample_size": 1})
    scorer.assert_not_called()
    assert len(subset) == 3
    assert len(pd.read_csv(kwargs["output_csv"])) == 18


def test_legacy_id_only_cache_is_preserved_but_not_reused(judge_case):
    kwargs, _, _, scorer = judge_case
    kwargs["output_csv"].parent.mkdir()
    pd.DataFrame([{"example_id": "legacy", "system": "main", **rubric()}]).to_csv(kwargs["output_csv"], index=False)
    judge.judge_predictions(**kwargs)
    assert scorer.call_count == 9
    retained = pd.read_csv(kwargs["output_csv"], dtype=str)
    assert len(retained) == 10
    assert "legacy" in set(retained["example_id"])
    scorer.reset_mock()
    judge.judge_predictions(**kwargs)
    scorer.assert_not_called()


def test_identical_content_reuses_scores_but_preserves_system_identity(judge_case):
    kwargs, predictions, _, scorer = judge_case
    predictions["draft_reply"] = "Identical draft across systems"
    predictions.to_csv(kwargs["predictions_csv"], index=False)
    result = judge.judge_predictions(**kwargs)
    assert scorer.call_count == 3
    assert len(result) == 9
    assert result.groupby("example_id")["system"].nunique().eq(3).all()


def test_evidence_object_key_order_does_not_invalidate_cache(judge_case):
    kwargs, predictions, _, scorer = judge_case
    judge.judge_predictions(**kwargs)
    predictions["prediction_json"] = json.dumps({"evidence": [{"similarity": 0.5, "customer_tweet_id": "historical"}]})
    predictions.to_csv(kwargs["predictions_csv"], index=False)
    scorer.reset_mock()
    judge.judge_predictions(**kwargs)
    scorer.assert_not_called()


@pytest.mark.parametrize("problem", ["missing_system", "duplicate_prediction", "unknown_id", "duplicate_golden", "blank_text"])
def test_malformed_or_unpaired_inputs_rejected(judge_case, problem):
    kwargs, predictions, golden, scorer = judge_case
    if problem == "missing_system":
        predictions = predictions.iloc[1:]
    elif problem == "duplicate_prediction":
        predictions = pd.concat([predictions, predictions.iloc[[0]]])
    elif problem == "unknown_id":
        predictions.loc[0, "example_id"] = "unknown"
    elif problem == "duplicate_golden":
        golden = pd.concat([golden, golden.iloc[[0]]])
    else:
        golden.loc[0, "text"] = " "
    predictions.to_csv(kwargs["predictions_csv"], index=False)
    golden.to_csv(kwargs["golden_csv"], index=False)
    with pytest.raises(ValueError):
        judge.judge_predictions(**kwargs)
    scorer.assert_not_called()


@pytest.mark.parametrize("sample_size", [0, -1, 1.5, True])
def test_invalid_sample_size_rejected(judge_case, sample_size):
    kwargs, _, _, scorer = judge_case
    with pytest.raises(ValueError, match="positive"):
        judge.judge_predictions(**{**kwargs, "sample_size": sample_size})
    scorer.assert_not_called()


def test_agreement_rejects_mixed_cached_versions(judge_case):
    kwargs, _, _, _ = judge_case
    scores = judge.judge_predictions(**kwargs)
    human_csv = kwargs["output_csv"].with_name("human.csv")
    scores.to_csv(human_csv, index=False)
    judge.judge_predictions(**{**kwargs, "model": "other-model"})
    with pytest.raises(ValueError, match="one score per example/system"):
        judge.judge_human_agreement(kwargs["output_csv"], human_csv)


@pytest.fixture
def agreement_case(tmp_path):
    machine = pd.DataFrame([
        {
            "example_id": example_id, "system": system,
            "cache_key": judge._cache_key("local:test", example_id, system, "[]"),
            **dict.fromkeys(judge.RUBRIC_FIELDS, value),
        }
        for example_id, system, value in [
            ("001", "main", "1"), ("001", "keyword", "3"),
            ("002", "main", "4"), ("002", "keyword", "5"),
        ]
    ])
    human = machine.copy()
    for field in judge.RUBRIC_FIELDS:
        human[field] = ["1", "4", "2", "5"]
    human["reviewed"] = "yes"
    human["annotator"] = "Reviewer"
    return machine, human, tmp_path / "judge.csv", tmp_path / "human.csv"


def agreement(case):
    machine, human, judge_csv, human_csv = case
    machine.to_csv(judge_csv, index=False)
    human.to_csv(human_csv, index=False)
    return judge.judge_human_agreement(judge_csv, human_csv)


@pytest.mark.parametrize("legacy", [True, False])
def test_agreement_numeric_metrics_and_legacy_compatibility(agreement_case, legacy):
    machine, human, judge_csv, human_csv = agreement_case
    if legacy:
        human = human.drop(columns=["cache_key", "reviewed", "annotator"])
    result = agreement((machine, human, judge_csv, human_csv))
    assert list(result.columns) == ["criterion", "spearman", "exact_agreement", "within_one_agreement", "examples"]
    assert result["criterion"].tolist() == list(judge.RUBRIC_FIELDS)
    assert result["examples"].eq(4).all()
    assert result["spearman"].tolist() == pytest.approx([0.8] * 5)
    assert result["exact_agreement"].eq(0.5).all()
    assert result["within_one_agreement"].eq(0.75).all()


def test_agreement_ignores_unreviewed_and_unmatched_incomplete_rows(agreement_case):
    machine, human, judge_csv, human_csv = agreement_case
    human.loc[3, "reviewed"] = "no"
    human.loc[3, [*judge.RUBRIC_FIELDS, "annotator", "cache_key"]] = ""
    # A partially completed worksheet must not count as human ground truth.
    human.loc[3, "relevance"] = "2"
    unmatched_human = human.iloc[[3]].copy()
    unmatched_human["example_id"] = "unmatched human"
    unmatched_human["reviewed"] = "yes"
    human = pd.concat([human, unmatched_human], ignore_index=True)
    unmatched_judge = machine.iloc[[0]].copy()
    unmatched_judge["example_id"] = "unmatched judge"
    unmatched_judge[list(judge.RUBRIC_FIELDS)] = ""
    machine = pd.concat([machine, unmatched_judge], ignore_index=True)
    result = agreement((machine, human, judge_csv, human_csv))
    assert result["examples"].eq(3).all()
    assert result["exact_agreement"].tolist() == pytest.approx([1 / 3] * 5)
    assert result["within_one_agreement"].tolist() == pytest.approx([2 / 3] * 5)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("field", judge.RUBRIC_FIELDS)
@pytest.mark.parametrize("value", ["", "True", "False", "NaN", "Infinity", "-inf", "1.5", "0", "6", "3.00000000000000001"])
def test_agreement_rejects_any_invalid_matched_score(agreement_case, side, field, value):
    agreement_case[side].loc[0, field] = value
    with pytest.raises(ValueError, match=f"invalid {field}"):
        agreement(agreement_case)


def test_agreement_accepts_numeric_integral_csv_values(agreement_case):
    machine, human, _, _ = agreement_case
    machine.loc[0, list(judge.RUBRIC_FIELDS)] = "1.0"
    human.loc[0, list(judge.RUBRIC_FIELDS)] = "1e0"
    assert agreement(agreement_case)["examples"].eq(4).all()


@pytest.mark.parametrize("key", ["", " ", "stale-key", "0" * 64])
def test_agreement_rejects_stale_or_blank_matched_keys(agreement_case, key):
    agreement_case[1].loc[0, "cache_key"] = key
    with pytest.raises(ValueError, match="cache_key"):
        agreement(agreement_case)


@pytest.mark.parametrize("annotator", ["", " \t"])
def test_agreement_requires_matched_reviewed_annotator(agreement_case, annotator):
    agreement_case[1].loc[0, "annotator"] = annotator
    with pytest.raises(ValueError, match="annotator"):
        agreement(agreement_case)


@pytest.mark.parametrize("side,column", [(0, "cache_key"), (1, "annotator"), (1, "reviewed"), (0, "safety"), (1, "tone")])
def test_agreement_rejects_missing_required_columns(agreement_case, side, column):
    agreement_case[side].drop(columns=column, inplace=True)
    with pytest.raises(ValueError, match="required|require|missing"):
        agreement(agreement_case)


@pytest.mark.parametrize("side", [0, 1])
def test_agreement_rejects_duplicate_pairs(agreement_case, side):
    frames = list(agreement_case)
    duplicate = frames[side].iloc[[0]].copy()
    # Even a duplicate unreviewed row is an ambiguous identity.
    if side == 1:
        duplicate["reviewed"] = "no"
    frames[side] = pd.concat([frames[side], duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="one score per example/system"):
        agreement(frames)


@pytest.mark.parametrize("unreviewed", [True, False])
def test_agreement_rejects_no_matching_reviewed_scores(agreement_case, unreviewed):
    if unreviewed:
        agreement_case[1]["reviewed"] = "no"
    else:
        agreement_case[1]["example_id"] = "unmatched"
        agreement_case[1]["system"] = ["a", "b", "c", "d"]
    with pytest.raises(ValueError, match="No matching"):
        agreement(agreement_case)


def test_agreement_reviewed_flag_normalization_and_optional_judge_metadata(agreement_case):
    machine, human, _, _ = agreement_case
    human["reviewed"] = " YES "
    machine["annotator"] = ""
    machine["reviewed"] = "no"
    assert agreement(agreement_case)["examples"].eq(4).all()


def test_agreement_legacy_unmatched_partial_row_does_not_break_metrics(agreement_case):
    machine, human, judge_csv, human_csv = agreement_case
    human = human.drop(columns=["reviewed", "cache_key", "annotator"])
    unmatched = human.iloc[[0]].copy()
    unmatched["example_id"] = "other"
    unmatched[list(judge.RUBRIC_FIELDS)] = ""
    human = pd.concat([human, unmatched], ignore_index=True)
    assert agreement((machine, human, judge_csv, human_csv))["examples"].eq(4).all()