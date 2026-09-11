import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.evaluation import _system_metrics, _validate_golden, evaluate_all


@pytest.fixture
def evaluation_case(tmp_path):
    golden = pd.DataFrame([
        {
            "example_id": f"gold-{index}",
            "text": f"Golden customer request number {index}",
            "intent": "account_access",
            "decision": "escalate" if index % 2 else "auto_handle",
            "thread_id": f"gold-thread-{index}",
            "tweet_id": f"00{index}",
            "reviewed": "yes",
        }
        for index in range(150)
    ])
    training = pd.DataFrame([
        {
            "example_id": f"train-{index}",
            "text": f"Training customer problem number {index}",
            "intent": "account_access",
            "decision": "auto_handle",
            "thread_id": f"train-thread-{index}",
            "tweet_id": f"train-tweet-{index}",
            "reviewed": "yes",
            "label_source": "human",
        }
        for index in range(100)
    ])
    golden_csv = tmp_path / "golden.csv"
    training_csv = tmp_path / "train.csv"
    golden.to_csv(golden_csv, index=False)
    training.to_csv(training_csv, index=False)
    agent = SimpleNamespace(
        config=SimpleNamespace(intents=("account_access", "other_unclear")),
        model_metadata={
            "label_source": "human_reviewed",
            "training_data_sha256": hashlib.sha256(training_csv.read_bytes()).hexdigest(),
        },
        retrieval={
            "version": 2,
            "excluded_golden_sha256": hashlib.sha256(golden_csv.read_bytes()).hexdigest(),
            "pairs": pd.DataFrame([{
                "thread_id": "retrieval-thread",
                "customer_tweet_id": "retrieval-tweet",
                "customer_text": "Independent historical question",
            }]),
        },
        respond=Mock(return_value={
            "intent": "account_access", "decision": "auto_handle",
            "draft_reply": "Please try account recovery.",
            "evidence": [
                {"customer_tweet_id": f"retrieval-{i}", "similarity": 0.9 - i / 10}
                for i in range(3)
            ],
        }),
    )
    return SimpleNamespace(
        golden=golden, training=training, golden_csv=golden_csv,
        training_csv=training_csv, agent=agent, output_dir=tmp_path / "results",
    )


def evaluate(case):
    return evaluate_all(case.golden_csv, case.training_csv, case.agent, case.output_dir)


def save_training(case):
    case.training.to_csv(case.training_csv, index=False)
    case.agent.model_metadata["training_data_sha256"] = hashlib.sha256(
        case.training_csv.read_bytes()
    ).hexdigest()


def test_metrics_use_position_with_index_gaps_and_save_report(tmp_path):
    golden = pd.DataFrame({
        "intent": ["a", "b", "a", "b"],
        "decision": ["escalate", "escalate", "auto_handle", "auto_handle"],
    }, index=[2, 5, 8, 11])
    predictions = [
        {"intent": intent, "decision": decision}
        for intent, decision in zip(["a", "a", "a", "b"],
                                    ["auto_handle", "escalate", "auto_handle", "auto_handle"])
    ]
    metrics = _system_metrics("test", golden, predictions, tmp_path)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["false_auto_handle_count"] == 1
    assert metrics["must_escalate_count"] == 2
    assert metrics["auto_handle_count"] == 3
    assert metrics["false_auto_handle_rate"] == pytest.approx(0.5)
    assert metrics["false_auto_handle_fraction_among_auto_handled"] == pytest.approx(1 / 3)
    assert metrics["auto_handle_coverage"] == pytest.approx(0.75)
    report = json.loads((tmp_path / "per_intent_report_test.json").read_text())
    assert report["a"]["support"] == 2
    assert report["b"]["recall"] == pytest.approx(0.5)
    assert (tmp_path / "confusion_test.csv").exists()


@pytest.mark.parametrize("decision", ["auto_handle", "escalate"])
def test_zero_denominators_are_reported_with_counts(tmp_path, decision):
    golden = pd.DataFrame({"intent": ["a"], "decision": [decision]})
    metrics = _system_metrics("test", golden, [{"intent": "a", "decision": decision}], tmp_path)
    assert metrics["false_auto_handle_count"] == 0
    assert metrics["false_auto_handle_rate"] == 0
    assert metrics["false_auto_handle_fraction_among_auto_handled"] == 0


@pytest.mark.parametrize("field", ["example_id", "text", "intent", "decision", "thread_id", "reviewed"])
@pytest.mark.parametrize("value", [None, " \t"])
def test_golden_rejects_every_blank_required_field(evaluation_case, field, value):
    evaluation_case.golden.loc[0, field] = value
    with pytest.raises(ValueError, match="blank required field"):
        _validate_golden(evaluation_case.golden)


@pytest.mark.parametrize("field", ["example_id", "text", "intent", "decision", "thread_id", "reviewed"])
def test_golden_requires_all_columns(evaluation_case, field):
    with pytest.raises(ValueError, match="missing columns"):
        _validate_golden(evaluation_case.golden.drop(columns=field))


@pytest.mark.parametrize("size", [149, 251])
def test_golden_size_limits(evaluation_case, size):
    golden = evaluation_case.golden.reindex(range(size))
    golden = golden.fillna("extra")
    with pytest.raises(ValueError, match="150-250"):
        _validate_golden(golden)


@pytest.mark.parametrize("field,value,message", [
    ("reviewed", "no", "reviewed=yes"),
    ("decision", "ignore", "invalid decisions"),
    ("intent", "unknown", "configured intents"),
    ("example_id", "gold-1", "duplicate example_id"),
    ("text", "@Someone GOLDEN CUSTOMER REQUEST NUMBER 1", "duplicate normalized texts"),
])
def test_golden_rejects_invalid_values(evaluation_case, field, value, message):
    evaluation_case.golden.loc[0, field] = value
    with pytest.raises(ValueError, match=message):
        _validate_golden(evaluation_case.golden, evaluation_case.agent.config.intents)


def test_evaluate_rejects_malformed_row_instead_of_dropping(evaluation_case):
    # There would still be 150 valid rows if the malformed row were silently dropped.
    bad_row = evaluation_case.golden.iloc[[0]].copy()
    bad_row["text"] = ""
    pd.concat([evaluation_case.golden, bad_row]).to_csv(evaluation_case.golden_csv, index=False)
    with pytest.raises(ValueError, match="blank required field"):
        evaluate(evaluation_case)
    evaluation_case.agent.respond.assert_not_called()
    assert not evaluation_case.output_dir.exists()


@pytest.mark.parametrize("source", ["weak_provisional", "human", None])
def test_weak_and_legacy_models_blocked_before_inference(evaluation_case, source):
    evaluation_case.agent.model_metadata["label_source"] = source
    with pytest.raises(ValueError, match="human_reviewed"):
        evaluate(evaluation_case)
    evaluation_case.agent.respond.assert_not_called()


@pytest.mark.parametrize("fingerprint", [None, "stale-hash"])
def test_training_fingerprint_required(evaluation_case, fingerprint):
    evaluation_case.agent.model_metadata["training_data_sha256"] = fingerprint
    with pytest.raises(ValueError, match="training_data_sha256"):
        evaluate(evaluation_case)
    evaluation_case.agent.respond.assert_not_called()


def test_training_file_bytes_must_match_model(evaluation_case):
    evaluation_case.training.loc[0, "text"] = "Edited after model training"
    evaluation_case.training.to_csv(evaluation_case.training_csv, index=False)
    with pytest.raises(ValueError, match="training_data_sha256"):
        evaluate(evaluation_case)


@pytest.mark.parametrize("field,value,message", [
    ("reviewed", "no", "reviewed=yes"),
    ("label_source", "weak_keyword_rule_not_human_ground_truth", "label_source must be human"),
    ("text", "", "blank required field"),
    ("intent", "", "blank required field"),
    ("text", "@Someone TRAINING CUSTOMER PROBLEM NUMBER 1", "duplicate normalized texts"),
    ("thread_id", "gold-thread-0", "Training leakage: thread_id"),
    ("tweet_id", "000", "Training leakage: tweet_id"),
    ("text", "@Someone GOLDEN CUSTOMER REQUEST NUMBER 0", "Training leakage: normalized text"),
])
def test_training_review_and_leakage_checks(evaluation_case, field, value, message):
    evaluation_case.training.loc[0, field] = value
    save_training(evaluation_case)
    with pytest.raises(ValueError, match=message):
        evaluate(evaluation_case)
    evaluation_case.agent.respond.assert_not_called()


def test_training_customer_tweet_id_alias_checked(evaluation_case):
    evaluation_case.training = evaluation_case.training.rename(columns={"tweet_id": "customer_tweet_id"})
    evaluation_case.training.loc[0, "customer_tweet_id"] = "000"
    save_training(evaluation_case)
    with pytest.raises(ValueError, match="Training leakage: customer_tweet_id"):
        evaluate(evaluation_case)


@pytest.mark.parametrize("field,value,message", [
    ("version", None, "old indexes"),
    ("version", 1, "old indexes"),
    ("excluded_golden_sha256", None, "excluded_golden_sha256"),
    ("excluded_golden_sha256", "stale-hash", "excluded_golden_sha256"),
    ("pairs", None, "pairs DataFrame"),
])
def test_retrieval_artifact_provenance(evaluation_case, field, value, message):
    evaluation_case.agent.retrieval[field] = value
    with pytest.raises(ValueError, match=message):
        evaluate(evaluation_case)
    evaluation_case.agent.respond.assert_not_called()


@pytest.mark.parametrize("field,value,message", [
    ("thread_id", "gold-thread-0", "thread_id"),
    ("customer_tweet_id", "000", "customer_tweet_id"),
    ("customer_text", "@Someone GOLDEN CUSTOMER REQUEST NUMBER 0 <URL>", "normalized customer text"),
])
def test_retrieval_leakage(evaluation_case, field, value, message):
    evaluation_case.agent.retrieval["pairs"].loc[0, field] = value
    with pytest.raises(ValueError, match=f"Retrieval leakage: {message}"):
        evaluate(evaluation_case)
    evaluation_case.agent.respond.assert_not_called()


def test_retrieval_requires_thread_metadata(evaluation_case):
    evaluation_case.agent.retrieval["pairs"] = evaluation_case.agent.retrieval["pairs"].drop(columns="thread_id")
    with pytest.raises(ValueError, match="missing columns"):
        evaluate(evaluation_case)


def test_valid_evaluation_saves_inspection_outputs(evaluation_case):
    # label_source is optional in the CSV, but reviewed and model provenance are not.
    evaluation_case.training = evaluation_case.training.drop(columns="label_source")
    save_training(evaluation_case)
    result = evaluate(evaluation_case)
    assert set(result["system"]) == {"trivial", "keyword", "main"}
    assert evaluation_case.agent.respond.call_count == 150
    predictions = pd.read_csv(evaluation_case.output_dir / "predictions.csv")
    assert len(predictions) == 450
    main = predictions[predictions["system"].eq("main")].iloc[0]
    assert main["retrieval_top1_id"] == "retrieval-0"
    assert main["retrieval_top1_similarity"] == pytest.approx(0.9)
    assert json.loads(main["retrieval_top3_ids"]) == ["retrieval-0", "retrieval-1", "retrieval-2"]
    assert json.loads(main["retrieval_top3_similarities"]) == pytest.approx([0.9, 0.8, 0.7])
    baseline = predictions[predictions["system"].eq("trivial")].iloc[0]
    assert pd.isna(baseline["retrieval_top1_similarity"])
    assert json.loads(baseline["retrieval_top3_ids"]) == []
    for system in result["system"]:
        assert (evaluation_case.output_dir / f"per_intent_report_{system}.json").exists()
    assert (evaluation_case.output_dir / "headline_metrics.csv").exists()