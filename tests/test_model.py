import hashlib
from unittest.mock import Mock

import pandas as pd
import pytest

from src import model as model_module
from src.model import build_intent_pipeline, load_intent_model, predict_intent, train_intent_model


def test_model_trains_and_predicts() -> None:
    rows = []
    for index in range(6):
        rows.append((f"cannot login password {index}", "account_access"))
        rows.append((f"music will not play app {index}", "playback_app_issue"))
    frame = pd.DataFrame(rows, columns=["text", "intent"])
    model = build_intent_pipeline(42).fit(frame["text"], frame["intent"])
    intent, confidence, second = predict_intent(model, "password login problem")
    assert intent == "account_access"
    assert confidence > second


@pytest.fixture
def reviewed_labels():
    return pd.DataFrame([
        {
            "text": f"{'cannot login account' if index % 2 else 'music playback freezes'} example {index}",
            "intent": "account_access" if index % 2 else "playback_app_issue",
            "reviewed": "yes",
            "label_source": "human",
        }
        for index in range(100)
    ])


@pytest.mark.parametrize("reviewed", [None, "", "no", "true", "1"])
def test_training_requires_every_row_explicitly_reviewed(tmp_path, monkeypatch, reviewed_labels, reviewed):
    if reviewed is None:
        reviewed_labels = reviewed_labels.drop(columns="reviewed")
    else:
        reviewed_labels.loc[99, "reviewed"] = reviewed
    labels_csv = tmp_path / "labels.csv"
    output = tmp_path / "model.joblib"
    reviewed_labels.to_csv(labels_csv, index=False)
    builder = Mock(side_effect=AssertionError("Invalid labels must be rejected before fitting"))
    monkeypatch.setattr(model_module, "build_intent_pipeline", builder)
    with pytest.raises(ValueError, match="explicitly reviewed=yes"):
        train_intent_model(labels_csv, output, random_seed=42)
    builder.assert_not_called()
    assert not output.exists()


def test_reviewed_training_rejects_nonhuman_labels(tmp_path, monkeypatch, reviewed_labels):
    reviewed_labels.loc[99, "label_source"] = "weak"
    labels_csv = tmp_path / "labels.csv"
    reviewed_labels.to_csv(labels_csv, index=False)
    builder = Mock(side_effect=AssertionError("Nonhuman labels must not be fitted"))
    monkeypatch.setattr(model_module, "build_intent_pipeline", builder)
    with pytest.raises(ValueError, match="non-human labels"):
        train_intent_model(labels_csv, tmp_path / "model.joblib", random_seed=42)
    builder.assert_not_called()


def test_reviewed_training_rejects_normalized_duplicates_before_fitting(tmp_path, monkeypatch, reviewed_labels):
    reviewed_labels.loc[0, "text"] = "Music FREEZES @Alice person@example.com https://a.example 123456"
    reviewed_labels.loc[2, "text"] = "music  freezes @Bob other@example.org https://b.example 987654"
    labels_csv = tmp_path / "labels.csv"
    output = tmp_path / "model.joblib"
    reviewed_labels.to_csv(labels_csv, index=False)
    builder = Mock(side_effect=AssertionError("Normalized duplicates must be rejected before fitting"))
    monkeypatch.setattr(model_module, "build_intent_pipeline", builder)
    with pytest.raises(ValueError, match="Deduplicate normalized training text"):
        train_intent_model(labels_csv, output, random_seed=42)
    builder.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("allow_weak", [False, True])
def test_training_metadata_round_trips_exact_input_hash(tmp_path, reviewed_labels, allow_weak):
    if allow_weak:
        reviewed_labels = reviewed_labels.drop(columns="reviewed")
        reviewed_labels["label_source"] = "weak"
    else:
        reviewed_labels.loc[0, "reviewed"] = " YES "
    labels_csv = tmp_path / "labels.csv"
    output = tmp_path / "artifacts" / "model.joblib"
    reviewed_labels.to_csv(labels_csv, index=False)
    trained = train_intent_model(labels_csv, output, random_seed=42, allow_weak_labels=allow_weak)
    loaded, metadata = load_intent_model(output)
    assert metadata == {
        "label_source": "weak_provisional" if allow_weak else "human_reviewed",
        "training_examples": 100,
        "training_data_sha256": hashlib.sha256(labels_csv.read_bytes()).hexdigest(),
    }
    assert len(metadata["training_data_sha256"]) == 64
    query = "music playback freezes"
    assert predict_intent(loaded, query) == predict_intent(trained, query)
    assert predict_intent(loaded, query)[0] == "playback_app_issue"
