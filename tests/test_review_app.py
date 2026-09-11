"""Exercise the real review script against isolated CSVs, without launching a server."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import config as config_module
from src.human_scoring import WORKSHEET_COLUMNS


REVIEW_APP = Path(__file__).resolve().parents[1] / "review_app.py"
PREPARATION_MESSAGE = "First run evaluation, then judge, then prepare-human-scores."


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


def snapshot(directory):
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*") if path.is_file()
    }


@pytest.fixture
def review_case(tmp_path, monkeypatch):
    base = config_module.load_config()
    config = replace(base, data={key: tmp_path / key / path.name for key, path in base.data.items()})
    path = config.data["label_queue_csv"]
    path.parent.mkdir(parents=True)
    pd.DataFrame([{
        "example_id": "001", "text": "My music stops playing after a minute.",
        "thread_id": "thread-001", "intent": "", "decision": "", "reviewed": "no",
        "historical_reply": "Restart the app; this is historical context only.",
        "escalation_reason": "", "required_reply_points": "", "forbidden_claims": "",
        "annotator": "", "label_source": "unreviewed",
    }]).to_csv(path, index=False)
    loader = Mock(return_value=config)
    monkeypatch.setattr(config_module, "load_config", loader)
    return SimpleNamespace(config=config, root=tmp_path, loader=loader)


def test_initial_annotation_view_has_no_writes_or_invented_labels(review_case):
    before = snapshot(review_case.root)
    app = AppTest.from_file(str(REVIEW_APP)).run()
    assert snapshot(review_case.root) == before
    assert not app.exception
    review_case.loader.assert_called_once_with()
    assert app.title[0].value == "Human review workspace"
    assert app.sidebar.radio[0].value == "Intent and decision labels"
    assert app.sidebar.text_input[0].value == ""
    assert any("0 reviewed; 0 usable unique-thread rows. Minimum 300" in item.value for item in app.info)
    assert app.selectbox[0].value == "001"
    assert app.selectbox[1].value == "Choose an intent"
    assert app.selectbox[2].value == "Choose a decision"
    assert all(not widget.value for widget in app.checkbox)
    assert all(widget.value == "" for widget in app.text_area)
    assert any("My music stops playing" in item.value for item in app.text)
    assert any("Single reviewer/editor" in item.value for item in app.warning)
    assert not review_case.config.data["human_scores_csv"].exists()


def test_annotation_save_requires_personal_confirmation_without_writes(review_case):
    app = AppTest.from_file(str(REVIEW_APP)).run()
    assert not app.exception
    before = snapshot(review_case.root)
    app.button[0].click().run()
    assert snapshot(review_case.root) == before
    assert not app.exception
    assert [item.value for item in app.error] == ["Confirm your personal review before saving."]


@pytest.mark.parametrize("worksheet_state", ["missing", "legacy_headers", "current_headers"])
def test_switch_to_scoring_without_prepared_worksheet_is_read_only(review_case, worksheet_state):
    path = review_case.config.data["human_scores_csv"]
    if worksheet_state != "missing":
        path.parent.mkdir(parents=True)
        columns = WORKSHEET_COLUMNS if worksheet_state == "current_headers" else [
            "example_id", "system", "relevance", "actionability", "grounding",
            "tone", "safety", "annotator", "reviewed",
        ]
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    before = snapshot(review_case.root)
    app = AppTest.from_file(str(REVIEW_APP)).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Reply quality scores").run()
    assert snapshot(review_case.root) == before
    assert not app.exception, [item.message for item in app.exception]
    assert app.sidebar.radio[0].value == "Reply quality scores"
    assert any(PREPARATION_MESSAGE in item.value for item in app.info)
    assert len(app.button) == 0
    assert len(app.selectbox) == 0