import json
import urllib.error
from unittest.mock import Mock

import pytest

from src.local_rag import select_evidence


@pytest.fixture
def candidates():
    return [
        {"customer_text": "app freezes", "safe_guidance": "Restart the app.", "customer_tweet_id": "11"},
        {"customer_text": "old app version", "safe_guidance": "Try updating the app.", "customer_tweet_id": "22"},
    ]


@pytest.fixture
def transport(monkeypatch):
    response = Mock()
    response.read.return_value = json.dumps({"response": '{"index": 1}'}).encode()
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    opener = Mock()
    opener.open.return_value = context
    factory = Mock(return_value=opener)
    monkeypatch.setattr("urllib.request.build_opener", factory)
    return factory, opener, response


@pytest.mark.parametrize("index", [0, 1])
def test_valid_integer_selects_exact_candidate_and_uses_loopback_without_proxies(candidates, transport, index):
    factory, opener, response = transport
    response.read.return_value = json.dumps({"response": json.dumps({"index": index})}).encode()
    selected, mode = select_evidence("app trouble", candidates, "offline-test-model")
    assert selected is candidates[index]
    assert mode == "ollama_selected"
    factory.assert_called_once()
    assert factory.call_args.args[0].proxies == {}
    opener.open.assert_called_once()
    request = opener.open.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:11434/api/generate"
    assert request.get_method() == "POST"
    assert opener.open.call_args.kwargs == {"timeout": 30}
    payload = json.loads(request.data)
    assert payload["model"] == "offline-test-model"
    assert payload["stream"] is False
    assert payload["format"] == "json"
    assert payload["options"]["temperature"] == 0
    prompt = json.loads(payload["prompt"])
    assert prompt["customer"] == "app trouble"
    assert prompt["candidates"] == [
        {"index": i, "customer": item["customer_text"], "guidance": item["safe_guidance"]}
        for i, item in enumerate(candidates)
    ]
    response.read.assert_called_once_with(65536)


@pytest.mark.parametrize("body", [
    b"not json",
    b'{"response": "not json"}',
    b"{}",
    b'{"response": "{}"}',
    b'{"response": "null"}',
    b'{"response": "[]"}',
])
def test_malformed_model_output_falls_back_without_generated_claims(candidates, transport, body):
    _, _, response = transport
    response.read.return_value = body
    selected, mode = select_evidence("app trouble", candidates, "offline-test-model")
    assert selected is candidates[0]
    assert mode == "extractive_fallback"


@pytest.mark.parametrize("index", [-2, 2, 999, True, False, 1.0, "1", None])
def test_out_of_range_and_noninteger_indices_fall_back(candidates, transport, index):
    _, _, response = transport
    response.read.return_value = json.dumps({"response": json.dumps({"index": index})}).encode()
    selected, mode = select_evidence("app trouble", candidates, "offline-test-model")
    assert selected is candidates[0]
    assert mode == "extractive_fallback"


@pytest.mark.parametrize("error", [TimeoutError("timed out"), urllib.error.URLError("model unavailable")])
def test_transport_failure_uses_extractive_fallback(candidates, transport, error):
    _, opener, _ = transport
    opener.open.side_effect = error
    selected, mode = select_evidence("app trouble", candidates, "offline-test-model")
    assert selected is candidates[0]
    assert mode == "extractive_fallback"


def test_minus_one_abstains_without_falling_back(candidates, transport):
    _, _, response = transport
    response.read.return_value = json.dumps({"response": json.dumps({"index": -1})}).encode()
    assert select_evidence("app trouble", candidates, "offline-test-model") == (None, "ollama_abstained")


@pytest.mark.parametrize("model", [None, ""])
def test_no_model_defaults_to_extractive_without_network(candidates, transport, model):
    factory, _, _ = transport
    selected, mode = select_evidence("app trouble", candidates, model)
    assert selected is candidates[0]
    assert mode == "extractive"
    factory.assert_not_called()


def test_no_candidates_returns_no_evidence_without_network(transport):
    factory, _, _ = transport
    assert select_evidence("app trouble", [], "offline-test-model") == (None, "no_safe_evidence")
    factory.assert_not_called()