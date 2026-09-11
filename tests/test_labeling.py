from src.labeling import keyword_intent

import pandas as pd

from src.labeling import create_weak_labels


def test_security_has_priority_over_billing() -> None:
    intent, _ = keyword_intent("My account was hacked and then charged")
    assert intent == "account_security"


def test_unknown_message_uses_other() -> None:
    intent, match = keyword_intent("Can somebody help me with this?")
    assert intent == "other_unclear"
    assert match == ""


def test_bootstrap_removes_redacted_empty_and_duplicate_messages() -> None:
    pairs = pd.DataFrame({"customer_text": [
        "@customer https://example.com", "<HANDLE> <URL>",
        "@one Restart the app", "@two restart the app", "Cannot login",
    ]})
    weak = create_weak_labels(pairs, 10, 42)
    assert len(weak) == 2
    assert set(weak["intent"]) == {"playback_app_issue", "account_access"}
