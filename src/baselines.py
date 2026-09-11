from __future__ import annotations

from collections import Counter

from src.labeling import keyword_intent


GENERIC_REPLY = (
    "Thanks for contacting support. A human support specialist will review this."
)


def trivial_prediction(majority_intent: str) -> dict[str, str]:
    return {
        "intent": majority_intent,
        "decision": "escalate",
        "draft_reply": GENERIC_REPLY,
    }


def keyword_prediction(text: str) -> dict[str, str]:
    intent, match = keyword_intent(text)
    sensitive = intent in {"account_security", "billing_subscription"}
    return {
        "intent": intent,
        "decision": "escalate" if sensitive or intent == "other_unclear" else "auto_handle",
        "draft_reply": (
            GENERIC_REPLY
            if sensitive
            else f"Thanks for contacting us about {intent.replace('_', ' ')}. "
            "Please try the relevant help-center steps."
        ),
        "rule_match": match,
    }


def majority_label(labels: list[str]) -> str:
    return Counter(labels).most_common(1)[0][0]
