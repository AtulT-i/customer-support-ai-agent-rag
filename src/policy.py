import re
from dataclasses import dataclass
from typing import Iterable

from src.schemas import AgentInput, PolicyDecision


@dataclass(frozen=True)
class PolicyConfig:
    intent_confidence_threshold: float
    intent_margin_threshold: float
    retrieval_similarity_threshold: float
    always_escalate_intents: frozenset[str]
    risk_terms: tuple[str, ...]
    auto_handle_intents: frozenset[str] = frozenset({
        "account_access", "playback_app_issue", "feature_availability"
    })


def _matched_risk_terms(text: str, terms: Iterable[str]) -> tuple[str, ...]:
    normalized = text.casefold()
    return tuple(
        term
        for term in terms
        if re.search(rf"\b{re.escape(term.casefold())}\b", normalized)
    )


def decide(input_: AgentInput, config: PolicyConfig) -> PolicyDecision:
    risks = _matched_risk_terms(input_.text, config.risk_terms)
    if risks:
        return PolicyDecision(
            action="escalate",
            reason="risk term requires human review",
            triggered_risks=risks,
        )

    if input_.intent in config.always_escalate_intents:
        return PolicyDecision(
            action="escalate",
            reason=f"{input_.intent} is configured for human review",
            triggered_risks=(),
        )

    if input_.intent not in config.auto_handle_intents:
        return PolicyDecision(
            action="escalate",
            reason="intent is not approved for automatic handling",
            triggered_risks=(),
        )

    if input_.intent_confidence < config.intent_confidence_threshold:
        return PolicyDecision(
            action="escalate",
            reason="intent confidence is below threshold",
            triggered_risks=(),
        )

    margin = input_.intent_confidence - input_.second_intent_confidence
    if margin < config.intent_margin_threshold:
        return PolicyDecision(
            action="escalate",
            reason="top intent predictions are too close",
            triggered_risks=(),
        )

    if input_.retrieval_similarity < config.retrieval_similarity_threshold:
        return PolicyDecision(
            action="escalate",
            reason="no sufficiently similar historical resolution",
            triggered_risks=(),
        )

    return PolicyDecision(
        action="auto_handle",
        reason="low-risk case with confident intent and supporting evidence",
        triggered_risks=(),
    )

