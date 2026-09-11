from src.policy import PolicyConfig, decide
from src.schemas import AgentInput


CONFIG = PolicyConfig(
    intent_confidence_threshold=0.72,
    intent_margin_threshold=0.15,
    retrieval_similarity_threshold=0.35,
    always_escalate_intents=frozenset({"account_security", "billing_subscription"}),
    risk_terms=("hacked", "unauthorized", "legal"),
)


def test_auto_handles_supported_low_risk_message() -> None:
    result = decide(
        AgentInput("App will not play", "playback_app_issue", 0.90, 0.05, 0.70),
        CONFIG,
    )
    assert result.action == "auto_handle"


def test_escalates_risk_term_even_with_high_confidence() -> None:
    result = decide(
        AgentInput(
            "This was an unauthorized charge",
            "complaint_feedback",
            0.95,
            0.02,
            0.90,
        ),
        CONFIG,
    )
    assert result.action == "escalate"
    assert result.triggered_risks == ("unauthorized",)


def test_escalates_when_predictions_are_ambiguous() -> None:
    result = decide(
        AgentInput("It is not working", "other_unclear", 0.76, 0.68, 0.60),
        CONFIG,
    )
    assert result.action == "escalate"


def test_matches_whole_risk_words_only() -> None:
    result = decide(
        AgentInput("The legalized text is visible", "feature_availability", 0.90, 0.05, 0.70),
        CONFIG,
    )
    assert result.action == "auto_handle"
