from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AgentInput:
    text: str
    intent: str
    intent_confidence: float
    second_intent_confidence: float
    retrieval_similarity: float


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason: str
    triggered_risks: Sequence[str]

