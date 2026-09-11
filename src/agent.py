from __future__ import annotations

from pathlib import Path

from src.config import ProjectConfig
from src.local_rag import select_evidence
from src.model import load_intent_model, predict_intent
from src.policy import decide
from src.retrieval import load_retrieval_index, retrieve
from src.schemas import AgentInput
from src.text import normalize_for_matching, normalize_text, safe_guidance


class SupportAgent:
    def __init__(self, config: ProjectConfig, ollama_model: str | None = None) -> None:
        self.config = config
        self.ollama_model = ollama_model
        self.model, self.model_metadata = load_intent_model(
            config.data["model_path"]
        )
        self.retrieval = load_retrieval_index(config.data["retrieval_path"])

    def respond(self, text: str) -> dict[str, object]:
        if not isinstance(text, str) or not normalize_for_matching(text):
            raise ValueError("Enter a customer message with usable text.")
        if len(text) > 4000:
            raise ValueError("Customer message must be 4000 characters or fewer.")
        text = normalize_text(text)
        intent, confidence, second_confidence = predict_intent(self.model, text)
        evidence = retrieve(self.retrieval, text, self.config.retrieval["top_k"])
        best_similarity = float(evidence[0]["similarity"]) if evidence else 0.0
        policy = decide(
            AgentInput(
                text=text,
                intent=intent,
                intent_confidence=confidence,
                second_intent_confidence=second_confidence,
                retrieval_similarity=best_similarity,
            ),
            self.config.policy,
        )
        action, reason = policy.action, policy.reason
        selected, mode = None, "not_used_for_escalation"
        if action == "auto_handle":
            candidates = []
            for item in evidence:
                guidance = safe_guidance(item["brand_reply"])
                if guidance and item["similarity"] >= self.config.policy.retrieval_similarity_threshold:
                    candidates.append({**item, "safe_guidance": guidance})
            selected, mode = select_evidence(text, candidates, self.ollama_model)
            if selected is None:
                action = "escalate"
                reason = "no safe applicable historical guidance" if mode != "ollama_abstained" else "local model abstained from evidence selection"
        used_evidence = [selected] if selected is not None else []
        draft = self._draft(intent, used_evidence, action)
        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "second_intent_confidence": round(second_confidence, 4),
            "decision": action,
            "reason": reason,
            "triggered_risks": list(policy.triggered_risks),
            "draft_reply": draft,
            "evidence": evidence,
            "evidence_tweet_ids": [item["customer_tweet_id"] for item in used_evidence],
            "rag_mode": mode,
            "ollama_model": self.ollama_model,
            "model_metadata": self.model_metadata,
        }

    def _draft(
        self,
        intent: str,
        evidence: list[dict[str, object]],
        action: str,
    ) -> str:
        template = self.config.reply_templates.get(
            intent, self.config.reply_templates["other_unclear"]
        )
        if action == "escalate":
            return f"{template} Human review is recommended; this demo has not opened a support ticket."
        if not evidence:
            return template
        guidance = safe_guidance(evidence[0]["brand_reply"])
        if not guidance:
            return template
        return f"{template} A similar historical support case suggested: {guidance}"


def require_artifacts(config: ProjectConfig) -> None:
    missing = [
        path
        for path in (
            config.data["model_path"],
            config.data["retrieval_path"],
        )
        if not Path(path).exists()
    ]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Build model artifacts first: {formatted}")
