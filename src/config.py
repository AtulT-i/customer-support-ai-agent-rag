from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.policy import PolicyConfig


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectConfig:
    brand: str
    random_seed: int
    candidate_brands: tuple[str, ...]
    intents: tuple[str, ...]
    policy: PolicyConfig
    data: dict[str, Path]
    retrieval: dict[str, int]
    reply_templates: dict[str, str]


def load_config(path: Path | None = None) -> ProjectConfig:
    config_path = path or ROOT / "config" / "project.yaml"
    with config_path.open(encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    policy = raw["policy"]
    return ProjectConfig(
        brand=raw["brand"],
        random_seed=int(raw["random_seed"]),
        candidate_brands=tuple(raw["candidate_brands"]),
        intents=tuple(raw["intents"]),
        policy=PolicyConfig(
            intent_confidence_threshold=float(policy["intent_confidence_threshold"]),
            intent_margin_threshold=float(policy["intent_margin_threshold"]),
            retrieval_similarity_threshold=float(
                policy["retrieval_similarity_threshold"]
            ),
            always_escalate_intents=frozenset(policy["always_escalate_intents"]),
            risk_terms=tuple(policy["risk_terms"]),
            auto_handle_intents=frozenset(policy.get("auto_handle_intents", (
                "account_access", "playback_app_issue", "feature_availability"
            ))),
        ),
        data={key: ROOT / value for key, value in raw["data"].items()},
        retrieval={key: int(value) for key, value in raw["retrieval"].items()},
        reply_templates=dict(raw["reply_templates"]),
    )
