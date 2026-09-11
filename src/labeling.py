from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.text import normalize_for_matching


@dataclass(frozen=True)
class IntentRule:
    intent: str
    pattern: re.Pattern[str]


RULES = (
    IntentRule(
        "account_security",
        re.compile(r"\b(hack|hacked|stolen|fraud|unauthori[sz]ed|compromis|security)\w*\b", re.I),
    ),
    IntentRule(
        "plan_change_cancel",
        re.compile(r"\b(cancel|downgrade|upgrade|change plan|switch plan|premium plan)\w*\b", re.I),
    ),
    IntentRule(
        "billing_subscription",
        re.compile(r"\b(charge|charged|billing|payment|refund|invoice|subscription|money|price)\w*\b", re.I),
    ),
    IntentRule(
        "account_access",
        re.compile(r"\b(log ?in|sign ?in|password|reset|locked out|access account)\w*\b", re.I),
    ),
    IntentRule(
        "service_outage",
        re.compile(r"\b(down|outage|offline|everyone|service unavailable)\w*\b", re.I),
    ),
    IntentRule(
        "playback_app_issue",
        re.compile(r"\b(play|playing|pause|skip|crash|app|stream|download|sound|offline mode)\w*\b", re.I),
    ),
    IntentRule(
        "feature_availability",
        re.compile(r"\b(feature|available|support|playlist|lyrics|podcast|device|country)\w*\b", re.I),
    ),
    IntentRule(
        "complaint_feedback",
        re.compile(r"\b(awful|terrible|hate|annoy|frustrat|disappoint|feedback|please add)\w*\b", re.I),
    ),
)


def keyword_intent(text: str) -> tuple[str, str]:
    for rule in RULES:
        match = rule.pattern.search(text)
        if match:
            return rule.intent, match.group(0)
    return "other_unclear", ""


def create_weak_labels(
    pairs: pd.DataFrame, max_examples: int, random_seed: int
) -> pd.DataFrame:
    # Redaction can leave handle/link-only tweets with no usable model input.
    pairs = pairs.assign(_normalized=pairs["customer_text"].map(normalize_for_matching))
    pairs = pairs[pairs["_normalized"].ne("")].drop_duplicates("_normalized")
    pairs = pairs.drop(columns="_normalized")
    if len(pairs) > max_examples:
        pairs = pairs.sample(max_examples, random_state=random_seed)
    output = pairs.copy()
    inferred = output["customer_text"].map(keyword_intent)
    output["intent"] = inferred.map(lambda value: value[0])
    output["rule_match"] = inferred.map(lambda value: value[1])
    output["label_source"] = "weak_keyword_rule_not_human_ground_truth"
    return output


def split_reviewed_labels(
    reviewed_csv: str,
    train_csv: str,
    golden_csv: str,
    golden_size: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(reviewed_csv, dtype=str, keep_default_na=False)
    required = {"text", "intent", "decision", "thread_id", "reviewed"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Reviewed labels are missing columns: {sorted(missing)}")
    labels = labels[
        labels["reviewed"].astype(str).str.casefold().eq("yes")
    ].dropna(subset=["text", "intent", "decision", "thread_id"])
    for field in ("text", "intent", "decision", "thread_id"):
        if labels[field].str.strip().eq("").any():
            raise ValueError(f"Reviewed labels contain blank {field}.")
    if not labels["decision"].isin(["auto_handle", "escalate"]).all():
        raise ValueError("Reviewed decisions must be auto_handle or escalate.")
    labels = labels.assign(_normalized=labels["text"].map(normalize_for_matching))
    if labels["_normalized"].eq("").any():
        raise ValueError("Reviewed labels contain empty normalized text.")
    labels = labels.drop_duplicates("_normalized").drop(columns="_normalized")
    labels = labels.drop_duplicates("thread_id")
    if golden_size < 150 or golden_size > 250:
        raise ValueError("Golden size must be between 150 and 250.")
    if len(labels) < golden_size + 100:
        raise ValueError(
            f"Need at least {golden_size + 100} reviewed, unique-thread labels; "
            f"found {len(labels)}."
        )
    intent_counts = labels["intent"].value_counts()
    rare = intent_counts[intent_counts < 2]
    if not rare.empty:
        raise ValueError(
            "Every intent needs at least two reviewed examples before a "
            f"stratified split. Too small: {rare.to_dict()}"
        )

    train, golden = train_test_split(
        labels,
        test_size=golden_size,
        random_state=random_seed,
        stratify=labels["intent"],
    )
    overlap = set(train["thread_id"]) & set(golden["thread_id"])
    if overlap:
        raise AssertionError("Thread leakage detected while splitting labels.")
    train.to_csv(train_csv, index=False)
    golden.to_csv(golden_csv, index=False)
    return train, golden
