from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from src.text import normalize_text, sanitize_reply


RAW_COLUMNS = (
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
)


def _require_raw_csv(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download twcs.csv from Kaggle first."
        )


def profile_brands(
    raw_csv: Path, brands: tuple[str, ...], chunksize: int = 200_000
) -> pd.DataFrame:
    _require_raw_csv(raw_csv)
    counts = {
        brand: {
            "brand": brand,
            "brand_messages": 0,
            "direct_replies": 0,
            "unique_customers_replied_to": 0,
        }
        for brand in brands
    }
    customer_ids: dict[str, set[int]] = {brand: set() for brand in brands}

    for chunk in pd.read_csv(
        raw_csv,
        usecols=["tweet_id", "author_id", "inbound", "in_response_to_tweet_id"],
        chunksize=chunksize,
        low_memory=False,
    ):
        for brand in brands:
            rows = chunk[(chunk["author_id"] == brand) & (~chunk["inbound"])]
            replies = rows[rows["in_response_to_tweet_id"].notna()]
            counts[brand]["brand_messages"] += len(rows)
            counts[brand]["direct_replies"] += len(replies)
            customer_ids[brand].update(
                replies["in_response_to_tweet_id"].astype("int64").tolist()
            )

    for brand in brands:
        counts[brand]["unique_customers_replied_to"] = len(customer_ids[brand])
    return pd.DataFrame(counts.values()).sort_values(
        "direct_replies", ascending=False
    )


def _load_parent_map(raw_csv: Path, chunksize: int) -> dict[int, int]:
    parents: dict[int, int] = {}
    for chunk in pd.read_csv(
        raw_csv,
        usecols=["tweet_id", "in_response_to_tweet_id"],
        chunksize=chunksize,
    ):
        linked = chunk[chunk["in_response_to_tweet_id"].notna()]
        parents.update(
            zip(
                linked["tweet_id"].astype("int64"),
                linked["in_response_to_tweet_id"].astype("int64"),
            )
        )
    return parents


def _root_id(tweet_id: int, parents: dict[int, int], max_depth: int = 100) -> int:
    current = tweet_id
    visited: set[int] = set()
    for _ in range(max_depth):
        if current in visited or current not in parents:
            break
        visited.add(current)
        current = parents[current]
    return current


def prepare_pairs(
    raw_csv: Path,
    output_csv: Path,
    brand: str,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    _require_raw_csv(raw_csv)
    brand_replies: list[pd.DataFrame] = []
    for chunk in pd.read_csv(raw_csv, usecols=RAW_COLUMNS, chunksize=chunksize):
        replies = chunk[
            (chunk["author_id"] == brand)
            & (~chunk["inbound"])
            & (chunk["in_response_to_tweet_id"].notna())
        ].copy()
        if not replies.empty:
            brand_replies.append(replies)

    if not brand_replies:
        raise ValueError(f"No direct replies found for brand {brand!r}.")

    replies = pd.concat(brand_replies, ignore_index=True)
    replies["customer_tweet_id"] = replies["in_response_to_tweet_id"].astype("int64")
    wanted_customer_ids = set(replies["customer_tweet_id"])

    customer_messages: list[pd.DataFrame] = []
    for chunk in pd.read_csv(raw_csv, usecols=RAW_COLUMNS, chunksize=chunksize):
        matching = chunk[chunk["tweet_id"].isin(wanted_customer_ids)].copy()
        if not matching.empty:
            customer_messages.append(matching)

    customers = pd.concat(customer_messages, ignore_index=True)
    customers = customers[
        ["tweet_id", "author_id", "created_at", "text"]
    ].rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "author_id": "customer_author_id",
            "created_at": "customer_created_at",
            "text": "customer_text",
        }
    )
    reply_view = replies[
        ["tweet_id", "customer_tweet_id", "created_at", "text"]
    ].rename(
        columns={
            "tweet_id": "reply_tweet_id",
            "created_at": "reply_created_at",
            "text": "brand_reply",
        }
    )
    pairs = customers.merge(reply_view, on="customer_tweet_id", how="inner")
    pairs = pairs.sort_values(["customer_tweet_id", "reply_created_at"]).drop_duplicates(
        "customer_tweet_id", keep="first"
    )
    pairs["customer_text"] = pairs["customer_text"].map(normalize_text)
    pairs["brand_reply"] = pairs["brand_reply"].map(sanitize_reply)
    pairs = pairs[
        (pairs["customer_text"].str.len() >= 8)
        & (pairs["brand_reply"].str.len() >= 8)
        & (~pairs["customer_text"].str.startswith("RT "))
    ]
    pairs = pairs.drop_duplicates("customer_text").reset_index(drop=True)

    parents = _load_parent_map(raw_csv, chunksize)
    pairs["thread_id"] = [
        _root_id(int(tweet_id), parents)
        for tweet_id in pairs["customer_tweet_id"]
    ]
    pairs["brand"] = brand
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return pairs


def make_label_queue(
    pairs_csv: Path,
    output_csv: Path,
    sample_size: int,
    random_seed: int,
) -> pd.DataFrame:
    pairs = pd.read_csv(pairs_csv)
    sample_size = min(sample_size, len(pairs))
    sampled = pairs.sample(sample_size, random_state=random_seed).copy()
    queue = pd.DataFrame(
        {
            "example_id": [f"label-{index:04d}" for index in range(sample_size)],
            "tweet_id": sampled["customer_tweet_id"].astype(str),
            "thread_id": sampled["thread_id"].astype(str),
            "text": sampled["customer_text"],
            "historical_reply": sampled["brand_reply"],
            "intent": "",
            "secondary_intent": "",
            "decision": "",
            "escalation_reason": "",
            "required_reply_points": "",
            "forbidden_claims": "",
            "annotator_confidence": "",
            "annotator": "",
            "reviewed": "no",
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output_csv, index=False)
    return queue
