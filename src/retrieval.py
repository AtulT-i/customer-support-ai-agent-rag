from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text import normalize_for_matching, safe_guidance


def exclude_held_out(pairs: pd.DataFrame, held_out: pd.DataFrame) -> pd.DataFrame:
    """Remove complete held-out threads, IDs and normalized duplicate messages."""
    if held_out.empty:
        return pairs.copy()
    if not {"thread_id", "text"}.issubset(held_out.columns):
        raise ValueError("Held-out labels require thread_id and text for safe retrieval.")
    if "thread_id" not in pairs.columns:
        raise ValueError("Re-prepare pairs with thread_id before building retrieval.")
    remove = pairs["thread_id"].astype(str).isin(held_out["thread_id"].astype(str))
    remove |= pairs["customer_text"].map(normalize_for_matching).isin(
        held_out["text"].map(normalize_for_matching)
    )
    if "tweet_id" in held_out.columns:
        remove |= pairs["customer_tweet_id"].astype(str).isin(held_out["tweet_id"].astype(str))
    return pairs.loc[~remove].copy()


def build_retrieval_index(
    pairs_csv: Path,
    output_path: Path,
    max_corpus_size: int,
    max_features: int,
    random_seed: int,
    golden_csv: Path | None = None,
) -> None:
    if max_corpus_size < 1 or max_features < 1:
        raise ValueError("Retrieval corpus size and feature limit must be positive.")
    pairs = pd.read_csv(pairs_csv, dtype=str, keep_default_na=False)
    required = {"customer_tweet_id", "customer_text", "brand_reply", "thread_id"}
    if missing := required - set(pairs.columns):
        raise ValueError(f"Missing retrieval columns: {sorted(missing)}")
    original_size = len(pairs)
    golden_hash = None
    if golden_csv is not None:
        held_out = pd.read_csv(golden_csv, dtype=str, keep_default_na=False)
        pairs = exclude_held_out(pairs, held_out)
        golden_hash = hashlib.sha256(golden_csv.read_bytes()).hexdigest()
    excluded_count = original_size - len(pairs)
    pairs["_normalized"] = pairs["customer_text"].map(normalize_for_matching)
    pairs = pairs[pairs["_normalized"].ne("") & pairs["brand_reply"].str.strip().ne("")]
    pairs = pairs.drop_duplicates("_normalized").drop(columns="_normalized")
    if len(pairs) > max_corpus_size:
        pairs = pairs.sample(max_corpus_size, random_state=random_seed)
    pairs = pairs.reset_index(drop=True)
    if pairs.empty:
        raise ValueError("No retrieval examples remain after cleaning and held-out exclusion.")
    pairs["safe_guidance"] = pairs["brand_reply"].map(safe_guidance)
    word = TfidfVectorizer(
        preprocessor=normalize_for_matching, ngram_range=(1, 2),
        min_df=1, sublinear_tf=True, max_features=max_features,
    )
    char = TfidfVectorizer(
        preprocessor=normalize_for_matching, analyzer="char_wb", ngram_range=(3, 5),
        min_df=1, sublinear_tf=True, max_features=max_features,
    )
    word_matrix = word.fit_transform(pairs["customer_text"])
    char_matrix = char.fit_transform(pairs["customer_text"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": 2, "vectorizer": word, "matrix": word_matrix,
            "char_vectorizer": char, "char_matrix": char_matrix,
            "word_weight": 0.65, "pairs": pairs,
            "excluded_golden_sha256": golden_hash,
            "excluded_rows": excluded_count,
        }, output_path,
    )


def retrieve(
    bundle: dict[str, object], text: str, top_k: int
) -> list[dict[str, object]]:
    if bundle.get("version") != 2:
        raise ValueError("Old retrieval index: rebuild with bootstrap or train for local RAG v2.")
    if top_k < 1 or not normalize_for_matching(text):
        return []
    word = bundle["vectorizer"]
    char = bundle["char_vectorizer"]
    word_scores = cosine_similarity(word.transform([text]), bundle["matrix"]).ravel()
    char_scores = cosine_similarity(char.transform([text]), bundle["char_matrix"]).ravel()
    weight = float(bundle["word_weight"])
    scores = weight * word_scores + (1 - weight) * char_scores
    pairs: pd.DataFrame = bundle["pairs"]
    results = []
    seen_threads = set()
    for index in (-scores).argsort(kind="stable"):
        if scores[index] <= 0:
            break
        row = pairs.iloc[index]
        thread = str(row["thread_id"])
        if thread in seen_threads:
            continue
        seen_threads.add(thread)
        results.append({
            "customer_tweet_id": str(row["customer_tweet_id"]),
            "thread_id": thread,
            "customer_text": str(row["customer_text"]),
            "brand_reply": str(row["brand_reply"]),
            "safe_guidance": str(row["safe_guidance"]),
            "similarity": float(scores[index]),
            "word_similarity": float(word_scores[index]),
            "char_similarity": float(char_scores[index]),
        })
        if len(results) == top_k:
            break
    return results


def load_retrieval_index(path: Path) -> dict[str, object]:
    bundle = joblib.load(path)
    if bundle.get("version") != 2:
        raise ValueError("Old retrieval index: rebuild with bootstrap or train for local RAG v2.")
    return bundle
