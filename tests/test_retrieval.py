import hashlib

import joblib
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from src import retrieval


def pair(tweet_id, thread_id, text, reply="Please restart the app."):
    return {
        "customer_tweet_id": tweet_id,
        "thread_id": thread_id,
        "customer_text": text,
        "brand_reply": reply,
    }


@pytest.fixture
def build_index(tmp_path):
    def build(rows, held_out=None):
        pairs_csv = tmp_path / "pairs.csv"
        output_path = tmp_path / "retrieval.joblib"
        pd.DataFrame(rows).to_csv(pairs_csv, index=False)
        golden_csv = None
        if held_out is not None:
            golden_csv = tmp_path / "golden.csv"
            pd.DataFrame(held_out).to_csv(golden_csv, index=False)
        retrieval.build_retrieval_index(
            pairs_csv, output_path, max_corpus_size=100,
            max_features=10000, random_seed=42, golden_csv=golden_csv,
        )
        return retrieval.load_retrieval_index(output_path)

    return build


def test_held_out_threads_ids_and_normalized_duplicates_excluded_before_fit(
    build_index, monkeypatch, tmp_path,
):
    kept = [
        pair("101", "train-a", "music playback freezes"),
        pair("102", "train-b", "download unavailable offline"),
    ]
    excluded = [
        pair("201", "held-thread", "threadonlytoken first message"),
        pair("202", "held-thread", "siblingonlytoken later message"),
        pair("999", "unrelated-thread", "idonlytoken different wording"),
        pair("204", "duplicate-thread", "DUPLICATEONLYTOKEN freezes @Alice 123456"),
    ]
    held_out = [{
        "tweet_id": "999", "thread_id": "held-thread",
        "text": "duplicateonlytoken  FREEZES @Bob 987654",
    }]
    fitted_documents = []
    original_fit = retrieval.TfidfVectorizer.fit_transform

    def record_fit(vectorizer, documents, *args, **kwargs):
        documents = list(documents)
        fitted_documents.append((vectorizer.analyzer, documents))
        return original_fit(vectorizer, documents, *args, **kwargs)

    monkeypatch.setattr(retrieval.TfidfVectorizer, "fit_transform", record_fit)
    bundle = build_index(excluded + kept, held_out)

    expected = [row["customer_text"] for row in kept]
    assert fitted_documents == [("word", expected), ("char_wb", expected)]
    assert bundle["pairs"]["customer_tweet_id"].tolist() == ["101", "102"]
    assert bundle["excluded_rows"] == 4
    assert bundle["excluded_golden_sha256"] == hashlib.sha256(
        (tmp_path / "golden.csv").read_bytes()
    ).hexdigest()
    assert bundle["matrix"].shape[0] == bundle["char_matrix"].shape[0] == 2
    for token in ("threadonlytoken", "siblingonlytoken", "idonlytoken", "duplicateonlytoken"):
        assert token not in bundle["vectorizer"].vocabulary_


def test_index_deduplicates_normalized_text_and_removes_empty_rows(build_index):
    bundle = build_index([
        pair("1", "a", "Music freezes @Alice 123456"),
        pair("2", "b", "MUSIC  FREEZES @Bob 987654"),
        pair("3", "c", "@Alice https://example.com 123456"),
        pair("4", "d", "download stuck", "  "),
        pair("5", "e", "offline download unavailable"),
    ])
    assert bundle["pairs"]["customer_tweet_id"].tolist() == ["1", "5"]
    assert bundle["pairs"]["safe_guidance"].tolist() == [
        "Please restart the app.", "Please restart the app.",
    ]
    assert bundle["excluded_golden_sha256"] is None


def test_excluding_all_rows_fails_before_vectorizer_fit(build_index, monkeypatch):
    def unexpected_fit(*args, **kwargs):
        pytest.fail("Vectorizers must not fit a held-out-only corpus")

    monkeypatch.setattr(retrieval.TfidfVectorizer, "fit_transform", unexpected_fit)
    with pytest.raises(ValueError, match="No retrieval examples remain"):
        build_index([pair("1", "held", "music freezes")], [{
            "tweet_id": "2", "thread_id": "held", "text": "another message",
        }])


@pytest.fixture
def scored_index(build_index):
    return build_index([
        pair("1", "thread-a", "music playback app frozen"),
        pair("2", "thread-a", "music playback app frozen again"),
        pair("3", "thread-b", "music playback pauses"),
        pair("4", "thread-c", "music download unavailable offline"),
    ])


def test_retrieve_reports_weighted_word_char_scores_and_diverse_threads(scored_index):
    query = "music playback app frozen"
    results = retrieval.retrieve(scored_index, query, top_k=3)
    word = cosine_similarity(
        scored_index["vectorizer"].transform([query]), scored_index["matrix"],
    ).ravel()
    char = cosine_similarity(
        scored_index["char_vectorizer"].transform([query]), scored_index["char_matrix"],
    ).ravel()
    assert len(results) == 3
    assert results[0]["customer_tweet_id"] == "1"
    assert {row["thread_id"] for row in results} == {"thread-a", "thread-b", "thread-c"}
    assert [row["similarity"] for row in results] == sorted(
        [row["similarity"] for row in results], reverse=True,
    )
    for result in results:
        index = scored_index["pairs"].index[
            scored_index["pairs"]["customer_tweet_id"].eq(result["customer_tweet_id"])
        ][0]
        assert result["word_similarity"] == pytest.approx(word[index])
        assert result["char_similarity"] == pytest.approx(char[index])
        assert result["similarity"] == pytest.approx(0.65 * word[index] + 0.35 * char[index])
    assert retrieval.retrieve(scored_index, query, top_k=1) == results[:1]


def test_character_scores_recover_a_query_without_word_overlap(scored_index):
    results = retrieval.retrieve(scored_index, "playbak", top_k=3)
    assert results
    for result in results:
        assert result["word_similarity"] == 0.0
        assert result["char_similarity"] > 0.0
        assert result["similarity"] == pytest.approx(0.35 * result["char_similarity"])


@pytest.mark.parametrize("query", ["", " \n\t ", "@Alice <URL> 123456", "zzzzzzzz qqqqqqqq"])
def test_empty_identifier_only_and_unsupported_queries_return_no_evidence(scored_index, query):
    assert retrieval.retrieve(scored_index, query, top_k=3) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_nonpositive_top_k_returns_no_evidence(scored_index, top_k):
    assert retrieval.retrieve(scored_index, "music playback", top_k) == []


@pytest.mark.parametrize("old_bundle", [{}, {"version": 1}])
def test_old_index_rejected_by_loader_and_retrieve(tmp_path, old_bundle):
    path = tmp_path / "old.joblib"
    joblib.dump(old_bundle, path)
    with pytest.raises(ValueError, match="Old retrieval index.*rebuild"):
        retrieval.load_retrieval_index(path)
    with pytest.raises(ValueError, match="Old retrieval index.*rebuild"):
        retrieval.retrieve(old_bundle, "music playback", top_k=3)