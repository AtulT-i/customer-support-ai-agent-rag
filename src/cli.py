from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.agent import SupportAgent, require_artifacts
from src.config import load_config
from src.data import make_label_queue, prepare_pairs, profile_brands
from src.evaluation import annotation_agreement, evaluate_all
from src.judge import judge_human_agreement, judge_predictions
from src.labeling import create_weak_labels, split_reviewed_labels
from src.model import train_intent_model
from src.retrieval import build_retrieval_index
from src.retrieval import exclude_held_out
from src.human_scoring import prepare_human_scores
from src.submission import submission_status, export_failure_queue


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Twitter support agent")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("profile", help="compare candidate brands")
    commands.add_parser("prepare", help="create cleaned customer/reply pairs")
    status = commands.add_parser("submission-status", help="show human annotation progress and submission blockers")
    status.add_argument("--golden-size", type=int, default=200)
    commands.add_parser("prepare-human-scores", help="prepare blank, content-bound human worksheet from current paired judge sample")
    commands.add_parser("failure-queue", help="export measured main-system errors for human failure analysis")

    label_queue = commands.add_parser(
        "make-label-queue", help="sample messages for manual labelling"
    )
    label_queue.add_argument("--size", type=int, default=800)

    bootstrap = commands.add_parser(
        "bootstrap", help="build provisional weak-label demo artifacts"
    )
    bootstrap.add_argument("--size", type=int, default=12_000)

    split = commands.add_parser(
        "split-labels",
        help="split reviewed queue into group-safe train and golden sets",
    )
    split.add_argument("--size", type=int, default=200)

    train = commands.add_parser("train", help="train from reviewed labels")
    train.add_argument(
        "--labels", type=Path, default=Path("data/labels/train.csv")
    )

    predict = commands.add_parser("predict", help="run one agent prediction")
    predict.add_argument("text")
    predict.add_argument("--ollama-model", default=None, help="optional local evidence selector, e.g. qwen2.5:3b")

    evaluate = commands.add_parser("evaluate", help="evaluate all systems on golden set")
    evaluate.add_argument("--ollama-model", default=None)

    agreement = commands.add_parser(
        "annotation-agreement", help="measure agreement between two annotators"
    )
    agreement.add_argument(
        "--labels", type=Path, default=Path("data/labels/double_labels.csv")
    )

    judge = commands.add_parser("judge", help="score replies using local Ollama")
    judge.add_argument("--model", default="qwen2.5:3b")
    judge.add_argument("--size", type=int, default=50)

    judge_agreement = commands.add_parser(
        "judge-agreement", help="compare judge and human reply scores"
    )
    judge_agreement.add_argument(
        "--human", type=Path, default=Path("data/labels/human_reply_scores.csv")
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_config()

    if args.command == "submission-status":
        print(json.dumps(submission_status(config, args.golden_size), indent=2))
        return

    if args.command == "prepare-human-scores":
        directory = config.data["model_path"].parent / "evaluation"
        result = prepare_human_scores(directory / "predictions.csv", config.data["golden_csv"],
                                      directory / "judge_scores_current.csv", config.data["human_scores_csv"])
        print(f"Prepared {len(result)} human-scoring rows; no judge scores copied. Existing work preserved.")
        return

    if args.command == "failure-queue":
        directory = config.data["model_path"].parent / "evaluation"
        result = export_failure_queue(directory / "predictions.csv", config.data["golden_csv"],
                                      config.data["pairs_csv"].parents[1] / "labels" / "failure_review.csv")
        print(f"Exported {len(result)} measured intent/decision errors. Review causes and select five real failures; never invent missing failures.")
        return

    if args.command == "profile":
        result = profile_brands(
            config.data["raw_csv"], config.candidate_brands
        )
        output = config.data["pairs_csv"].parent / "brand_profile.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(result.to_string(index=False))
        return

    if args.command == "prepare":
        pairs = prepare_pairs(
            config.data["raw_csv"],
            config.data["pairs_csv"],
            config.brand,
        )
        print(f"Wrote {len(pairs):,} cleaned pairs to {config.data['pairs_csv']}")
        return

    if args.command == "make-label-queue":
        if config.data["label_queue_csv"].exists() and not pd.read_csv(config.data["label_queue_csv"]).empty:
            raise ValueError("Label queue already contains rows. Archive it explicitly before generating a new sample.")
        queue = make_label_queue(
            config.data["pairs_csv"],
            config.data["label_queue_csv"],
            args.size,
            config.random_seed,
        )
        print(f"Wrote {len(queue):,} rows to {config.data['label_queue_csv']}")
        return

    if args.command == "bootstrap":
        pairs = pd.read_csv(config.data["pairs_csv"], dtype=str, keep_default_na=False)
        golden = pd.read_csv(config.data["golden_csv"], dtype=str, keep_default_na=False)
        pairs = exclude_held_out(pairs, golden)
        weak = create_weak_labels(pairs, args.size, config.random_seed)
        weak.to_csv(config.data["weak_labels_csv"], index=False)
        train_intent_model(
            config.data["weak_labels_csv"],
            config.data["model_path"],
            config.random_seed,
            allow_weak_labels=True,
        )
        build_retrieval_index(
            config.data["pairs_csv"],
            config.data["retrieval_path"],
            config.retrieval["max_corpus_size"],
            config.retrieval["max_features"],
            config.random_seed,
            golden_csv=config.data["golden_csv"],
        )
        print(
            "Built provisional weak-label artifacts. Do not report these as "
            "human-labelled headline results."
        )
        return

    if args.command == "split-labels":
        for key in ("train_labels_csv", "golden_csv"):
            if config.data[key].exists() and not pd.read_csv(config.data[key]).empty:
                raise ValueError("Training/golden split already contains rows. Keep the frozen split, or explicitly archive it before creating a new experiment.")
        train, golden = split_reviewed_labels(
            str(config.data["label_queue_csv"]),
            str(config.data["train_labels_csv"]),
            str(config.data["golden_csv"]),
            args.size,
            config.random_seed,
        )
        print(
            f"Wrote {len(train):,} training and {len(golden):,} frozen golden "
            "examples with no thread overlap."
        )
        return

    if args.command == "train":
        # Validate frozen split BEFORE writing any new model artifacts.
        from src.evaluation import _validate_golden
        from src.text import normalize_for_matching

        training = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
        golden = pd.read_csv(config.data["golden_csv"], dtype=str, keep_default_na=False)
        _validate_golden(golden, config.intents)
        if not {"thread_id", "text", "intent"}.issubset(training.columns):
            raise ValueError("Training labels require thread_id, text and intent.")
        if not training["intent"].isin(config.intents).all():
            raise ValueError("Training contains unknown intents.")
        if set(training["thread_id"]) & set(golden["thread_id"]):
            raise ValueError("Training threads overlap the frozen golden set.")
        if set(training["text"].map(normalize_for_matching)) & set(golden["text"].map(normalize_for_matching)):
            raise ValueError("Training text overlaps the frozen golden set.")
        train_intent_model(
            args.labels,
            config.data["model_path"],
            config.random_seed,
        )
        build_retrieval_index(
            config.data["pairs_csv"],
            config.data["retrieval_path"],
            config.retrieval["max_corpus_size"],
            config.retrieval["max_features"],
            config.random_seed,
            golden_csv=config.data["golden_csv"],
        )
        print("Built reviewed-label model and retrieval artifacts.")
        return

    if args.command == "predict":
        require_artifacts(config)
        print(
            json.dumps(
                SupportAgent(config, ollama_model=args.ollama_model).respond(args.text),
                indent=2,
                ensure_ascii=True,
            )
        )
        return

    if args.command == "evaluate":
        require_artifacts(config)
        results = evaluate_all(
            config.data["golden_csv"],
            config.data["train_labels_csv"],
            SupportAgent(config, ollama_model=args.ollama_model),
            config.data["model_path"].parent / "evaluation",
        )
        print(results.to_string(index=False))
        return

    if args.command == "annotation-agreement":
        print(json.dumps(annotation_agreement(args.labels), indent=2))
        return

    if args.command == "judge":
        result = judge_predictions(
            config.data["model_path"].parent / "evaluation" / "predictions.csv",
            config.data["golden_csv"],
            config.data["model_path"].parent / "evaluation" / "judge_scores.csv",
            args.model,
            args.size,
            config.random_seed,
        )
        # Keep a single paired run separate from the multi-version historical cache.
        result.to_csv(config.data["model_path"].parent / "evaluation" / "judge_scores_current.csv", index=False)
        print(result.groupby("system").mean(numeric_only=True).to_string())
        return

    if args.command == "judge-agreement":
        result = judge_human_agreement(
            config.data["model_path"].parent
            / "evaluation"
            / "judge_scores_current.csv",
            args.human,
        )
        output = (
            config.data["model_path"].parent
            / "evaluation"
            / "judge_human_agreement.csv"
        )
        result.to_csv(output, index=False)
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
