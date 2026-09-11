import pandas as pd

from src.labeling import split_reviewed_labels


def test_split_has_no_thread_leakage(tmp_path) -> None:
    rows = []
    intents = ("account_access", "playback_app_issue")
    for index in range(300):
        rows.append(
            {
                "example_id": f"e-{index}",
                "text": f"message {index}",
                "intent": intents[index % 2],
                "decision": "auto_handle",
                "thread_id": f"thread-{index}",
                "reviewed": "yes",
            }
        )
    source = tmp_path / "reviewed.csv"
    train_path = tmp_path / "train.csv"
    golden_path = tmp_path / "golden.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    train, golden = split_reviewed_labels(
        str(source), str(train_path), str(golden_path), 150, 42
    )

    assert len(golden) == 150
    assert set(train["thread_id"]).isdisjoint(golden["thread_id"])
