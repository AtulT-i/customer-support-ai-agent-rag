# Data guide

Download `twcs.csv` from Kaggle's
`thoughtvector/customer-support-on-twitter` dataset and place it in `raw/`.
Do not commit the raw dataset.

`labels/train.csv` is for development and model training. `labels/golden.csv`
must be a separately sampled, hand-labelled, frozen evaluation set. Never put a
thread ID in both files.

The reviewed training, golden and initial human-score templates currently have
no labelled rows; the annotation queue and provisional weak labels are populated.
Do not replace reviewed data with synthetic or model-generated labels and call
them a golden set. Use [the submission workflow](../SUBMISSION.md) and
[human review app](../review_app.py) to complete actual human review.

Recommended sampling columns:

- `example_id`
- `tweet_id`
- `thread_id`
- `text`
- `intent`
- `secondary_intent`
- `decision`
- `escalation_reason`
- `required_reply_points`
- `forbidden_claims`
- `annotator_confidence`
- `annotator`

