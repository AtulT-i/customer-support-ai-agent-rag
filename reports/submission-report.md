# Spotify support assistant — submission report draft

**Status: incomplete empirical study, not ready for final submission.** Methods
below reflect implemented code. Measurements, independent human agreement and five
reviewed failures remain pending. Check the rendered report against the original
maximum six-page requirement after completing the evidence.

## 1. Problem, scope and brand choice

This project studies whether historical support exchanges can support useful
first-response drafts while abstaining from risky automation. SpotifyCares offers
a narrower account/playback/subscription surface than a general retailer and
sufficient data for experimentation. Preparation historically produced 40,770
customer-to-next-brand-reply pairs, not verified resolved cases.

The assistant predicts intent, retrieves historical examples and recommends
automatic handling or human review with a draft and source IDs. It never sends
replies, opens tickets, accesses accounts or performs refunds. Escalation is a
recommendation, not proof a human has been contacted.

## 2. Data, annotation and evaluation design

Preparation selects brand replies, joins parent customer tweets, redacts variable
identifiers, removes short/duplicate/retweet text and traces thread roots. Language
detection and semantic near-duplicate detection are not implemented. Historical
replies may be incomplete or stale and must not be treated as unquestionable truth.

The nine intents cover access, billing, playback, security, features, plan changes,
outages, complaints and unclear messages. The 800-row annotation queue supports
true intent, expected action, required points and forbidden claims. See the
[annotation protocol](sampling-and-labeling.md) for class definitions, independent
review and the pending study record.

The intended 200-row golden set requires at least 300 usable reviewed unique-thread
examples including training; 500+ are recommended. The splitter removes normalized
duplicates and keeps one row per thread before intent stratification. Development
tuning must use a subset of the training pool, never golden labels. Actual review
counts, distribution, freeze date and independent annotation agreement are **pending**.

Golden threads, tweet IDs and normalized duplicate messages are excluded before
retrieval vectorizers are fitted. Evaluation checks overlap and training/golden
hashes, rejecting provisional or stale artifacts. Keyword-derived demo labels are
not accepted as final human-reviewed evaluation evidence.

## 3. Systems and design choices

**Trivial baseline:** majority intent and always escalate, testing whether apparent
safety recall comes entirely from eliminating automation coverage.

**Keyword baseline:** ordered intent rules and simple responses, with sensitive
and unclear cases escalated. It is deliberately simpler than the main policy;
generic human-review promises should be judged as written, not silently fixed
after seeing the evaluation.

**Main system:** word unigrams/bigrams and character 3–5 grams feed balanced
logistic regression. A separate local index combines word cosine similarity (65%)
and character cosine similarity (35%), returning up to three distinct threads.
Templates and filtered historical suggestions produce cited drafts. The system
uses sparse lexical vectors, not dense transformer embeddings.

Policy checks risk terms, allowed intents, confidence (0.72), top-two margin
(0.15), retrieval similarity (0.35) and safe reusable evidence. These are defaults,
not calibrated or experimentally optimal thresholds. With normalized probabilities,
the margin gate is redundant at the current confidence setting; future tuning
should account for interactions rather than treating gates as independent.

Optional local Ollama selection can choose filtered evidence or abstain, but cannot
write arbitrary customer claims. Invalid output falls back to deterministic
selection. Record whether it was enabled and the model identity for each run.
See the [decision log](decision-log.md) for the original engineering rationale.

## 4. Results and judge validation — pending

**No verified accuracy, macro F1 or automation coverage is available yet.** Earlier
regression tests and the working UI demonstrate code behavior, not model quality.
Record the latest test count separately from empirical evaluation.

| System | Macro F1 | Must-escalate recall | False auto / must-escalate | Auto coverage | Human reply scores |
|---|---|---|---|---|---|
| Majority | Pending | Pending | Pending | Pending | Pending |
| Keyword | Pending | Pending | Pending | Pending | Pending |
| Main | Pending | Pending | Pending | Pending | Pending |

Fill these cells only from evaluated outputs. Also report actual unsafe counts
and the unsafe fraction among automatic cases, which has a different denominator.
Inspect per-intent metrics and confusion matrices, especially minority and risky
categories. Human top-1/top-3 relevance assessment is needed; similarity is not a
measured retrieval relevance score.

The reply rubric scores relevance, actionability, grounding, tone and safety
from 1–5. Shared judge samples, randomized order and content-keyed caches improve
comparability. Human reviewers receive blank score fields and matching drafts/
evidence without judge scores. Report judge identity, sample sizes, human matched
counts by system, Spearman, exact agreement and within-one agreement. Constant
score correlations must be reported as undefined. **These results remain pending.**

## 5. Five real failure examples — pending

The `failure-queue` command exports actual evaluated main-system intent/decision
errors with unsafe auto-handles first. Review those rows and independently scored
drafts, then select five representative real cases. Each needs input/ID, expected
outcome, actual draft/decision, likely cause, operational risk and proposed fix.
Treat causal explanations as hypotheses unless validated.

**Five reviewed failures are not yet available; none are invented here.** Areas
to investigate—not observed golden findings—include weak-label ambiguity, lexical
paraphrase misses, safe guidance below rank three, stale historical advice and
overly restrictive evidence filters. Correct security escalations are not failures
merely because they reduce automation. Disclose fewer observed errors rather than
fabricating them; any expanded challenge set must be reported separately.

## 6. What is misleading about my headline number?

Intent classification alone does not establish safe support automation. A small
golden sample provides limited rare-class evidence; the project author may also
design the taxonomy and annotate data. Independent labels reduce, not eliminate,
subjectivity. Keyword demo labels can reproduce rules without generalizing.

One brand and historical Twitter period do not represent current support traffic.
Pair filtering alters the population. Thread separation cannot remove every semantic
duplicate, and historical replies do not guarantee complete resolution or current
policy. Neither cosine similarity nor classifier confidence is a calibrated
probability that a reply will be safe and correct.

Conservative escalation can maximize safety recall while minimizing usefulness.
Earlier, 531 of 30,000 indexed replies had safe extracts; that corpus percentage
is **not auto-handle coverage**. Six earlier smoke messages all escalated. Macro
F1 does not measure reply truthfulness or operational risk, so report both safety
and coverage rather than selecting the most flattering number.

LLM judges can share biases and disagree with people; caching improves repeatability,
not validity. File hashes bind data/results, not annotation honesty or the entire
experiment. Report uncertainty, matched sample counts and all material limitations.

## 7. One more week and reproducibility

First complete independent labels, freeze splits and inspect unsafe errors. Then
human-score retrieval/replies, establish development calibration and risk-coverage
analysis, and compare feature/retrieval ablations and larger candidate pools.
Only after that evaluate semantic retrieval or local selection for measurable
quality at acceptable latency/risk. A verified current knowledge base matters
more than unvalidated generative fluency.

Follow [the submission checklist](../SUBMISSION.md) and [run guide](../RUN_LOCAL.md).
Record code/config version, dependencies, hashes, sample dates and LLM identities.
Review privacy and dataset rights before publishing anything. Archive approved
evaluation results and check final rendered page count before submission.