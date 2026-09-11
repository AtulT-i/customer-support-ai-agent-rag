# Report outline (maximum 6 pages)

See [the substantive report draft](submission-report.md) and
[sampling/annotation protocol](sampling-and-labeling.md). Empirical results and
five real reviewed failures must be added only after actual evaluation.

## 1. Problem framing

- Brand and evidence for choosing it
- Definition of a useful and safe response
- Scope and non-goals

## 2. Data and labels

- Thread reconstruction and cleaning
- Intent taxonomy derived from data
- Train/development/golden split by thread
- Golden-set sampling and annotation agreement

## 3. Systems

- Trivial baseline
- Keyword/template baseline
- Main classifier, retrieval, drafting, and escalation policy

## 4. Results

- Macro F1 and per-intent results
- Escalation risk/coverage results
- Reply rubric scores
- LLM judge versus human agreement

## 5. Failure analysis

Include five real examples. For each, provide the input, expected behavior,
actual behavior, likely cause, risk, and proposed fix.

## 6. What is misleading about my headline number?

Explain sampling, annotation, filtering, judge, generalization, and operational
limitations.

## 7. One more week

Prioritize improvements by expected risk reduction, not novelty.

## Decision log

Keep 10-15 dated bullets with the decision, alternatives, evidence, and reason.

