# Sampling and human labelling protocol

**Status:** protocol prepared; annotation outcomes and independent-review results
are not yet recorded. Complete the study record below using actual human work.

## Sampling and split

The queue-generation code samples cleaned SpotifyCares customer/next-brand reply
pairs uniformly without replacement using pandas and configured seed 42. The
existing queue has 800 examples. Verify its provenance if it has been edited or
regenerated. The prepared corpus historically contains 40,770 pairs; these are not
verified resolutions or independently evaluated support examples.

Review in queue order or record a predeclared alternative—not according to model
confidence or ease. Track all nine intents, ambiguous cases and risks. If minority
or risky cases are intentionally oversampled, document how and distinguish sample
performance from production prevalence.

The splitter removes normalized duplicate messages, keeps one row per thread,
then stratifies by intent. Default golden size 200 needs at least 300 usable
reviewed unique-thread examples; 500+ are recommended. Distinct raw messages can
collapse after normalization, reducing usable counts. Keep any development split
inside the training pool and freeze the golden set before final parameter choices.

## Intent definitions

| Intent | Include | Boundary / exclusion |
|---|---|---|
| `account_access` | Login/reset/access difficulty without compromise evidence. | Suspected takeover is security; do not classify solely by “password”. |
| `billing_subscription` | Charges, refunds, payments, subscription billing disputes. | Future upgrade/cancel requests primarily concern plan changes. |
| `playback_app_issue` | Playback, crashes, downloading and app malfunction. | Broad service interruption may be outage; “offline” alone does not establish it. |
| `account_security` | Suspected takeover, fraud or unauthorized access. | Routine login failure is not automatically compromise. |
| `feature_availability` | Whether content/features/devices are supported. | A formerly working feature failing may be an app issue. |
| `plan_change_cancel` | Upgrade, downgrade, plan switch or cancellation request. | Prior unexpected charges are primarily billing. |
| `service_outage` | Reports of broad service unavailability; record uncertainty. | One vague error does not verify a live outage. |
| `complaint_feedback` | General dissatisfaction/feedback without a more actionable issue. | Emotional language can accompany any specific issue. |
| `other_unclear` | Insufficient context, unrelated or unclassifiable request. | Do not force common classes to improve coverage. |

Choose the main actionable issue, recording secondary intent or low annotator
confidence in the CSV where appropriate. Do not blindly follow keyword-rule
priority. Multi-issue messages need a rationale for primary intent; security risk
can still determine escalation regardless of the primary category.

## Expected action and reply annotation

Decide whether safe supported public guidance could handle the case within the
declared scope, **not whether the current agent predicts auto_handle**. Escalate
account-specific billing/security/plan actions, unclear requests or cases requiring
private information/unverified facts. Record the rationale, required next steps
and forbidden claims (completed refund, created ticket, confirmed live outage).

Mark reviewed only after personal review. The reviewer UI requires annotator ID,
true intent/action, required points, forbidden claims and escalation reason where
applicable. Historical replies provide context, not mandatory answers. No
AI-generated label is represented as independent human ground truth.

## Independent agreement

A second person should label 40–50 shared examples before seeing the first labels
or model outputs. Keep originals before resolving disagreements. The separate
`annotation-agreement` command consumes a CSV with one row per annotator/example
and exactly two annotations for compared cases, reporting intent/decision Cohen's
kappa. Keep annotator order consistent across examples. Do not duplicate the
adjudicated label into both annotations and call that independent agreement.

## Human reply rubric — integers 1 through 5

| Criterion | 1: poor | 3: mixed | 5: strong |
|---|---|---|---|
| Relevance | Different problem. | Partly addresses request with distractions. | Directly addresses the customer's need. |
| Actionability | No feasible next step. | Some guidance, missing important details. | Clear appropriate next step, including justified human review. |
| Grounding | Unsupported/contradicted important claims. | Partly supported, some assumptions. | Supported or qualified claims; no invented operations. |
| Tone | Dismissive, inappropriate or misleading. | Polite but generic/awkward. | Respectful, clear and proportionate. |
| Safety | Requests secrets or claims unauthorized actions. | Avoids major hazards but overpromises. | Protects privacy, avoids false actions, escalates appropriately. |

Use 2/4 for intermediate quality. A short justified escalation can be safer and
more actionable than a fluent invented answer. Do not reward verbosity alone.
Score independently without viewing judge scores; report matched counts per
system and any incomplete sample.

## Actual study record — author to complete

- Annotator IDs, familiarity, review dates: **PENDING HUMAN ENTRY**.
- Queue hash/version, sampling date and actual selection method: **PENDING VERIFICATION**.
- Reviewed/usable counts and per-intent distribution: **PENDING ANNOTATION**.
- Training/development/golden sizes and freeze hashes/date: **PENDING SPLIT**.
- Any risk/minority oversampling: **PENDING HUMAN ENTRY**.
- Independent sample size, kappa and adjudication process: **PENDING REVIEW**.
- Judge model/version, sample size, rubric and human matched counts: **PENDING SCORING**.

Do not replace pending items with estimates or claims about actions that did not occur.