# Decision log

1. **Selected SpotifyCares over AmazonHelp and AppleSupport.** Spotify has
   43,243 direct replies and 40,770 cleaned pairs, enough data for evaluation
   with a narrower support surface than Amazon.
2. **Used customer-to-next-brand-reply pairs.** They provide traceable evidence
   for drafting without pretending that every thread has a final resolution.
3. **Split by thread ID.** Tweet-level random splits can place near-duplicate
   turns from one conversation on both sides and inflate results.
4. **Kept punctuation and emojis.** They contain useful complaint and urgency
   signals in informal Twitter text.
5. **Redacted handles, URLs, emails, and long numbers.** Retrieval should not
   reproduce customer identifiers or stale links.
6. **Started with nine operational intents.** Each intent implies different
   guidance or escalation; `other_unclear` avoids forced classifications.
7. **Used word and character TF-IDF.** It is CPU-friendly, reproducible, and
   robust to misspellings, making it a stronger student baseline than a large
   opaque model.
8. **Kept weak labels separate.** Keyword labels make the demo runnable but are
   never treated as the hand-labelled golden set or headline evidence.
9. **Used logistic regression probabilities.** They support confidence and
   margin-based abstention and are easy to inspect.
10. **Used lexical retrieval over historical messages.** It is fast, traceable,
    and lets every draft expose supporting tweet IDs.
11. **Do not append retrieved text when escalating.** Low-confidence or risky
    cases should avoid propagating irrelevant or account-specific guidance.
12. **Always escalate security and billing.** Incorrect automation in these
    categories has higher cost than reduced coverage.
13. **Report macro F1 instead of only accuracy.** Intent imbalance would make
    accuracy overstate performance on minority issues.
14. **Use a risk-coverage trade-off.** The auto-handle threshold is chosen for
    safety rather than maximum classification accuracy.
15. **Use a local Ollama judge and human overlap.** This avoids paid cloud
    dependencies while measuring how much the automated rubric can be trusted.
