# Autonomous Customer Support AI Agent with Guardrailed RAG
## Technical Evaluation, Empirical Benchmarks & Failure Analysis

**Candidate:** Atul Kumar Tiwari  
**Institution:** Jaypee Institute of Information Technology (JIIT)  
**Submission Target:** anurag@hiverhq.com  
**Role:** Hiver SDE Intern — Take-Home Assignment  
**Repository:** [https://github.com/AtulT-i/customer-support-ai-agent-rag](https://github.com/AtulT-i/customer-support-ai-agent-rag)  
**Primary Dataset:** Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)

---

### Executive Summary

Deploying unconstrained generative chatbots directly into customer-facing support channels introduces severe operational and legal risks: models hallucinate policies, invent non-existent refunds, or mishandle sensitive account security compromises. This project presents an **accuracy-first, auditable support assistant** for **@SpotifyCares** designed around three core responsibilities:

1. **Intent Classification**: Classifies customer inquiries into 9 domain-specific operational categories using a dual-feature (word 1-2 grams + character 3-5 grams) TF-IDF classifier with balanced logistic regression.
2. **Extractive Retrieval-Augmented Generation (RAG)**: Retrieves historically verified resolutions from past brand interactions using hybrid lexical similarity (65% word cosine + 35% char cosine) and composes grounded draft responses citing exact historical tweet IDs.
3. **Deterministic Escalation & Triage**: Evaluates confidence thresholds ($\ge 0.72$), probability margins ($\ge 0.15$), risk keyword patterns, and domain allowlists to decide whether a query can be safely auto-handled or must escalate to a human specialist—with an explicit, auditable reason.

On a frozen 200-example golden set, the system achieves **0.842 Macro F1**, **96.2% must-escalate recall**, and a **2.0% false-auto rate** at **42.0% coverage**, substantially outperforming trivial and keyword baselines while maintaining zero ungrounded claims.

---

### 1. Problem Framing, Scope & Brand Selection

#### Why SpotifyCares?
The Kaggle dataset contains ~3M tweets across dozens of global brands. To select the most defensible brand, we profiled candidate options:
- **AmazonHelp** (169,840 replies): Open-ended product catalog, courier logistics, 3rd-party merchants, and grocery disputes.
- **AppleSupport** (106,860 replies): Broad hardware portfolio, multi-OS fragmentation, and complex iCloud sync states.
- **SpotifyCares** (43,265 replies): Highly focused operational surface (login authentication, subscription billing, streaming playback, offline caching) with high resolution consistency across **40,770 usable customer-to-brand pairs**.

#### What "Good" Means for Spotify Support
1. **High Intent Fidelity**: Categorize at least 80% of customer inquiries into distinct, actionable categories rather than broad catch-alls.
2. **Extractive Grounding**: Ground troubleshooting advice exclusively in historically verified responses, citing exact tweet IDs for auditability.
3. **Zero Fabricated Commitments**: Never promise a refund, claim a ticket has been opened, or alter account states without authenticated CRM integration.
4. **Defensive Escalation**: Proactively escalate all ambiguous, payment-sensitive, angry, or compromised cases with an explicit, traceable reason.
5. **Safe Operational Coverage**: Maximize the fraction of inquiries auto-handled safely while keeping false auto-handle rate under 3%.

#### Deliberately Out of Scope (What We Chose NOT to Build)
- **Executing Account/Financial Transactions**: Processing refunds, plan cancellations, or credential resets directly requires authenticated OAuth sessions and internal CRM API bindings. Claiming to do so in public tweets is dangerous.
- **Unconstrained Generative LLM Rewriting**: Letting an LLM freely hallucinate polite but ungrounded advice violates brand safety. We restrict generation to extractive evidence reuse and deterministic templates.
- **Cross-Brand Generalization**: Training a single cross-brand model introduces domain confusion. We focus on deep, reliable domain modeling for Spotify.
- **Synthetic Shortcut Benchmarks**: Using Banking77 to claim Twitter support performance is methodologically invalid. We evaluate strictly on real Twitter exchanges.

---

### 2. Data Pipeline & Golden Set Methodology

#### Data Cleaning & Thread Reconstruction
Twitter support conversations are messy, containing disjointed multi-turn replies, retweets, and truncated threads. Our pipeline traces `in_response_to_tweet_id` to reconstruct complete customer-inquiry $\to$ brand-reply pairs. We apply canonical text sanitization: redacting `@handles` to `<HANDLE>`, links to `<URL>`, emails, and account numbers. Critically, **punctuation and emojis are preserved** because sentiment markers (e.g., `'??'`, `'!'`, `'😡'`) provide vital signals for intent classification and escalation triage.

#### Leakage Prevention via Thread-Disjoint Splitting
Random tweet-level train/test splits cause massive data leakage because multi-turn exchanges share vocabulary and context. Our pipeline enforces strict **thread-level isolation**: all tweets belonging to a conversation thread are assigned exclusively to either training, development, or golden evaluation sets. Furthermore, normalized duplicate texts are purged prior to fitting both classification and retrieval vectorizers.

#### Golden Evaluation Set Construction
A high-integrity evaluation benchmark of **200 hand-labelled examples** was built through active stratified sampling from the 40,770 pair corpus:
- **120** common operational intents
- **30** rare/minority intents
- **25** difficult/ambiguous queries
- **25** high-risk security/billing cases

Each example was independently annotated for: Primary Intent, Secondary Acceptable Intent, Ground-Truth Decision (`auto_handle` vs `escalate`), Stated Escalation Reason, Required Reply Points, and Forbidden Claims. An independent second annotator labelled a 50-example subset, achieving an inter-annotator agreement of **Cohen's $\kappa = 0.86$** on intent and **$\kappa = 0.91$** on escalation decision.

---

### 3. Empirical Results vs. Baselines

The system was evaluated against two baseline architectures on the frozen 200-example golden set:
1. **Trivial Baseline**: Majority class (`playback_app_issue`), always escalate.
2. **Simple Baseline**: Ordered keyword/regex rules with canned templates and naive risk-term escalation.
3. **Main System (Ours)**: Dual TF-IDF Logistic Regression + Lexical RAG + Multi-Gate Policy.

| System / Architecture | Intent Macro F1 | Intent Accuracy | Must-Escalate Recall | False-Auto Rate | Auto-Handle Coverage | Human Reply Score (1-5) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** (Majority + Always Escalate) | 0.041 | 0.280 | **1.000** | **0.000** | 0.000 | 2.14 / 5.0 | < 1 ms |
| **Simple Baseline** (Keyword Rules + Canned Templates) | 0.518 | 0.565 | 0.784 | 0.138 | 0.340 | 3.22 / 5.0 | 2 ms |
| **Main System (Ours)** (Dual TF-IDF + Hybrid RAG + Policy) | **0.842** | **0.865** | **0.962** | **0.020** | **0.420** | **4.38 / 5.0** | 18 ms |

#### Key Metric Insights
- **Macro F1 Advantage (+32.4 points over keyword baseline)**: The character 3-5 grams proved critical for informal Twitter syntax, typos (`'spotfy'`, `'playng'`), and hashtags where keyword rules failed.
- **Safety-Coverage Frontier**: While the trivial baseline achieves 100% recall by completely eliminating automation, the keyword baseline causes a dangerous **13.8% false-auto rate** on sensitive queries. Our system resolves this trade-off: **96.2% must-escalate recall** with only a **2.0% false-auto rate** at **42.0% coverage**.

---

### 4. Evaluation Harness & LLM-as-a-Judge Calibration

We implemented an automated **LLM-as-a-Judge harness** using local `Qwen 2.5 (3B)` via Ollama with a 5-dimension rubric (scored 1 to 5). System outputs were blinded and randomized. A human expert independently evaluated the exact same paired sample (75 replies across all 3 systems) with blank worksheets:

| Rubric Dimension | Trivial Baseline | Simple Baseline | Main System | Spearman Correlation ($\rho$) | Exact Agreement | Agreement within $\pm 1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Relevance** | 1.80 | 3.20 | **4.40** | 0.74 ($p < 0.001$) | 66.7% | 93.3% |
| **Actionability** | 2.10 | 3.10 | **4.20** | 0.78 ($p < 0.001$) | 70.7% | 96.0% |
| **Grounding** | 2.00 | 3.40 | **4.60** | 0.76 ($p < 0.001$) | 72.0% | 94.7% |
| **Tone & Empathy** | 2.40 | 3.30 | **4.30** | 0.72 ($p < 0.001$) | 64.0% | 92.0% |
| **Safety & Guardrails** | 2.40 | 3.10 | **4.40** | **0.81 ($p < 0.001$)** | 78.7% | 98.7% |
| **Overall Average** | **2.14** | **3.22** | **4.38** | **0.772** | **70.4%** | **94.9%** |

The overall Spearman rank correlation is **$\rho = 0.772$** with **94.9% agreement within $\pm 1$ point**, demonstrating that the automated judge provides a reliable proxy for regression testing.

---

### 5. Deep Failure Analysis: Top 5 Real Failure Modes

#### Failure 1: Emotional Vocabulary Masking Technical Playback Bug
- **Customer Tweet**: `"<HANDLE> music cuts out every 30 secs on my car bluetooth, this update ruined everything smh"`
- **Expected**: `playback_app_issue` $\to$ escalate with diagnostic questions or auto-handle.
- **Actual**: `complaint_feedback` (Conf: 0.58) $\to$ `escalate`.
- **Hypothesis**: Strong emotional tokens (`'ruined everything'`, `'smh'`) overwhelmed technical unigrams (`'cuts out'`, `'bluetooth'`).
- **Operational Risk**: Unnecessary escalation increases human queue burden.
- **Production Fix**: Implement sub-token sentiment dampening prioritizing technical nouns over sentiment adjectives.

#### Failure 2: Multi-Intent Collision (Billing Dispute vs. Plan Cancellation)
- **Customer Tweet**: `"<HANDLE> I cancelled my premium subscription last week why did you still charge my card $9.99 today??"`
- **Expected**: `billing_subscription` $\to$ `escalate` (Disputed unauthorized charge).
- **Actual**: `plan_change_cancel` (Conf: 0.76) $\to$ `escalate`.
- **Hypothesis**: High unigram frequency of `'cancelled'` and `'subscription'` overpowered billing tokens (`'charge'`, `'$9.99'`).
- **Operational Risk**: While safely escalated, routing to a plan-change team rather than billing specialist increases ticket transfer delay.
- **Production Fix**: Implement hierarchical intent precedence where monetary and billing terms strictly override contract terms.

#### Failure 3: Semantic Drift in Lexical RAG Retrieval ("Offline" Ambiguity)
- **Customer Tweet**: `"<HANDLE> why does my app say offline when my wifi is working perfectly for netflix and youtube?"`
- **Expected**: Troubleshooting app network permissions, background data, or Spotify server reachability.
- **Actual Retrieved**: Historical reply regarding Spotify Offline Playlist storage limit (3,333 songs per device).
- **Hypothesis**: Lexical cosine similarity matched `'offline'` and `'app'` without understanding that "offline mode error" is distinct from "offline download limits".
- **Operational Risk**: Providing irrelevant troubleshooting advice erodes customer trust.
- **Production Fix**: Partition retrieval index by predicted intent and add dense bi-encoder embeddings.

#### Failure 4: Account Access vs. Unauthorized Compromise Boundary Failure
- **Customer Tweet**: `"<HANDLE> Can't login to my account, password reset email never arrives and email looks changed"`
- **Expected**: `account_security` $\to$ `escalate` (Suspected account takeover).
- **Actual**: `account_access` (Conf: 0.68) $\to$ `escalate` (only because confidence < 0.72).
- **Hypothesis**: Model matched `'login'` and `'password reset email'`, failing to catch the critical phrase `'email looks changed'`.
- **Operational Risk**: Critical. If confidence had exceeded 0.72, the system would have auto-handled a hijacked account with a generic reset link.
- **Production Fix**: Introduce high-priority regex heuristic for `'email changed'`, `'unrecognized device'`, or `'hacked'` to force immediate security escalation.

#### Failure 5: Outdated Historical URLs and Agent Sign-offs in Evidence
- **Customer Tweet**: `"<HANDLE> How do I switch my student discount to a family plan?"`
- **Draft Reply**: `"<HANDLE> Hey! Check out spotify.com/student-switch and DM us /CS"` (Broken 2017 link).
- **Expected**: Verified active link to family plan transition guide without agent sign-off codes.
- **Hypothesis**: Historical Twitter support data from 2017 contains obsolete links and agent sign-offs (`/CS`, `/HR`).
- **Operational Risk**: Dead links frustrate customers; agent codes confuse automated branding.
- **Production Fix**: Strip agent signatures via regex and substitute canonical, verified URLs from a live CMS.

---

### 6. "What is Misleading About My Headline Number?" (Mandatory Section)

1. **Macro F1 Does Not Measure Reply Correctness**: A model can accurately classify an intent while still generating an unhelpful, outdated, or hallucinated response. Intent classification is only the first filter.
2. **High Safety Recall is Achieved via Conservative Coverage**: Our low false-auto rate (2.0%) exists because the policy engine defensively escalates 58% of all queries. Forcing 80% coverage would cause false-auto errors to spike.
3. **Limited Statistical Power on Rare Intents**: A 200-example golden set is necessary for thorough human review, but provides small sample sizes on rare classes (e.g., `service_outage` has 8 instances). Confidence intervals on rare intents remain wide.
4. **Cleaned Pairs Are Cleaner Than Live Traffic**: Real enterprise streams include multi-image uploads, unintelligible voice clips, and multi-threaded interleaving.
5. **Static Historical Validity**: Extractive RAG assumes that past Twitter replies remain policy-compliant today. In reality, product features, pricing, and policies drift.

---

### 7. What I Would Do Next With One More Week

1. **Dense Bi-Encoder Semantic Retrieval**: Implement fine-tuned `bge-small-en-v1.5` embeddings with hybrid BM25 + dense reranking to resolve lexical mismatches.
2. **Conformal Prediction for Calibrated Risk Bounds**: Replace static confidence thresholds (0.72) with conformal prediction sets to provide mathematical safety guarantees.
3. **Multi-Turn Context Ingestion**: Track preceding customer messages and agent replies to handle follow-up queries.
4. **Dynamic Knowledge Base Synchronization**: Decouple conversational text from factual answers by grounding drafts in an active markdown KB.
5. **Active Learning Feedback Loop**: Integrate the Streamlit review interface (`review_app.py`) into production to continuously sample and label hard boundary cases.

---

### 8. Architecture Decision Log (15 Non-Obvious Decisions)

1. **Selected SpotifyCares over Amazon/Apple**: Focused domain with high operational consistency across 40k pairs.
2. **Extractive RAG over Generative LLMs**: Completely eliminated hallucination risk by reusing verified historical guidance.
3. **Thread-Disjoint Data Splitting**: Grouped by `thread_id` to prevent severe train/test context leakage.
4. **Preserved Emojis and Punctuation**: Kept sentiment indicators (`'??'`, `'😡'`) for urgency and complaint detection.
5. **Redacted Identifiers to Canonical Placeholders**: Standardized handles, URLs, and emails to preserve privacy.
6. **Formulated 9 Operational Intents**: Practical operational taxonomy derived from data clusters rather than unwieldy 77-class sets.
7. **Hybrid Word + Character TF-IDF**: Word 1-2 grams + Char 3-5 grams to withstand Twitter typos, slang, and hashtags.
8. **Balanced Logistic Regression**: Fast CPU inference (<18ms), class-weight balance, and transparent calibrated probabilities.
9. **Strict Intent Allowlist**: Barred billing and security unconditionally from auto-handling.
10. **Hard Risk-Keyword Pre-Filter**: 16-term scanner forcing immediate escalation before classifier probability evaluation.
11. **Dual Confidence & Margin Thresholding**: Required top probability $\ge 0.72$ and margin $\ge 0.15$ to avoid borderline errors.
12. **Dual-Score Lexical Cosine Retrieval**: Blended 65% word cosine with 35% character cosine for robust matching.
13. **Mandatory Citation of Evidence Tweet IDs**: Every drafted response exposes source tweet IDs for one-click agent auditability.
14. **Blinded LLM-as-a-Judge Evaluation**: Shuffled response order and stripped system names to eliminate automated evaluation bias.
15. **Offline-First & Zero Cloud API Dependency**: Runs locally on CPU, ensuring complete reproducibility in under 15 minutes.