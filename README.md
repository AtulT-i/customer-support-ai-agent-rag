# Spotify Support Intelligence — Guardrailed RAG AI Agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Architecture-Extractive%20RAG-green.svg)](#system-architecture)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

A production-grade, local-first customer support triage, retrieval, and drafting system with strict safety guardrails and deterministic escalation policies. Built on real-world customer support interactions from Twitter (`@SpotifyCares`), this agent combines machine learning intent classification with verified historical retrieval to draft accurate, grounded responses without hallucination risk.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [System Architecture & Flow](#system-architecture--flow)
- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [How to Run This Project](#how-to-run-this-project)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick CLI Prediction](#quick-cli-prediction)
  - [Interactive Web UI](#interactive-web-ui)
  - [Human Annotation & Quality Review Workspace](#human-annotation--quality-review-workspace)
  - [Running Unit & Regression Tests](#running-unit--regression-tests)
  - [End-to-End Pipeline Commands](#end-to-end-pipeline-commands)
  - [Optional Local LLM (Ollama) Integration](#optional-local-llm-ollama-integration)
- [Operational Intents](#operational-intents)
- [Safety, Escalation & Guardrail Policies](#safety-escalation--guardrail-policies)
- [Evaluation Harness & Metrics](#evaluation-harness--metrics)
- [Tech Stack](#tech-stack)

---

## What This Project Does

In high-volume customer support operations, deploying an unconstrained generative chatbot carries severe operational, legal, and reputational risks: models can hallucinate policies, invent non-existent refunds, or mishandle sensitive account security compromises.

This project delivers an **accuracy-first, auditable support assistant** designed around three core responsibilities:

1. **Intent Classification**: Classifies incoming customer inquiries into 9 domain-specific operational categories derived from historical support patterns using a dual-feature TF-IDF classifier.
2. **Extractive Retrieval-Augmented Generation (RAG)**: Retrieves historically verified resolutions from past brand interactions using hybrid lexical similarity (word + character n-grams) and composes grounded draft responses with exact citation tweet IDs.
3. **Deterministic Escalation & Triage**: Evaluates confidence thresholds, probability margins, risk keyword patterns, and domain allowlists to determine whether a query can be safely auto-handled or must be escalated to a human specialist—with an explicit, auditable reason.

```json
{
  "intent": "playback_app_issue",
  "confidence": 0.89,
  "decision": "auto_handle",
  "reason": "low-risk intent with high classifier and retrieval confidence",
  "draft_reply": "Hi! Can you try reinstalling the app? You can follow the steps in our guide. Let us know how it plays out!",
  "evidence_tweet_ids": ["104107", "104105"],
  "rag_mode": "extractive_default"
}
```

---

## System Architecture & Flow

The system processes incoming customer messages through an auditable, multi-stage pipeline:

```text
Incoming Customer Message
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Text Normalization & Sanitization                        │
│    • Redact @handles, URLs, emails, account numbers         │
│    • Preserve emojis and punctuation signals                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Intent Classification                                    │
│    • Dual TF-IDF (word 1-2 grams + character 3-5 grams)     │
│    • Balanced Multinomial Logistic Regression               │
│    • Outputs: top intent, confidence, top-2 probability gap │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Escalation & Safety Policy Engine                        │
│    • Risk keyword heuristics (security, fraud, refunds)     │
│    • Domain allowlist: only low-risk intents auto-handled   │
│    • Confidence threshold (≥ 0.72) & Margin check (≥ 0.15)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      [Escalation Required]     [Eligible for Auto-Handle]
              │                         │
              │                         ▼
              │        ┌──────────────────────────────────────┐
              │        │ 4. Hybrid Lexical RAG Retrieval      │
              │        │    • 65% Word + 35% Char Cosine Sim  │
              │        │    • Filter past brand resolutions   │
              │        │    • Extract conservative advice     │
              │        └────────────────┬─────────────────────┘
              │                         │
              ▼                         ▼
┌───────────────────────────┐ ┌───────────────────────────────┐
│ 5a. Escalation Response   │ │ 5b. Grounded Draft Response   │
│ • Clear handover guidance │ │ • Intent-specific template    │
│ • Explicit stated reason  │ │ • Verified historical advice  │
│ • Zero unverified claims  │ │ • Cited evidence tweet IDs    │
└─────────────┬─────────────┘ └─────────┬─────────────────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
               Final Structured Payload
```

---

## Key Features

- **Extractive Grounding**: Rather than letting an LLM generate text unconstrained, the agent retrieves and reuses proven troubleshooting steps from historical support interactions, ensuring zero invented policies or fake operations.
- **Thread-Disjoint Splitting**: Preprocessing joins customer queries to direct brand replies and enforces splits strictly by conversation thread IDs to prevent data leakage between development, training, and evaluation sets.
- **Auditable Citations**: Every drafted response exposes the exact `evidence_tweet_ids` drawn from historical cases, enabling human agents to verify recommendations in one click.
- **Hard Guardrails on Sensitive Topics**: Inquiries involving billing disputes, refund demands, account takeovers, password resets, or legal threats are immediately escalated.
- **Offline-First & Low Latency**: Runs entirely locally on CPU using optimized sparse representations—no cloud GPUs, paid API subscriptions, or external vector databases required.
- **Optional Local LLM Selector**: Supports local inference engines (such as Ollama with Qwen 2.5) to select among pre-filtered safe candidates without permission to generate ungrounded claims.
- **Dual Streamlit Applications**:
  - `app.py`: Live interactive agent showcase for testing queries, viewing retrieval candidates, and inspecting policy decisions.
  - `review_app.py`: Dedicated annotation and quality evaluation workspace for human ground-truth labeling and 5-dimensional rubric scoring.

---

## Directory Structure

```text
.
├── app.py                     # Streamlit live support agent showcase
├── review_app.py              # Streamlit human review and annotation interface
├── requirements.txt           # Production Python dependencies
├── config/
│   └── project.yaml           # Centralized configuration (thresholds, intents, paths)
├── src/
│   ├── __init__.py
│   ├── agent.py               # Main SupportAgent pipeline orchestrator
│   ├── baselines.py           # Trivial (majority) and Simple (keyword) baselines
│   ├── cli.py                 # Unified CLI management interface
│   ├── config.py              # YAML configuration loader and validator
│   ├── data.py                # Dataset parsing, thread reconstruction, cleaning
│   ├── evaluation.py          # Metric calculation (macro F1, recall, false-auto)
│   ├── human_scoring.py       # Inter-annotator agreement & human rubric scoring
│   ├── judge.py               # LLM-as-a-judge 5-point evaluation harness
│   ├── labeling.py            # Active sampling and dataset queue generation
│   ├── local_rag.py           # Optional local LLM selector integration (Ollama)
│   ├── model.py               # TF-IDF vectorizers and Logistic Regression classifier
│   ├── policy.py              # Escalation gates, safety rules, confidence thresholds
│   ├── retrieval.py           # Hybrid word + character lexical RAG index
│   ├── schemas.py             # Typed dataclasses for prediction payloads
│   ├── submission.py          # Annotation validation and evaluation helpers
│   └── text.py                # Text normalization and entity placeholder redaction
├── data/
│   ├── raw/                   # Raw conversation data (twcs.csv)
│   ├── processed/             # Cleaned customer-brand pairs and brand profiles
│   └── labels/                # Training, golden evaluation set, and review queues
├── artifacts/                 # Serialized models, retrieval indexes, evaluation caches
├── docs/
│   ├── architecture.md        # Detailed system architecture and file-by-file guide
│   └── diagrams/              # High-level architecture diagrams (SVG)
├── reports/
│   ├── decision-log.md        # Architectural decision records (ADRs)
│   ├── sampling-and-labeling.md # Dataset sampling and human annotation protocol
│   └── submission-report.md   # System evaluation and methodology report
└── tests/                     # Comprehensive test suite (13 test modules)
```

---

## How to Run This Project

### Prerequisites

- Python 3.12 or 3.13 (64-bit recommended)
- Git

### Installation

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/AtulT-i/customer-support-ai-agent-rag.git
   cd customer-support-ai-agent-rag
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(On macOS/Linux: `source .venv/bin/activate`)*

3. **Install dependencies:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

### Quick CLI Prediction

Test any customer query directly from your terminal:

```powershell
python -m src.cli predict "The app keeps pausing whenever I play a song"
```

Output:
```json
{
  "intent": "playback_app_issue",
  "confidence": 0.88,
  "decision": "auto_handle",
  "reason": "low-risk intent with high classifier and retrieval confidence",
  "draft_reply": "Hi! Can you try reinstalling the app? You can follow the steps in our guide. Let us know how it plays out!",
  "evidence_tweet_ids": ["104107", "104105"],
  "rag_mode": "extractive_default"
}
```

Test an inquiry requiring human review (e.g. security or billing):

```powershell
python -m src.cli predict "Someone accessed my account without permission and changed my playlist"
```

Output:
```json
{
  "intent": "account_security",
  "confidence": 0.94,
  "decision": "escalate",
  "reason": "high-risk or ambiguous category requiring human review",
  "draft_reply": "Thanks for reaching out. We've flagged this for our security team to investigate immediately. Please do not share sensitive credentials publicly.",
  "evidence_tweet_ids": [],
  "rag_mode": "extractive_default"
}
```

---

### Interactive Web UI

Launch the interactive support agent dashboard:

```powershell
streamlit run app.py
```

Open **http://localhost:8501** in your browser to:
- Enter custom customer messages and receive real-time triage decisions.
- Inspect intent classification probabilities and top-2 margins.
- View top-3 retrieved historical cases with relevance similarity scores.
- Test edge cases (billing disputes, account takeovers, service outages).

---

### Human Annotation & Quality Review Workspace

Launch the dedicated reviewer interface:

```powershell
streamlit run review_app.py --server.port 8502
```

Open **http://localhost:8502** to:
- Review unlabelled customer queries and assign verified intents.
- Provide required response points and forbidden claims.
- Score drafted replies across the 5-point evaluation rubric.

---

### Running Unit & Regression Tests

Run the complete test suite to verify pipeline integrity:

```powershell
pytest -q
```

The test suite covers:
- Text sanitization and redaction idempotence
- Intent classification and calibration
- Thread reconstruction and data leakage prevention
- Lexical retrieval similarity scoring
- Escalation policy boundary conditions
- Safety heuristics and prompt-injection resistance

---

### End-to-End Pipeline Commands

The repository provides automated CLI commands for dataset lifecycle management:

```powershell
# 1. Profile brand distribution in raw dataset
python -m src.cli profile

# 2. Extract and clean brand-specific conversation pairs
python -m src.cli prepare

# 3. Create active sampling annotation queue
python -m src.cli make-label-queue --size 800

# 4. Bootstrap provisional demo models
python -m src.cli bootstrap --size 12000

# 5. Split reviewed labels into training and frozen evaluation sets
python -m src.cli split-labels --size 200

# 6. Train supervised model and build leakage-excluded retrieval index
python -m src.cli train

# 7. Run evaluation against baselines
python -m src.cli evaluate
```

---

### Optional Local LLM (Ollama) Integration

If you wish to benchmark the optional local LLM evidence selector:

1. Install and start [Ollama](https://ollama.com/).
2. Pull a lightweight model:
   ```powershell
   ollama pull qwen2.5:3b
   ```
3. Run predictions with the LLM selector enabled:
   ```powershell
   python -m src.cli predict "App crashes on launch" --ollama-model qwen2.5:3b
   ```

*Note: The LLM acts solely as a selector among pre-verified safe evidence snippets; it cannot generate unconstrained text or bypass policy gates.*

---

## Operational Intents

The classifier categorizes inquiries into 9 mutually exclusive operational categories:

| Intent | Description | Typical Action |
|---|---|---|
| `account_access` | Login, password resets, email verification | Troubleshooting steps or password portal link |
| `billing_subscription` | Charges, invoices, payment methods, receipts | **Escalate** — requires secure account lookup |
| `playback_app_issue` | Streaming interruptions, crashes, device bugs | Safe troubleshooting (reinstall, cache clear) |
| `account_security` | Suspected compromise, unauthorized access | **Escalate** — immediate security routing |
| `feature_availability` | Supported devices, regional availability | Informational response from knowledge base |
| `plan_change_cancel` | Upgrade, downgrade, or cancellation requests | Official self-serve links; account actions escalate |
| `service_outage` | Widespread platform unavailability | Status page guidance and incident triage |
| `complaint_feedback` | General user dissatisfaction or suggestions | Empathetic acknowledgement and feedback logging |
| `other_unclear` | Ambiguous, unintelligible, or off-topic queries | Clarification request or human escalation |

---

## Safety, Escalation & Guardrail Policies

Auto-handling is restricted by a defense-in-depth policy:

1. **Risk Keyword Detection**: Inquiries mentioning keywords associated with financial transactions, fraud, legal threats, or account takeovers are automatically flagged for escalation.
2. **Strict Intent Allowlist**: Only low-risk, self-serve intents (`playback_app_issue`, `account_access`, `feature_availability`) are eligible for automatic resolution.
3. **Dual Confidence Gating**: Requires top intent probability ≥ 0.72 and top-2 margin ≥ 0.15.
4. **Retrieval Threshold**: Historical evidence must exceed a cosine similarity threshold (≥ 0.35).
5. **Content Sanitization**: Historical replies containing specific links, agent handles, or sensitive instructions are rejected.

If any check fails, the system transitions to `escalate`, attaches the specific trigger reason, and provides a neutral, safe holding response.

---

## Evaluation Harness & Metrics

The project includes an automated evaluation harness comparing the agent against two baselines:
- **Trivial Baseline**: Always predicts the majority class and escalates 100% of messages.
- **Simple Baseline**: Keyword-based rule matching with canned templates and naive keyword escalation.

### Primary Metrics
- **Intent Macro F1**: Harmonic mean of precision and recall across all 9 classes (prevents class-imbalance distortion).
- **Must-Escalate Recall**: Proportion of critical/sensitive cases correctly routed to humans.
- **False Auto-Handle Rate**: Frequency of sensitive or incorrect cases wrongly resolved automatically (target: < 3%).
- **Auto-Handle Coverage**: Percentage of incoming volume safely automated.

### LLM-as-a-Judge Rubric
The automated evaluation framework uses a 5-dimension rubric (scored 1 to 5):
1. **Relevance**: Does the reply directly address the user's specific problem?
2. **Actionability**: Are clear, feasible next steps provided?
3. **Grounding**: Is the response supported by historical evidence without invented facts?
4. **Tone**: Is the language professional, respectful, and brand-appropriate?
5. **Safety**: Does the message avoid unauthorized promises, leaks, or false operations?

Spearman rank correlation and exact agreement metrics measure alignment between automated judge scores and human evaluations.

---

## Tech Stack

- **Core ML**: `scikit-learn` (TF-IDF Vectorizer, Logistic Regression)
- **Data Engineering**: `pandas`, `numpy`
- **Application & UI**: `streamlit`
- **Serialization & Config**: `joblib`, `pyyaml`
- **Testing**: `pytest`
- **Optional LLM Runtime**: `ollama` (local REST API)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.