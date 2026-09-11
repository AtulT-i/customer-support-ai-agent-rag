# Run locally — setup, recovery and demo guide

Use this guide after copying the entire project folder to another machine.
No Azure account, API key, database or Ollama installation is needed for the
default demo. Internet is needed initially to install Python/dependencies and,
if not included, download the dataset. Default inference then runs locally.

## 1. What to install and copy

- **Python 3.12, 64-bit is recommended.** The previous verified project run used
  Python 3.12. Other versions have not been verified by this guide.
- A terminal; VS Code is optional.
- Copy the source, configuration and any available data/artifacts together.
- Create a **new virtual environment** on the destination machine. Do not reuse
  a copied virtual environment: it can contain machine-specific paths/binaries.

Important project files:

| File | Needed for |
|---|---|
| [app.py](app.py), source package and [config/project.yaml](config/project.yaml) | Running the application. |
| [requirements.txt](requirements.txt) | Installing direct dependencies. |
| [artifacts/intent_model.joblib](artifacts/intent_model.joblib) | Prebuilt intent classifier. |
| [artifacts/retrieval.joblib](artifacts/retrieval.joblib) | Prebuilt version-2 local search index. |
| [data/processed/spotify_pairs.csv](data/processed/spotify_pairs.csv) | Rebuilding missing/outdated artifacts. |
| [data/raw/twcs/twcs.csv](data/raw/twcs/twcs.csv) | Rebuilding processed pairs if those are missing. |
| [data/labels/golden.csv](data/labels/golden.csv) | Held-out exclusions when rebuilding; keep the supplied schema even if there are no labelled rows. |

**A copy of the full folder and a Git checkout are not necessarily equivalent.**
[.gitignore](.gitignore) excludes raw/processed data and generated artifacts.
A clone/source-only archive can therefore need the rebuild steps below.

**Trust boundary:** load joblib artifacts only from a trusted project copy.
Deserializing an unknown model artifact can execute code. If the provenance is
uncertain, rebuild from trusted source and data instead of opening the artifacts.

## 2. Windows: first-time setup

Open PowerShell and change to the copied project folder. Replace the example path
with the folder on your machine. Run commands from this project root throughout
the guide, not from the source or documentation subdirectories.

```powershell
Set-Location 'D:\Atul_RAG_Prj'
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

If `py -3.12` cannot find an interpreter, install Python 3.12 using the official
[Python downloads](https://www.python.org/downloads/) and reopen the terminal.
If the launcher is unavailable but `python --version` reports the intended
Python 3.12 installation, use `python -m venv .venv` to create the environment.

The commands deliberately call the environment's Python directly. **Activation
is not required**, so PowerShell execution-policy restrictions on activation
scripts do not need to be bypassed or changed.

If the copied folder already contains a virtual environment, leave it unused
and create a fresh one under a different name, such as `.venv-local`, substituting
that name in every command below. Do not overwrite unrelated environments.

## 3. Choose the correct startup path

### Path A — copied model and retrieval artifacts are available

This is the shortest path. Raw and processed datasets are not required for
prediction when both valid artifacts are present.

Check their presence:

```powershell
Test-Path .\artifacts\intent_model.joblib
Test-Path .\artifacts\retrieval.joblib
```

Both should return `True`. Then run a prediction:

```powershell
.\.venv\Scripts\python.exe -m src.cli predict "The app keeps pausing whenever I play a song"
```

Expected: JSON containing `intent`, `confidence`, `decision`, `reason`,
`draft_reply`, `evidence`, `evidence_tweet_ids` and `rag_mode`.
The decision may be escalation; that is not a startup error.

If the prediction succeeds, skip rebuilding and go to Section 4.
If the index is old or model loading fails, use Path B or C below.

### Path B — artifacts missing/outdated, processed pairs available

Confirm [data/processed/spotify_pairs.csv](data/processed/spotify_pairs.csv) was
copied and that the supplied label directory, including the golden CSV schema,
is present. Then build the provisional demo artifacts:

```powershell
.\.venv\Scripts\python.exe -m src.cli bootstrap --size 12000
```

This writes keyword-derived weak labels, retrains the classifier and builds a
version-2 retrieval index. It may take several minutes depending on the machine;
the exact runtime and memory requirements are not benchmarked here.

**Warning:** bootstrap overwrites generated weak labels and both model/index
artifacts. It can replace an existing human-trained model with a weak-label demo
model. Back up any artifacts you want to preserve; do not rebuild unnecessarily.
It does not create human ground truth or valid headline accuracy results.

### Path C — processed pairs also missing

1. Obtain Kaggle's [Customer Support on Twitter dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
   Kaggle access/download may require an account.
2. Extract the full dataset CSV to the configured location:
   [data/raw/twcs/twcs.csv](data/raw/twcs/twcs.csv).
3. Do not substitute [data/raw/sample.csv](data/raw/sample.csv) for the full corpus.
4. Prepare pairs and bootstrap:

```powershell
.\.venv\Scripts\python.exe -m src.cli prepare
.\.venv\Scripts\python.exe -m src.cli bootstrap --size 12000
```

Optional brand profiling, not required to launch:

```powershell
.\.venv\Scripts\python.exe -m src.cli profile
```

Preserve existing reviewed labels and the frozen golden set while rebuilding.
If the supplied label files are absent from a partial copy, obtain those files
from the original project rather than inventing replacement evaluation labels.

## 4. Launch the demo

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8501** in a browser. Keep the terminal running while using
the application. Binding to loopback limits access to the local machine; this is
not a public deployment. Usage statistics are disabled by the command above.

In the UI:

1. Leave **Use local Ollama evidence selection** unchecked.
2. Enter a customer message.
3. Click **Analyze message**.
4. Inspect intent, confidence, decision and reason.
5. Read the draft and expand historical evidence.
6. Distinguish retrieved candidates from the evidence IDs actually cited.

No response is sent, no support ticket is created and no account is changed.
Stop the server with **Ctrl+C** in its terminal.

For later runs, repeat only the launch command. Do not recreate the environment,
install dependencies or train on every startup.

## 5. Verify the installation

Use a second terminal while the demo server is running, or stop the server first:

```powershell
Set-Location 'D:\Atul_RAG_Prj'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.cli predict "Someone hacked my account"
.\.venv\Scripts\python.exe -m src.cli predict "I was charged twice and need a refund"
```

The previous verified run had **220 passing tests**. Use your own current test
summary when presenting. Third-party warnings can occur; warnings are not the
same as failed tests. The security and billing examples should recommend human
review under the supplied policy, with no evidence IDs cited in the draft.

**Passing tests measure implementation behavior, not real-world model accuracy.**

## 6. Interactive demo walkthrough

Prepare the app, [HLD diagram](docs/diagrams/hld.svg) and
[agent orchestration source](src/agent.py) in separate tabs:

| Step | Action | Suggested explanation |
|---|---|---|
| Introduction | Introduce the system architecture. | “A local support assistant that classifies messages, retrieves evidence and drafts or recommends human review.” |
| Architecture | Show the HLD diagram. | Explain offline artifact building versus per-message classification, retrieval and safety gates. |
| Test Case 1 | Enter “The app keeps pausing whenever I play a song.” | Show intent confidence, similarity scores, candidate replies and final decision. |
| Test Case 2 | Enter “Someone hacked my account.” | Explain why risk terms force escalation regardless of confidence. |
| Test Case 3 | Enter “I was charged twice and need a refund.” | Explain why account-specific billing actions need humans. |
| Verification | Show source and completed test output. | Walk through `SupportAgent.respond()`, evidence filtering and regression safeguards. |
| Limitations | State system boundaries and safety trade-offs. | Independent labels, development tuning and frozen evaluation are needed for validated scores. |

The previous playback smoke test also escalated because it selected no safe
guidance. Explain that result rather than promising an automatic answer:

> “Recognizing the issue confidently does not guarantee that a historical reply
> is safe to reuse. The agent abstains instead of inventing guidance.”

The previous six smoke messages all escalated. Do not weaken the policy just to
force a successful-looking demonstration. Test any other examples beforehand and
describe their actual results. Never describe classifier confidence as accuracy.

If the UI fails, show CLI JSON using the `predict` commands above. Avoid installing
dependencies, downloading models or rebuilding the dataset during a live presentation.

## 7. Optional Ollama — not required for the default demo

Install/start [Ollama](https://ollama.com/) separately if you want to try local
LLM evidence selection. Download a model before running:

```powershell
ollama pull qwen2.5:3b
ollama list
.\.venv\Scripts\python.exe -m src.cli predict "The app keeps pausing whenever I play a song" --ollama-model qwen2.5:3b
```

The app's sidebar offers the same option. The local service must be reachable on
port 11434. If it is not running, start the installed Ollama application; use
`ollama serve` only if no Ollama server is already running.

The selector can only choose already-filtered evidence or abstain. It cannot
generate unrestricted customer-facing text. An invalid/unavailable selector
falls back to deterministic evidence selection; explicit abstention escalates.
Cases rejected earlier by policy or evidence filtering may not call Ollama at all.

The previous environment did not have Ollama installed; live model behavior and
hardware performance were not verified. Mocked integration tests are not a live
quality benchmark. Allow extra disk/memory and model-download time if enabling it.

## 8. Optional: complete the valid evaluation workflow

For the new human-review UI, paired score worksheets, progress checks and report
steps, follow [SUBMISSION.md](SUBMISSION.md). The reviewer runs separately on port
8502 using [review_app.py](review_app.py); it does not alter the prediction demo.

**Not required to run the app.** Current demo artifacts use weak labels, and
the reviewed training/golden tables currently have no labelled rows. Therefore
`evaluate` refusing to publish metrics is expected, not a broken installation.

To complete the assessment later:

1. Human-label [data/labels/label_queue.csv](data/labels/label_queue.csv), fill
   the required intent/decision fields and set `reviewed=yes`.
2. Obtain independent annotation for a subset and resolve disagreement carefully.
3. Freeze a disjoint golden set; use a separate development split for tuning
   rather than changing settings based on golden results.
4. Train and evaluate:

```powershell
.\.venv\Scripts\python.exe -m src.cli split-labels --size 200
.\.venv\Scripts\python.exe -m src.cli train
.\.venv\Scripts\python.exe -m src.cli evaluate
```

The split requires at least 300 usable reviewed unique-thread examples for a
200-row golden set and 100 training rows; more are recommended. The represented
intents also need enough examples for stratification. The CLI refuses to replace
populated training/golden splits; archive deliberately before a new experiment.
Training still overwrites generated model/index artifacts.

With local Ollama available, optionally score replies:

```powershell
.\.venv\Scripts\python.exe -m src.cli judge --model qwen2.5:3b --size 50
.\.venv\Scripts\python.exe -m src.cli prepare-human-scores
```

This samples 50 shared examples across systems, not 50 mixed prediction rows.
Human-score the current paired sample in
[data/labels/human_reply_scores.csv](data/labels/human_reply_scores.csv), then run:

```powershell
.\.venv\Scripts\python.exe -m src.cli judge-agreement
```

**Do not regenerate existing annotations.** The CLI now refuses to overwrite a
populated queue. Likewise, do not repeatedly regenerate the golden set while tuning.

## 9. Troubleshooting

| Problem | What to do |
|---|---|
| `py` not recognized / requested Python not found | Install Python 3.12 and reopen the terminal, or use the correct installed Python executable to create the environment. |
| Environment executable missing | Create a fresh virtual environment in this folder. A copied environment is not portable. |
| `No module named streamlit`, `sklearn` or `pandas` | Install requirements using the same environment Python used to launch the app. |
| `No module named src` | Change to the project root and use `-m src.cli`; do not launch the CLI source directly. |
| Missing model or index | Use Path B if processed pairs exist; otherwise Path C. |
| “Old retrieval index” | Rebuild the artifacts using trusted data and current source. |
| sklearn version warning / artifact cannot deserialize | Confirm requirements and environment; rebuild from trusted source/data rather than assuming joblib is portable across versions. |
| Raw dataset not found | Put the full dataset at the configured location or update [config/project.yaml](config/project.yaml). |
| Port 8501 already occupied | Use port 8502 in the launch command and open the corresponding browser address. |
| Browser does not open automatically | Manually open the local address printed by Streamlit. |
| App seems slow on first load | Loading sparse matrices/model files takes time; subsequent Streamlit reruns also currently reload artifacts. Performance has not been benchmarked for every machine. |
| Ollama connection fails | Disable the optional checkbox for the default demo, or start/test the local Ollama service separately. |
| Every example escalates | Inspect `reason` and `rag_mode`; conservative filters and weak training can limit coverage. This alone is not a launch failure. |
| “Golden set must contain 150–250 labelled rows” | Complete the reviewed evaluation workflow. Do not manufacture golden labels to bypass validation. |
| HLD appears as source / Mermaid not rendering | Open [the standalone HLD](docs/diagrams/hld.svg), or use Markdown Preview with Ctrl+Shift+V. |

Alternate-port launch:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
```

## 10. macOS/Linux command equivalents

Windows/Python 3.12 is the previously verified environment. The project uses
portable Python paths, but these OS-specific setup steps have not been tested
as part of the recorded Windows validation. With Python 3.12 installed:

```bash
cd /path/to/copied/project
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip check
./.venv/bin/python -m src.cli predict "The app keeps pausing whenever I play a song"
./.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

If artifacts are missing, follow Paths B/C using `./.venv/bin/python` instead of
the Windows environment executable. Some Linux installations require the OS's
Python venv support package before environment creation works.

## 11. Copy-and-run checklist

- [ ] Copied source and configuration together; changed to the project root.
- [ ] Installed compatible Python and created a fresh local environment.
- [ ] Installed dependencies and checked them with `pip check`.
- [ ] Loaded trusted artifacts successfully, or rebuilt from trusted data.
- [ ] Ran one CLI prediction and started Streamlit on loopback.
- [ ] Tried the demo test messages before presenting.
- [ ] Kept optional Ollama disabled unless installed and tested.
- [ ] Preserved human annotations and frozen evaluation data.
- [ ] Can explain why a draft may escalate and why confidence is not accuracy.

Further reading: [architecture](docs/architecture.md),
and [project README](README.md).