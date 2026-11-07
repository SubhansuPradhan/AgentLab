# AgentLab

A multi-agent system that automates the machine learning workflow: from a raw CSV to a trained model, an explainability analysis, and a formal PDF report, with a critic agent that checks the results before anything is finalized.

Unlike a standard AutoML pipeline that runs a fixed sequence of steps, AgentLab splits the workflow across independent agents that each own one job, and adds a critic agent that can send the process back a step when it finds a fixable problem (a leaked column, an identifier used as a feature). That loop-back mechanism is the project's research question: does letting a system critique and revise its own work produce more trustworthy results than a pipeline that just runs top to bottom?

## Status

| Component | Status | Notes |
|---|---|---|
| Validation Agent | Built, 4 unit tests | Duplicates, missing values, constant and identifier-like columns |
| EDA Agent | Built | Statistics and six plot types |
| Hypothesis Agent | Built | `run()` proposes hypotheses and flags likely leaks before modeling, `verify()` checks them against SHAP afterwards |
| Feature Engineering Agent | Built | Per-column imputation, scaling and encoding as a real sklearn pipeline |
| Model Selection and Tuning Agent | Built | One agent: recommends a model, decides by rules whether tuning is needed, runs Optuna only if it is |
| Explainability Agent | Built | SHAP, a LIME second opinion, and a check that the model is good enough for explanations to mean anything |
| Critic Agent | Built | Rule-based checks, loop-back, and an on/off switch for the ablation |
| Report Agent | Built | Reuses each specialist's own summary, writes only the executive summary and conclusion |
| Graph wiring (`graph/workflow.py`) | Built | Tested with the real Model Selection, Critic and Report agents plus stand-ins for the other five |
| Streamlit frontend | Built | One tab per agent, a full report tab, formal PDF download. Tested with Streamlit's AppTest |

**Not yet verified, read this before a demo:**

- The full pipeline with every real agent running together on real datasets.
- A complete run against a real Ollama install, local and cloud models together.
- The research ablation itself (critic on versus off across 3 to 4 datasets). The tooling exists, the results do not yet.
- Storing runs in PostgreSQL. The database is set up, but the pipeline does not write to it yet. MLflow logging is best-effort and is skipped quietly if it fails.

Only the Validation Agent has committed pytest files. The other agents were tested ad hoc during development, which is not the same thing.

## System architecture

```mermaid
flowchart TD
    U[User uploads CSV] --> V[Validation Agent]
    V --> E[EDA Agent]
    E --> H[Hypothesis Agent]
    H --> F[Feature Engineering Agent]
    F --> M[Model Selection and Tuning Agent]
    M --> X[Explainability Agent]
    X --> VH[Verify hypotheses against SHAP]
    VH --> C{Critic Agent}
    C -- leak or identifier column found --> A[Apply revision: drop the columns]
    A --> F
    C -- anything else --> R[Report Agent]
    R --> D[User downloads PDF report]
```

The critic's loop-back is the only conditional branch in the graph. It edits the pipeline only when the evidence is deterministic: a column the Hypothesis Agent proved predicts the target almost perfectly on its own, or an identifier column the model is still using. Everything else (no signal, overfitting, small samples) is reported as a warning, because re-running the same pipeline cannot fix it. Revisions are capped (default 2), so a run always ends.

## Key design decisions

- **Rules decide, the LLM explains.** Every verdict (leak found, tune or not, approve or revise) comes from a measured rule, so it is reproducible and defensible. The LLM only writes the plain-language explanation afterwards, because small local models produced confident wrong explanations of meaningless results during development.
- **Simplest model within noise.** The Model Selection Agent recommends the simplest model whose cross-validated score is within one standard error of the best, instead of the raw top scorer.
- **Tuning is a decision, not a default.** It is skipped when the model has no settings, the features carry almost no signal, the score is suspiciously perfect, or the data is too small. A tuned model is adopted only if it improves the held-out test score by at least 1 percent.
- **The critic re-measures.** It re-scores the model itself instead of trusting numbers reported upstream.
- **The report never contradicts the specialists.** The Report Agent copies each agent's own verified summary into the report and writes only two short sections itself.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Agent orchestration | LangGraph | Shared state between agents and the conditional loop the critic needs |
| LLM runtime | Ollama | Local model for most agents, cloud models where reasoning quality matters |
| Modeling | scikit-learn, XGBoost, LightGBM | Standard, well-understood, easy to defend |
| Tuning | Optuna | Search is focused on the settings the diagnostics point to |
| Explainability | SHAP, LIME | Feature-level reasons for predictions |
| Experiment tracking | MLflow | Run comparison without building a dashboard |
| Database | PostgreSQL | Set up for experiment history, not yet written to by the pipeline |
| Frontend | Streamlit | Tabs, live progress, file upload |
| PDF report | ReportLab | Full control over a formal, numbered layout |

### Model routing

| Agent | Model | Setting in `.env` |
|---|---|---|
| Validation, EDA, Feature Engineering, Model Selection, Explainability | `llama3.2:3b` (local) | `LOCAL_MODEL` |
| Hypothesis | `gpt-oss:20b-cloud` | `HYPOTHESIS_MODEL` |
| Critic | `gpt-oss:120b-cloud` | `CRITIC_MODEL` |
| Report | `nemotron-3-nano:30b-cloud` | `REPORT_MODEL` |

The cloud models need `ollama signin`, an internet connection, and count against your Ollama cloud usage limits. The gpt-oss models ignore `think=False` and need a reasoning level instead, which the agents already handle.

## Folder structure

```
agentlab/
├── agents/
│   ├── base_agent.py                shared contract every agent follows
│   ├── validation_agent.py
│   ├── eda_agent.py
│   ├── hypothesis_agent.py          run() before modeling, verify() after SHAP
│   ├── feature_engineering_agent.py
│   ├── model_selection_agent.py     model selection and tuning in one agent
│   ├── explainability_agent.py
│   ├── critic_agent.py              checks, decision, apply_revision()
│   └── report_agent.py
├── graph/
│   ├── state.py                     every key that flows between agents
│   └── workflow.py                  the LangGraph pipeline
├── prompts/                         one prompt file per agent
├── frontend/
│   ├── app.py                       Streamlit app
│   └── pdf_report.py                formal PDF generator
├── config/settings.py               reads .env, every variable must be declared here
├── database/  memory/  ml/          week-1 scaffold, only ml/mlflow_tracking.py is used
├── data/  reports/                  uploads and generated output (gitignored)
├── tests/                           validation agent tests only so far
└── docs/                            architecture notes and experiment results
```

If `frontend/pages/` still exists from the first scaffold, delete it. Streamlit turns every file in that folder into a sidebar page.

## Setup

### 1. Python environment

Use Python 3.11.

```bash
python3.11 -m venv venv
source venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` needs at least: `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, `shap`, `lime`, `matplotlib`, `seaborn`, `langgraph`, `ollama`, `streamlit`, `reportlab`, `mlflow`, `python-dotenv`, `sqlalchemy`, `psycopg2-binary`.

### 2. Ollama

```bash
# install Ollama: https://ollama.com/download
ollama pull llama3.2:3b
ollama signin            # needed once, for the cloud models
ollama serve             # leave running in a separate terminal
```

If Ollama is not reachable, every agent falls back to a plain templated summary instead of crashing. The numbers and decisions are unaffected, only the prose.

### 3. PostgreSQL

```bash
createdb agentlab
# or with Docker:
docker run --name agentlab-pg -e POSTGRES_USER=agentlab -e POSTGRES_PASSWORD=agentlab -e POSTGRES_DB=agentlab -p 5432:5432 -d postgres
```

### 4. Environment file

Create `.env` in the project root:

```
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LOCAL_MODEL=llama3.2:3b
HYPOTHESIS_MODEL=gpt-oss:20b-cloud
CRITIC_MODEL=gpt-oss:120b-cloud
REPORT_MODEL=nemotron-3-nano:30b-cloud
DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/agentlab
MLFLOW_TRACKING_URI=./mlruns
```

A value in `.env` only reaches the code if `config/settings.py` declares a matching attribute. If an agent fails with `'Settings' object has no attribute ...`, that is why.

### 5. Initialize the database tables

```bash
python -c "from database.db import init_db; init_db()"
```

## Running things

**The whole pipeline from the command line:**

```bash
python -m graph.workflow path/to/data.csv target_column
python -m graph.workflow path/to/data.csv target_column --no-critic     # critic only records, never changes anything
python -m graph.workflow path/to/data.csv target_column --ablation      # same data, critic on versus off
python -m graph.workflow --diagram                                      # Mermaid source of the graph
```

**The web app** (run from the project root):

```bash
streamlit run frontend/app.py
```

**One agent at a time:**

```bash
python -m agents.validation_agent path/to/data.csv
python -m agents.eda_agent path/to/data.csv
python -m agents.feature_engineering_agent path/to/data.csv
python -m agents.model_selection_agent path/to/data.csv target_column
python -m agents.explainability_agent path/to/data.csv target_column [model_name]
python -m agents.hypothesis_agent path/to/data.csv target_column [model_name]
```

The Critic and Report agents have no standalone command because they need the output of every agent before them. Run them through `graph.workflow`.

**Unit tests:**

```bash
pytest tests/ -v
```

## The research question

Run every test dataset through the pipeline twice, once with the Critic Agent active and once with it switched to audit-only, using `--ablation` or `graph.workflow.run_ablation()`. In audit-only mode the critic runs every check and records its flags but always approves, so the flags show exactly what the no-critic pipeline shipped without noticing.

**Read the flags next to the scores.** A higher score with the critic off is not a win if the critic's flags show the score came from a leaked column. The datasets used for this comparison must have real, imperfect signal: a dataset whose columns are all either rules applied to other columns or random noise (as one early test dataset was) cannot tell you anything about model quality. Compare across at least 3 to 4 datasets and record results in `docs/architecture.md`.

## Known limitations

- The Hypothesis Agent screens one feature at a time, so features that only matter in combination look like noise to it. The verification step reports when SHAP disagrees, but the screen cannot see interactions up front.
- Exact SHAP can be very slow on large random forests. The Explainability Agent works inside a time budget and explains fewer rows when it has to, and says so in its reliability note.
- Numeric ID columns can slip through Validation, because its identifier check only scans text columns.
- The critic only fixes leaks and identifier columns automatically. Overfitting and weak signal are reported as warnings, not corrected.

## What is left

1. Run the full pipeline with every real agent on real datasets and fix whatever the first complete run exposes.
2. Choose 3 to 4 datasets with real signal and run the critic on/off ablation on each.
3. Write the results into `docs/architecture.md`.
4. Commit tests for the agents beyond Validation.
5. Write runs to PostgreSQL if run history is part of the final demo.