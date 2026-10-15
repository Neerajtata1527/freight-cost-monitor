# 🚚 Freight Cost Monitor

A data-driven freight-cost monitoring and investigation assistant that calculates weekly shipping costs, identifies potential route-level anomalies, retrieves relevant operational context, and presents the results through a Streamlit dashboard and conversational assistant.

The project is designed to support **human investigation**, not to make automatic fraud accusations or treat contextual notes as proof of causation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Input Data](#input-data)
- [Analytics Methodology](#analytics-methodology)
- [Context Retrieval and Validation](#context-retrieval-and-validation)
- [Conversational Assistant](#conversational-assistant)
- [Installation and Setup](#installation-and-setup)
- [Run the Application](#run-the-application)
- [Run Tests and Evaluation](#run-tests-and-evaluation)
- [Output Files](#output-files)
- [LLM Usage and Cost Logging](#llm-usage-and-cost-logging)
- [Example Questions](#example-questions)
- [Limitations and Design Decisions](#limitations-and-design-decisions)
- [Future Improvements](#future-improvements)

## Overview

Freight rates can vary due to operational disruptions, route conditions, demand, and other factors. This project monitors weekly freight cost per tonne-kilometre for each route and highlights records that differ from the route's own history or from comparable routes.

For each route-week record, the system:

1. Calculates freight cost per tonne-kilometre from shipment-level data.
2. Compares the current week with the route's previous history.
3. Compares the route with other routes of the same route type during the same week.
4. Flags potential anomalies using a configurable threshold.
5. Searches supplied context notes and validates their route and date applicability.
6. Produces an investigation report with a reason and any eligible matched note ID.
7. Displays the results in Streamlit and lets users ask questions through a Groq-powered assistant.

### Current evaluation snapshot

Using the supplied project data, the evaluation completed with the following results:

| Metric | Result |
|---|---:|
| Shipment records processed | 2,940 |
| Routes represented | 7 |
| Weekly route records | 728 |
| Potential anomalies flagged | 20 |
| Context matches in the current report | 4 |
| Reproducibility runs in the evaluation harness | 3 |

These figures describe the current supplied dataset and report. They will change if the input data or configuration changes.

## Features

- **Weekly cost analytics:** Aggregates shipment-level costs by route and week.
- **Historical comparison:** Compares weekly cost with a rolling baseline built from up to eight preceding calendar weeks.
- **Peer comparison:** Compares a route with other routes of the same `route_type` during the same week.
- **Configurable anomaly threshold:** Uses a provisional 20% threshold in the current implementation.
- **Context-note retrieval:** Embeds and searches supplied operational notes using ChromaDB and FastEmbed.
- **Evidence validation:** Checks whether notes apply to the route and whether their dates are compatible with the anomaly week.
- **Investigation report:** Produces an eight-column CSV report suitable for downstream analysis.
- **Interactive dashboard:** Includes route and date filters, a cost trend chart, route-week records, and an investigation panel.
- **Conversational assistant:** Uses a Groq-hosted language model to explain analytics and eligible context.
- **Deterministic missing-context answers:** Filters flagged records with no matched context-note ID in Python instead of asking the LLM to make that decision.
- **LLM usage logging:** Records model name, token usage, timestamp, and estimated cost when pricing is configured.
- **Reproducibility checks:** Validates the output schema, cost calculations, anomaly flags, and consistency across repeated runs.

## Architecture

```text
Shipment records CSV ─────────────┐
                                  │
                                  ▼
                        Data validation
                                  │
                                  ▼
                     Weekly analytics pipeline
                                  │
                  ┌───────────────┴────────────────┐
                  ▼                                ▼
          Weekly metrics CSV                 SQLite database
                                                   │
                                                   ▼
Context notes CSV ──► Document loader ──► ChromaDB vector store
                                                   │
                                                   ▼
                                      Route/date evidence validation
                                                   │
                                                   ▼
                                      Investigation service
                                                   │
                                                   ▼
                                      final_anomaly_report.csv
                                                   │
                              ┌────────────────────┴─────────────────┐
                              ▼                                      ▼
                      Streamlit dashboard                 Freight Assistant
                                                                  │
                                                    Deterministic report facts
                                                    + validated context notes
                                                                  │
                                                                  ▼
                                                           Groq chat model
```

**Design principle:** Python owns calculations, anomaly flags, report filtering, and evidence eligibility. The language model explains supplied results; it is not the authority for numerical calculations or whether a record has a matched note.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python 3.11 |
| Data processing | Pandas, NumPy |
| Dashboard | Streamlit, Altair |
| Backend/API framework dependency | Flask |
| LLM integration | LangChain, `langchain-groq`, Groq API |
| Embeddings | FastEmbed (`BAAI/bge-small-en-v1.5`) |
| Vector store | ChromaDB |
| Persistence | SQLite, CSV |
| Configuration | `python-dotenv` |
| Environment and dependency management | `uv` |
| Testing | pytest |

Flask is included in the project environment. The primary user interface described here is the Streamlit application.

## Project Structure

The core files are organized approximately as follows:

```text
freight-cost-monitor/
├── backend/
│   ├── analytics/
│   │   └── mvp_pipeline.py
│   ├── database/
│   │   └── db.py
│   ├── rag/
│   │   ├── chatbot.py
│   │   ├── document_loader.py
│   │   ├── evidence_validator.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   └── services/
│       └── investigation_service.py
├── data/
│   ├── raw/
│   │   ├── shipment_records.csv
│   │   ├── context_notes.csv
│   │   └── sample_output_format_v2.csv
│   └── processed/
│       └── chroma_context_notes/
├── frontend/
│   └── streamlit_app.py
├── outputs/
│   ├── weekly_metrics.csv
│   ├── final_anomaly_report.csv
│   └── llm_usage.csv
├── tests/
│   ├── evaluate_pipeline.py
│   └── test_evidence_validator.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

The exact local tree may contain additional files. Generated databases, vector-store files, virtual environments, and secrets should not be committed unless explicitly required.

## Input Data

Place the supplied input CSVs in `data/raw/`.

### `shipment_records.csv`

The analytics pipeline expects these columns:

| Column | Description |
|---|---|
| `origin` | Shipment origin |
| `destination` | Shipment destination |
| `route_type` | Route category used for peer comparisons |
| `quantity_tonnes` | Shipment quantity in tonnes |
| `distance_km` | Distance in kilometres |
| `freight_cost_inr` | Freight cost in INR |
| `shipment_date` | Shipment date |

The pipeline validates required columns, parses dates and numeric fields, and rejects invalid records such as non-positive quantities or distances and negative freight costs.

### `context_notes.csv`

The context-note loader expects:

| Column | Description |
|---|---|
| `note_id` | Stable identifier for the note |
| `date` | Date associated with the note |
| `applies_to` | Route or route scope to which the note applies |
| `note` | Description of the operational event |

Notes may refer to a specific route or to all routes. A note is not considered a confirmed explanation merely because its text is semantically similar to an anomaly.

### `sample_output_format_v2.csv`

This file provides the required output format for the final report. The investigation service preserves the expected eight-column schema.

## Analytics Methodology

### 1. Weekly cost per tonne-kilometre

For each shipment:

```text
tonne_km = quantity_tonnes × distance_km
```

For each route and week:

```text
cost_per_tonne_km =
    sum(freight_cost_inr) / sum(quantity_tonnes × distance_km)
```

This is a ratio of aggregated freight cost to aggregated tonne-kilometres. It is not the average of individual shipment cost ratios.

Weeks are grouped by their Monday start date.

### 2. Historical baseline

For each route, the current weekly cost is compared with the average of available weekly costs from up to the **eight preceding calendar weeks**.

- The current week is excluded from its own baseline.
- Missing calendar weeks are not treated as zero-cost observations.
- If no prior baseline is available, the historical comparison remains unavailable.

### 3. Peer baseline

For a route-week record, the peer baseline is the average weekly cost per tonne-kilometre of **other routes with the same `route_type` during the same week**.

- The route being evaluated is excluded from its own peer average.
- The current implementation does not claim to match peers by distance, cargo type, or other criteria unless those criteria are explicitly added to the analytics logic.
- If no eligible peer baseline is available, the peer comparison remains unavailable.

### 4. Anomaly detection

The current implementation uses a configurable threshold of 20%.

A record is a potential anomaly when its cost is above an available historical baseline or peer baseline by at least the configured threshold. The investigation reason identifies the comparison or comparisons that triggered the flag.

The threshold is a project configuration, not a universal industry standard.

### 5. Investigation output

The final report contains these columns in this order:

```text
route
week_of
cost_per_tonne_km
vs_own_history
vs_similar_routes
flagged
matched_note_id
reason
```

A missing `matched_note_id` means that the investigation service did not record an eligible context-note match for that route-week record.

## Context Retrieval and Validation

The context pipeline loads the supplied notes, creates document embeddings, and stores them in a persistent ChromaDB collection. Retrieval uses FastEmbed embeddings and relevance scores.

The investigation workflow applies route and date validation before using a note as potential supporting context. The current implementation also uses a configurable relevance-score threshold and a 14-day lookback heuristic for notes whose event duration is not explicitly known.

Where an event's duration is known and explicitly configured, the validator can check whether the anomaly week overlaps that interval. For example, the flooding note for Chennai–Bangalore is accepted for weeks overlapping the documented disruption period and rejected for a week after the documented event ended.

Important distinctions:

- **Semantic relevance is not enough:** A note must also pass the route/date checks.
- **A matched note is contextual evidence:** It does not prove the note caused the cost change.
- **No eligible match does not mean no cause exists:** It means the supplied notes did not provide an eligible match under the current rules.
- **Human review remains necessary:** The application does not automatically declare fraud or misconduct.

The 14-day lookback is a configurable project heuristic, not a general rule for all operational events.

## Conversational Assistant

The assistant is implemented in `backend/rag/chatbot.py` and is called by the Streamlit frontend.

For a typical question, it:

1. Loads the final anomaly report.
2. Identifies any route mentioned in the question.
3. Selects relevant report records deterministically.
4. Builds an authoritative numerical summary from the report.
5. Retrieves and validates context notes for the selected route-week records.
6. Sends the supplied facts and eligible context to the Groq chat model.
7. Logs token usage when usage metadata is available.

Questions about flagged records without eligible context-note matches are handled with deterministic Python filtering, so the language model does not decide which records qualify for that list.

### Example questions

- `Why did the cost per tonne-km increase for Delhi Jaipur route?`
- `Explain the cost anomalies for Chennai Bangalore.`
- `Which routes have potential anomalies and no eligible context notes?`
- `Which anomalies have eligible context notes?`
- `Are any of these anomalies proof of fraud?`

The assistant should preserve reported numerical values and signs, distinguish observations from explanations, and acknowledge when the available evidence does not establish a cause.

## Installation and Setup

### Prerequisites

- Python 3.11
- `uv`
- A Groq API key for conversational LLM responses
- The supplied CSV files

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd freight-cost-monitor
```

Replace `<your-repository-url>` with your actual repository URL.

### 2. Create or synchronize the environment

If the repository contains `pyproject.toml` and `uv.lock`, install the locked dependencies:

```bash
uv sync
```

If development dependencies are not included in the default sync configuration, install them as appropriate for your project setup.

If setting up the project from scratch, install the dependencies listed in `pyproject.toml` and include pytest for testing.

### 3. Configure environment variables

Create a `.env` file in the project root. Do not commit this file.

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

# Optional: set these only when you have the applicable model prices.
GROQ_INPUT_COST_PER_MILLION_USD=
GROQ_OUTPUT_COST_PER_MILLION_USD=
```

Use a model identifier currently supported by your Groq account. The model shown above is the one used in the current project configuration; availability can change.

Estimated API cost remains blank when the input and output prices are not configured. Token usage can still be logged.

### 4. Place the input files

Ensure the following files exist:

```text
data/raw/shipment_records.csv
data/raw/context_notes.csv
data/raw/sample_output_format_v2.csv
```

Do not replace the supplied case-study inputs with synthetic data when reproducing the reported evaluation results.

## Run the Application

Run the commands from the project root.

### 1. Generate weekly analytics

```bash
uv run python -m backend.analytics.mvp_pipeline
```

This validates the shipment records, calculates weekly route metrics, writes the weekly metrics output, and persists the metrics to SQLite.

### 2. Generate the investigation report

```bash
uv run python -m backend.services.investigation_service
```

This retrieves and validates context notes for potential anomalies and writes:

```text
outputs/final_anomaly_report.csv
```

### 3. Launch Streamlit

```bash
uv run streamlit run frontend/streamlit_app.py
```

Open the local URL printed by Streamlit in your browser, normally `http://localhost:8501`.

If the source CSVs or pipeline logic have changed, rerun the analytics and investigation steps before using the dashboard so that the report reflects the latest data.

## Run Tests and Evaluation

### Evidence-validator tests

```bash
$env:PYTHONPATH = "."
uv run pytest tests/test_evidence_validator.py -v
```

The PowerShell environment-variable command above is for Windows. On macOS/Linux, use:

```bash
PYTHONPATH=. uv run pytest tests/test_evidence_validator.py -v
```

### Full pipeline evaluation

```bash
uv run python tests/evaluate_pipeline.py
```

The evaluation harness checks:

- The expected output schema and required fields.
- Weekly cost calculations against the source shipment data.
- Consistency between analytics decisions and report anomaly flags.
- Reproducibility across three runs for costs, flags, matched note IDs, and investigation reasons.

The latest recorded evaluation passed the schema, cost-calculation, flag-consistency, and three-run reproducibility checks.

The chatbot-specific behavior should also be tested manually in Streamlit, including questions that ask for records with and without eligible context-note matches.

## Output Files

| File | Purpose |
|---|---|
| `outputs/weekly_metrics.csv` | Aggregated weekly route metrics and comparison fields |
| `outputs/final_anomaly_report.csv` | Final investigation report with the required eight columns |
| `outputs/llm_usage.csv` | Per-request model and token-usage records, with estimated cost when configured |
| `data/freight_monitor.db` | SQLite database for persisted weekly metrics and anomaly assessments |
| `data/processed/chroma_context_notes/` | Persistent ChromaDB vector-store data for context notes |

Generated outputs and local databases can be recreated by running the pipeline. Whether to commit them depends on the submission requirements; avoid committing secrets, API keys, or unnecessary generated vector-store artifacts.

## LLM Usage and Cost Logging

The assistant logs usage information to `outputs/llm_usage.csv` when the model response includes usage metadata.

The log contains:

- UTC timestamp
- Model name
- Input token count
- Output token count
- Total token count
- Estimated cost in USD, when configured

The estimated cost is calculated from the configured input and output prices per million tokens. If either price is blank or invalid, the cost field remains empty. These are estimates based on the configured rates, not a substitute for the usage and billing information in the provider account.

Existing rows with blank cost fields are not automatically recalculated when pricing is added later.

## Limitations and Design Decisions

- The anomaly threshold is currently configured as 20% and may need calibration against business requirements.
- The historical baseline uses available observations from up to eight preceding calendar weeks; routes with limited history may have no baseline.
- Peer comparisons use the same `route_type` and week, excluding the route being evaluated. They do not imply richer matching criteria.
- Context-note coverage is limited to the supplied notes.
- The 14-day note lookback is a heuristic when an event's duration is not explicitly known.
- Event intervals must be explicitly configured when a note describes a known multi-day event and reliable interval validation is required.
- Relevance scores and text-based cost-relevance checks are safeguards, not proof of causality.
- A note that describes an event without a rate change should not be presented as an explanation for a freight-cost increase merely because it is semantically similar.
- LLM output can still be imperfect. Deterministic analytics and filtering are used for authoritative numerical facts and record eligibility.
- The application is an investigation-support tool. Human review is required before making operational or financial decisions.

## Future Improvements

Potential extensions include:

- Add unit and integration tests for the chatbot's deterministic query handling.
- Track request latency, request status, and request IDs alongside token usage.
- Add dashboard filters for anomaly type and context-match status.
- Add a review workflow for analysts to record investigation outcomes.
- Calibrate the anomaly threshold using business feedback and historical outcomes.
- Improve evidence provenance by recording the validation decision and reason for every retrieved note.
- Add automated tests for report schema and evidence-match consistency to continuous integration.
- Add deployment configuration and operational monitoring if the application is hosted.

---

## Responsible Use

This application highlights **potential freight-cost anomalies**, not confirmed fraud. Context notes may provide useful operational background, but they do not independently prove why a cost changed. Treat every flagged record as a lead for human investigation, and base conclusions on verified operational and financial evidence.
