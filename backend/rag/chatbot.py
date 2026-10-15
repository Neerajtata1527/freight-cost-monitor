# backend/rag/chatbot.py

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from backend.rag.retriever import retrieve_context_notes


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_FILE = PROJECT_ROOT / "outputs" / "final_anomaly_report.csv"
USAGE_LOG_FILE = PROJECT_ROOT / "outputs" / "llm_usage.csv"

load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
).strip()

MIN_RELEVANCE_SCORE = 0.50
MAX_RECORDS_IN_CONTEXT = 8
MAX_NOTES_PER_RECORD = 3

REQUIRED_REPORT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]


# ============================================================
# ROUTE AND VALUE NORMALIZATION
# ============================================================

def normalize_text(value: Any) -> str:
    """
    Normalize text for case-insensitive matching.

    Examples:
        Delhi Jaipur -> delhi jaipur
        Delhi-Jaipur -> delhi jaipur
        CHENNAI_BANGALORE -> chennai bangalore
    """

    if value is None:
        return ""

    text = str(value).strip().casefold()

    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_boolean(value: Any) -> bool:
    """
    Convert report values into reliable Boolean values.
    """

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    if isinstance(value, (int, float)):
        return bool(value)

    return str(value).strip().casefold() in {
        "true",
        "1",
        "yes",
        "y",
    }


def parse_number(value: Any) -> float | None:
    """
    Safely convert a value into a floating-point number.
    """

    try:
        number = float(value)

        if pd.isna(number):
            return None

        return number

    except (TypeError, ValueError):
        return None


# ============================================================
# REPORT LOADING
# ============================================================

@lru_cache(maxsize=1)
def load_report() -> pd.DataFrame:
    """
    Load the final anomaly report.

    The cache prevents repeatedly reading the same CSV during
    a single application session.
    """

    if not REPORT_FILE.exists():
        raise FileNotFoundError(
            f"Final anomaly report was not found: {REPORT_FILE}. "
            "Run the analytics and investigation pipelines first."
        )

    df = pd.read_csv(REPORT_FILE)

    missing_columns = [
        column
        for column in REQUIRED_REPORT_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The final anomaly report is missing required columns: "
            f"{missing_columns}"
        )

    df = df[REQUIRED_REPORT_COLUMNS].copy()

    df["route"] = df["route"].astype(str).str.strip()

    df["week_of"] = pd.to_datetime(
        df["week_of"],
        errors="coerce",
    )

    numeric_columns = [
        "cost_per_tonne_km",
        "vs_own_history",
        "vs_similar_routes",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["flagged"] = df["flagged"].apply(parse_boolean)

    df["matched_note_id"] = df["matched_note_id"].fillna("")

    df["reason"] = df["reason"].fillna("")

    df = df.dropna(
        subset=["route", "week_of"]
    ).copy()

    return df.sort_values(
        ["route", "week_of"]
    ).reset_index(drop=True)


# ============================================================
# ROUTE IDENTIFICATION
# ============================================================

def identify_mentioned_routes(
    query: str,
    report: pd.DataFrame | None = None,
) -> list[str]:
    """
    Identify route names mentioned in the user's question.

    Supports variations such as:
        Delhi-Jaipur
        Delhi Jaipur
        Delhi_Jaipur
    """

    if report is None:
        report = load_report()

    normalized_query = normalize_text(query)

    matched_routes = []

    routes = sorted(
        report["route"].dropna().unique(),
        key=len,
        reverse=True,
    )

    for route in routes:
        normalized_route = normalize_text(route)

        if not normalized_route:
            continue

        pattern = (
            r"(?<!\w)"
            + re.escape(normalized_route)
            + r"(?!\w)"
        )

        if re.search(pattern, normalized_query):
            matched_routes.append(route)

    return matched_routes


# ============================================================
# DETERMINISTIC ANALYTICS SELECTION
# ============================================================

def select_relevant_records(
    query: str,
    report: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Select relevant report records for the user's question.

    Route-specific questions prioritize that route's flagged weeks.
    Questions without an identifiable route prioritize flagged
    records with the largest available positive deviations.
    """

    if report is None:
        report = load_report()

    if report.empty:
        return report.copy()

    mentioned_routes = identify_mentioned_routes(
        query,
        report,
    )

    if mentioned_routes:
        selected = report[
            report["route"].isin(mentioned_routes)
        ].copy()

        if selected.empty:
            return selected

        flagged = selected[
            selected["flagged"]
        ].copy()

        if not flagged.empty:
            selected = flagged

        selected = selected.sort_values(
            "week_of",
            ascending=False,
        )

        return selected.head(
            MAX_RECORDS_IN_CONTEXT
        ).sort_values(
            ["route", "week_of"]
        )

    flagged = report[
        report["flagged"]
    ].copy()

    if flagged.empty:
        return report.tail(
            MAX_RECORDS_IN_CONTEXT
        ).copy()

    own_deviation = flagged[
        "vs_own_history"
    ].fillna(0)

    peer_deviation = flagged[
        "vs_similar_routes"
    ].fillna(0)

    flagged["_priority"] = (
        own_deviation.clip(lower=0)
        + peer_deviation.clip(lower=0)
    )

    selected = flagged.sort_values(
        "_priority",
        ascending=False,
    ).head(MAX_RECORDS_IN_CONTEXT)

    return selected.drop(
        columns=["_priority"]
    ).sort_values(
        ["route", "week_of"]
    )


# ============================================================
# AUTHORITATIVE NUMERICAL SUMMARY
# ============================================================

def build_deterministic_metric_summary(
    records: pd.DataFrame,
) -> str:
    """
    Build the authoritative numerical summary directly from
    the analytics report.

    The LLM must not calculate or rewrite these metrics.
    """

    if records is None or records.empty:
        return "No matching analytics records were found."

    lines = [
        "AUTHORITATIVE ANALYTICS FACTS",
        "Source: outputs/final_anomaly_report.csv",
        "",
        "These values are calculated by the analytics pipeline.",
        "Preserve the values and signs exactly as displayed.",
        "",
    ]

    for _, row in records.iterrows():

        route = str(row["route"])

        week = row["week_of"]

        if pd.notna(week):
            week_text = week.strftime("%Y-%m-%d")
        else:
            week_text = "Unavailable"

        cost = parse_number(
            row["cost_per_tonne_km"]
        )

        own_change = parse_number(
            row["vs_own_history"]
        )

        peer_change = parse_number(
            row["vs_similar_routes"]
        )

        flagged = parse_boolean(
            row["flagged"]
        )

        lines.append(f"Route: {route}")
        lines.append(f"Week starting: {week_text}")

        if cost is not None:
            lines.append(
                f"Cost per tonne-km: {cost:.6f}"
            )
        else:
            lines.append(
                "Cost per tonne-km: unavailable"
            )

        if own_change is not None:
            lines.append(
                "Change versus own historical baseline: "
                f"{own_change:+.2f}%"
            )
        else:
            lines.append(
                "Change versus own historical baseline: unavailable"
            )

        if peer_change is not None:
            lines.append(
                "Change versus similar routes: "
                f"{peer_change:+.2f}%"
            )
        else:
            lines.append(
                "Change versus similar routes: unavailable"
            )

        lines.append(
            f"Flagged by analytics pipeline: "
            f"{'Yes' if flagged else 'No'}"
        )

        reason = str(row.get("reason", "")).strip()

        if reason:
            lines.append(
                f"Recorded investigation reason: {reason}"
            )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# CONTEXT-NOTE RELEVANCE FILTERING
# ============================================================

def is_cost_relevant_note(note: dict) -> bool:
    """
    Reject notes that explicitly say the event did not change
    freight rates or costs.

    This is a conservative text filter, not a causal inference
    system.
    """

    text = str(
        note.get("note", "")
    ).strip().casefold()

    non_cost_phrases = [
        "without a rate change",
        "without any rate change",
        "no rate change",
        "rates remained unchanged",
        "no change in freight rates",
        "no change in rates",
        "costs remained unchanged",
        "no additional freight cost",
    ]

    return not any(
        phrase in text
        for phrase in non_cost_phrases
    )


def retrieve_validated_notes_for_record(
    route: str,
    week_of: str,
) -> list[dict]:
    """
    Retrieve context notes using the project's existing retriever.

    The retriever is expected to perform route/date validation
    through backend.rag.evidence_validator.
    """

    query = (
        f"Freight cost anomaly for route {route} "
        f"during the week starting {week_of}. "
        "Find operational events that may be relevant to "
        "freight cost changes."
    )

    try:
        notes = retrieve_context_notes(
            query=query,
            route=route,
            week_of=week_of,
            k=10,
            lookback_days=14,
        )

    except Exception as exc:
        print(
            "Context retrieval failed for "
            f"{route}, {week_of}: {exc}"
        )

        return []

    if not notes:
        return []

    valid_notes = []

    for note in notes:

        if not isinstance(note, dict):
            continue

        score = parse_number(
            note.get("score")
        )

        if score is None:
            score = 0.0

        if score < MIN_RELEVANCE_SCORE:
            continue

        # The retriever should return only date/route-validated
        # notes. Also reject explicit failed validation markers.
        if note.get("eligible") is False:
            continue

        if note.get("is_valid") is False:
            continue

        validation_reason = str(
            note.get("validation_reason", "")
        ).strip().casefold()

        if validation_reason.startswith(
            ("invalid:", "rejected:", "not eligible:")
        ):
            continue

        if not is_cost_relevant_note(note):
            continue

        valid_notes.append(
            {
                "note_id": str(
                    note.get(
                        "note_id",
                        note.get("id", "Unknown"),
                    )
                ),
                "date": str(
                    note.get("date", "Unknown")
                )[:10],
                "applies_to": str(
                    note.get("applies_to", "Unknown")
                ),
                "note": str(
                    note.get("note", "")
                ),
                "score": score,
                "validation_reason": str(
                    note.get(
                        "validation_reason",
                        "Route/date validation passed.",
                    )
                ),
            }
        )

    valid_notes.sort(
        key=lambda item: (
            -item["score"],
            item["note_id"],
        )
    )

    return valid_notes[:MAX_NOTES_PER_RECORD]


# ============================================================
# CONTEXT CONSTRUCTION
# ============================================================

def build_record_context(
    row: pd.Series,
) -> str:
    """
    Build context for a single route-week record.
    """

    route = str(row["route"])

    week_value = row["week_of"]

    if pd.notna(week_value):
        week_of = week_value.strftime("%Y-%m-%d")
    else:
        week_of = str(week_value)

    notes = retrieve_validated_notes_for_record(
        route=route,
        week_of=week_of,
    )

    lines = [
        f"Route: {route}",
        f"Week starting: {week_of}",
        f"Flagged: {parse_boolean(row['flagged'])}",
    ]

    reason = str(
        row.get("reason", "")
    ).strip()

    if reason:
        lines.append(
            f"Recorded reason: {reason}"
        )

    lines.append("")

    if notes:
        lines.append(
            "Eligible supporting context notes:"
        )

        for note in notes:
            lines.append(
                f"Note ID: {note['note_id']}"
            )

            lines.append(
                f"Note date: {note['date']}"
            )

            lines.append(
                f"Applies to: {note['applies_to']}"
            )

            lines.append(
                f"Relevance score: {note['score']:.3f}"
            )

            lines.append(
                f"Validation: {note['validation_reason']}"
            )

            lines.append(
                f"Note text: {note['note']}"
            )

            lines.append("")

    else:
        lines.append(
            "Eligible supporting context notes: none."
        )

        lines.append(
            "The cause of the anomaly is not established "
            "by the available context notes."
        )

    return "\n".join(lines)


def build_analytics_context(
    records: pd.DataFrame,
) -> str:
    """
    Construct validated context for the selected analytics records.
    """

    if records is None or records.empty:
        return "No matching analytics records were found."

    sections = []

    for _, row in records.iterrows():
        sections.append(
            build_record_context(row)
        )

    return "\n\n".join(sections)


# ============================================================
# LLM USAGE AND COST LOGGING
# ============================================================

def log_usage(
    model: str,
    usage_metadata: dict | None,
) -> None:
    """
    Append token usage and estimated API cost to CSV.

    Estimated cost is calculated only when the corresponding
    environment variables contain valid prices.
    """

    if not usage_metadata:
        usage_metadata = {}

    input_tokens = usage_metadata.get(
        "input_tokens",
        usage_metadata.get("prompt_tokens", 0),
    )

    output_tokens = usage_metadata.get(
        "output_tokens",
        usage_metadata.get("completion_tokens", 0),
    )

    total_tokens = usage_metadata.get(
        "total_tokens",
        input_tokens + output_tokens,
    )

    try:
        input_tokens = int(input_tokens or 0)
    except (TypeError, ValueError):
        input_tokens = 0

    try:
        output_tokens = int(output_tokens or 0)
    except (TypeError, ValueError):
        output_tokens = 0

    try:
        total_tokens = int(total_tokens or 0)
    except (TypeError, ValueError):
        total_tokens = input_tokens + output_tokens

    input_rate = parse_number(
        os.getenv("GROQ_INPUT_COST_PER_MILLION_USD")
    )

    output_rate = parse_number(
        os.getenv("GROQ_OUTPUT_COST_PER_MILLION_USD")
    )

    estimated_cost = None

    if input_rate is not None and output_rate is not None:
        estimated_cost = (
            input_tokens * input_rate
            + output_tokens * output_rate
        ) / 1_000_000

    row = {
        "timestamp": pd.Timestamp.now(
            tz="UTC"
        ).isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
    }

    USAGE_LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    usage_df = pd.DataFrame([row])

    usage_df.to_csv(
        USAGE_LOG_FILE,
        mode="a",
        header=not USAGE_LOG_FILE.exists(),
        index=False,
    )


# ============================================================
# GROQ CLIENT
# ============================================================

@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Initialize and cache the Groq chat model.
    """

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your project .env file."
        )

    if not GROQ_MODEL:
        raise RuntimeError(
            "GROQ_MODEL is missing. Set a supported Groq model "
            "in your .env file."
        )

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )


# ============================================================
# DETERMINISTIC CONTEXT-STATUS QUERIES
# ============================================================

def is_missing_context_query(query: str) -> bool:
    """Detect questions asking for flagged anomalies without context notes."""

    normalized = normalize_text(query)
    patterns = [
        r"\bno eligible context notes?\b",
        r"\bwithout (?:any )?(?:eligible )?context notes?\b",
        r"\bmissing context notes?\b",
        r"\bno supporting context notes?\b",
        r"\bwithout supporting notes?\b",
        r"\bno eligible notes?\b",
        r"\bwithout (?:any )?eligible notes?\b",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def get_anomalies_without_context(report: pd.DataFrame) -> pd.DataFrame:
    """Return flagged records whose report has no matched context-note ID.

    The final investigation report is authoritative for this query. This
    function intentionally does not run vector retrieval or ask the LLM to
    decide whether a record has a matched note.
    """

    if report is None or report.empty:
        return report.copy() if report is not None else pd.DataFrame()

    note_ids = (
        report["matched_note_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    missing_note = note_ids.eq("") | note_ids.str.casefold().isin(
        {"nan", "none", "null", "no eligible context note", "no eligible context notes"}
    )

    result = report.loc[report["flagged"] & missing_note].copy()
    return result.sort_values(["route", "week_of"]).reset_index(drop=True)


def format_missing_context_answer(records: pd.DataFrame) -> str:
    """Format a context-status answer deterministically from report values."""

    if records is None or records.empty:
        return (
            "## Flagged anomalies without eligible context notes\n\n"
            "No flagged route-week records without a matched context-note ID "
            "were found in the final report."
        )

    lines = [
        "## Flagged anomalies without eligible context notes",
        "",
        f"Found **{len(records)} route-week records**.",
        "",
        "| Route | Week starting | Cost per tonne-km | Change vs history | Change vs similar routes |",
        "|---|---|---:|---:|---:|",
    ]

    for _, row in records.iterrows():
        week = row["week_of"]
        week_text = week.strftime("%Y-%m-%d") if pd.notna(week) else "Unavailable"
        cost = parse_number(row["cost_per_tonne_km"])
        own_change = parse_number(row["vs_own_history"])
        peer_change = parse_number(row["vs_similar_routes"])

        cost_text = f"{cost:.6f}" if cost is not None else "Unavailable"
        own_text = f"{own_change:+.2f}%" if own_change is not None else "Unavailable"
        peer_text = f"{peer_change:+.2f}%" if peer_change is not None else "Unavailable"
        route_text = str(row["route"]).replace("|", "\\|")

        lines.append(
            f"| {route_text} | {week_text} | {cost_text} | {own_text} | {peer_text} |"
        )

    lines.extend([
        "",
        "These records were flagged by the analytics pipeline, but the final "
        "report contains no matched context-note ID for them.",
        "",
        "The available context does not establish the underlying cause. "
        "Human review is required; a potential anomaly is not proof of fraud.",
    ])
    return "\n".join(lines)


# ============================================================
# CHATBOT RESPONSE GENERATION
# ============================================================

SYSTEM_PROMPT = """
You are a freight-cost investigation assistant.

Your job is to explain potential freight-cost anomalies using
analytics records and validated context notes.

IMPORTANT RULES:

1. The analytics report is the authoritative source for numerical facts.

2. The application constructs the numerical summary directly from
   the analytics report. Do not independently recalculate these metrics.

3. Do not repeat numerical percentages or cost values in your
   explanatory prose. The application displays the authoritative
   figures separately.

4. Never claim that a positive percentage is negative or that a
   negative percentage is positive.

5. Explain the two comparisons accurately:
    - Historical baseline: the current week's route cost per tonne-kilometre
      compared with its rolling average across up to the eight preceding
      calendar weeks. The current week is excluded from its own baseline.
    - Similar routes: the current week's cost compared with the average cost
      of other routes with the same route_type during that week, excluding
      the route being evaluated.

6. Do not claim that peer routes are matched by distance, cargo type, or
   other characteristics unless the supplied analytics implementation says so.

7. Use only the eligible context notes provided in the supplied context.

8. Never reintroduce excluded, unvalidated, unrelated, or out-of-date notes.

9. A context note is supporting evidence, not proof of causation.

10. Do not invent causes for cost increases.

11. If no eligible note explains an anomaly, state that the cause cannot
    be established from the available evidence.

12. Do not claim fraud, manipulation, or wrongdoing merely because a route
    has been flagged.

13. Recommend human investigation when the available evidence cannot
    establish the cause.

14. Distinguish clearly between observed cost changes, potential operational
    explanations, and confirmed causes.

15. Keep the explanation concise and specific to the user's question.

16. If the supplied context does not answer the question, state that
    limitation instead of guessing.
"""


def answer_question(query: str) -> str:
    """
    Answer a freight-cost question using deterministic analytics,
    validated context notes, and Groq.

    The numerical summary is generated separately from the LLM
    response to prevent incorrect numerical signs.
    """

    query = str(query or "").strip()

    if not query:
        return "Please enter a question about freight costs or anomalies."

    try:
        report = load_report()

        # Answer context-status questions deterministically from the final
        # report. Do not let the LLM decide which records have note matches.
        if is_missing_context_query(query):
            missing_context_records = get_anomalies_without_context(report)
            return format_missing_context_answer(missing_context_records)

        records = select_relevant_records(
            query=query,
            report=report,
        )

        if records.empty:
            return (
                "I could not find matching route-week records "
                "in the final anomaly report."
            )

        # Generate numerical facts deterministically.
        metric_summary = build_deterministic_metric_summary(
            records
        )

        # Retrieve validated context for each selected record.
        analytics_context = build_analytics_context(
            records
        )

        user_prompt = f"""
USER QUESTION:
{query}

AUTHORITATIVE NUMERICAL SUMMARY:
{metric_summary}

ROUTE-WEEK CONTEXT AND VALIDATED NOTES:
{analytics_context}

YOUR TASK:
Explain what the available evidence indicates about the user's
question.

Do not repeat numerical percentages or cost values in your prose.
Do not recalculate or reinterpret the numerical summary.

Use only the supplied eligible context notes. If no note establishes
the cause of a cost increase, explicitly state that the cause remains
unknown and recommend an appropriate human review.

Do not claim that an eligible note caused the anomaly unless the
supplied evidence explicitly establishes that relationship.
"""

        llm = get_llm()

        response = llm.invoke(
            [
                SystemMessage(
                    content=SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=user_prompt
                ),
            ]
        )

        llm_answer = str(
            response.content
        ).strip()

        usage_metadata = getattr(
            response,
            "usage_metadata",
            None,
        )

        try:
            log_usage(
                model=GROQ_MODEL,
                usage_metadata=usage_metadata,
            )
        except Exception as logging_error:
            # A logging failure should not discard a valid answer.
            print(
                f"Warning: unable to log LLM usage: {logging_error}"
            )

        # Numerical facts remain separate from the LLM's prose.
        return (
            f"{metric_summary}\n\n"
            "EXPLANATION\n"
            f"{llm_answer}"
        )

    except FileNotFoundError as exc:
        return (
            "The analytics report is not available yet. "
            "Run the analytics and investigation pipelines first.\n\n"
            f"Details: {exc}"
        )

    except RuntimeError as exc:
        return (
            "The chatbot configuration needs attention.\n\n"
            f"Details: {exc}"
        )

    except Exception as exc:
        print(
            f"Chatbot error: {type(exc).__name__}: {exc}"
        )

        return (
            "I could not complete the investigation because an "
            "unexpected error occurred. Please check the application "
            "terminal for details and try again."
        )


# ============================================================
# OPTIONAL COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":
    print("Freight Cost Investigation Chatbot")
    print("Type 'exit' to stop.\n")

    while True:
        user_query = input("You: ").strip()

        if user_query.casefold() in {
            "exit",
            "quit",
        }:
            break

        print()
        print(answer_question(user_query))
        print()