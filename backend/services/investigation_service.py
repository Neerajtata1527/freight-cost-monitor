from pathlib import Path
import logging

import pandas as pd

from backend.database.db import (
    get_weekly_metrics,
    get_connection,
)
from backend.rag.retriever import retrieve_context_notes


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "final_anomaly_report.csv"

# Provisional relevance threshold. Validate this against labelled examples.
MIN_RELEVANCE_SCORE = 0.50

# Required assignment output schema.
OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def format_percentage(value) -> str:
    """Format a percentage deterministically."""
    if pd.isna(value):
        return "unavailable"

    return f"{float(value):+.2f}%"


def is_cost_relevant_note(note: dict) -> bool:
    """
    Apply a conservative check to avoid treating notes that explicitly
    deny a rate change as explanations for a cost increase.

    This is a heuristic, not a substitute for human review.
    """
    text = str(note.get("note", "")).casefold()

    phrases_indicating_no_rate_increase = [
        "without a rate change",
        "no rate change",
        "without any rate change",
        "rates remained unchanged",
        "no increase in rates",
        "did not increase rates",
    ]

    if any(
        phrase in text
        for phrase in phrases_indicating_no_rate_increase
    ):
        return False

    return True


def find_matching_note(
    route: str,
    week_of: str,
    cost_per_tonne_km: float,
    vs_own_history,
    vs_similar_routes,
) -> dict | None:
    """
    Retrieve eligible notes and select the highest-ranked candidate
    that passes the configured relevance and cost-relevance checks.
    """

    query = (
        f"Investigate a potential increase in shipping cost per "
        f"tonne-kilometre for route {route} during the week "
        f"starting {week_of}. Current cost per tonne-km is "
        f"{cost_per_tonne_km:.4f}. Change versus own history: "
        f"{format_percentage(vs_own_history)}. Change versus "
        f"similar routes: {format_percentage(vs_similar_routes)}. "
        f"Find contextual evidence such as flooding, detours, fuel "
        f"price changes, temporary surcharges, demand, or truck "
        f"availability."
    )

    notes = retrieve_context_notes(
        route=route,
        week_of=week_of,
        query=query,
        k=10,
        lookback_days=14,
    )

    eligible_notes = [
        note
        for note in notes
        if note["similarity_score"] >= MIN_RELEVANCE_SCORE
        and is_cost_relevant_note(note)
    ]

    if not eligible_notes:
        return None

    eligible_notes.sort(
        key=lambda note: (
            -note["similarity_score"],
            note["note_id"],
        )
    )

    return eligible_notes[0]


def build_reason(
    row: pd.Series,
    matched_note: dict | None,
) -> str:
    """Generate a deterministic, evidence-aware investigation reason."""

    own_change = format_percentage(row["vs_own_history"])
    peer_change = format_percentage(row["vs_similar_routes"])

    if not bool(row["anomaly_candidate"]):
        return (
            "No configured anomaly threshold was exceeded. "
            f"Change versus own history: {own_change}; "
            f"change versus similar routes: {peer_change}."
        )

    reasons = []

    if (
        pd.notna(row["vs_own_history"])
        and float(row["vs_own_history"]) >= 20.0
    ):
        reasons.append(
            f"cost is {own_change} above its historical baseline"
        )

    if (
        pd.notna(row["vs_similar_routes"])
        and float(row["vs_similar_routes"]) >= 20.0
    ):
        reasons.append(
            f"cost is {peer_change} above similar routes"
        )

    if not reasons:
        reasons.append(
            "the existing analytics pipeline flagged this record"
        )

    reason = (
        "Potential anomaly: "
        + "; ".join(reasons)
        + "."
    )

    if matched_note:
        reason += (
            f" Potential context match {matched_note['note_id']} "
            f"(dated {matched_note['date']}): "
            f"{matched_note['note']} "
            "This note is contextual evidence, not proof of "
            "causation. Human review is required."
        )
    else:
        reason += (
            " No eligible, sufficiently relevant context note was "
            "found. Human review is required."
        )

    return reason


def save_assessment(
    row: pd.Series,
    matched_note: dict | None,
    reason: str,
) -> None:
    """Persist the investigation result for a potential anomaly."""

    matched_note_id = (
        matched_note["note_id"] if matched_note else None
    )

    verdict = (
        "potential_context_match"
        if matched_note
        else "no_eligible_context"
    )

    week_of = pd.Timestamp(row["week_of"]).strftime("%Y-%m-%d")

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO anomaly_assessments (
                route,
                route_type,
                week_of,
                anomaly_candidate,
                matched_note_id,
                verdict,
                reason,
                review_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(route, route_type, week_of)
            DO UPDATE SET
                anomaly_candidate = excluded.anomaly_candidate,
                matched_note_id = excluded.matched_note_id,
                verdict = excluded.verdict,
                reason = excluded.reason,
                review_status = 'pending'
            """,
            (
                row["route"],
                row["route_type"],
                week_of,
                1,
                matched_note_id,
                verdict,
                reason,
                "pending",
            ),
        )


def run_investigation() -> pd.DataFrame:
    """
    Combine deterministic analytics with validated context retrieval.

    The output contains one row per weekly route record and exactly
    the eight columns specified by the assignment.
    """

    weekly = get_weekly_metrics()

    if weekly.empty:
        raise ValueError(
            "No weekly metrics found. Run the analytics pipeline first."
        )

    required_columns = {
        "route",
        "route_type",
        "week_of",
        "cost_per_tonne_km",
        "vs_own_history",
        "vs_similar_routes",
        "anomaly_candidate",
    }

    missing = required_columns - set(weekly.columns)

    if missing:
        raise ValueError(
            f"Weekly metrics are missing columns: {sorted(missing)}"
        )

    weekly = weekly.copy()

    weekly["week_of"] = pd.to_datetime(
        weekly["week_of"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")

    weekly = weekly.sort_values(
        ["week_of", "route", "route_type"]
    ).reset_index(drop=True)

    report_rows = []
    flagged_count = 0
    matched_count = 0

    for _, row in weekly.iterrows():
        flagged = bool(row["anomaly_candidate"])
        matched_note = None

        if flagged:
            flagged_count += 1

            try:
                matched_note = find_matching_note(
                    route=str(row["route"]),
                    week_of=str(row["week_of"]),
                    cost_per_tonne_km=float(
                        row["cost_per_tonne_km"]
                    ),
                    vs_own_history=row["vs_own_history"],
                    vs_similar_routes=row["vs_similar_routes"],
                )

            except Exception:
                logger.exception(
                    "Context retrieval failed for route=%s week=%s",
                    row["route"],
                    row["week_of"],
                )

                # Preserve the anomaly even if retrieval fails.
                matched_note = None

        if matched_note:
            matched_count += 1

        reason = build_reason(
            row=row,
            matched_note=matched_note,
        )

        if flagged:
            save_assessment(
                row=row,
                matched_note=matched_note,
                reason=reason,
            )

        report_rows.append(
            {
                "route": row["route"],
                "week_of": row["week_of"],
                "cost_per_tonne_km": round(
                    float(row["cost_per_tonne_km"]),
                    6,
                ),
                "vs_own_history": (
                    round(float(row["vs_own_history"]), 2)
                    if pd.notna(row["vs_own_history"])
                    else None
                ),
                "vs_similar_routes": (
                    round(float(row["vs_similar_routes"]), 2)
                    if pd.notna(row["vs_similar_routes"])
                    else None
                ),
                "flagged": flagged,
                "matched_note_id": (
                    matched_note["note_id"]
                    if matched_note
                    else None
                ),
                "reason": reason,
            }
        )

    report = pd.DataFrame(
        report_rows,
        columns=OUTPUT_COLUMNS,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    logger.info("Investigation completed.")
    logger.info("Weekly records: %s", len(report))
    logger.info("Flagged records: %s", flagged_count)
    logger.info("Potential context matches: %s", matched_count)
    logger.info(
        "Flagged records requiring review: %s",
        flagged_count,
    )
    logger.info("Output: %s", OUTPUT_FILE)

    return report


if __name__ == "__main__":
    report = run_investigation()

    print("\nFinal report preview:")
    print(report.head(10).to_string(index=False))

    print("\nColumn validation:")
    print(report.columns.tolist())

    print("\nTotal records:", len(report))