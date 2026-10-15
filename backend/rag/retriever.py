
from datetime import date, timedelta

from backend.rag.vector_store import get_vector_store
from backend.rag.evidence_validator import (
    validate_note_for_anomaly,
)


def retrieve_context_notes(
    route: str,
    week_of: str,
    query: str,
    k: int = 10,
    lookback_days: int = 14,
) -> list[dict]:
    """
    Retrieve notes for a route, then validate their dates in Python.

    ChromaDB performs route-scope filtering and semantic ranking.
    Python performs date validation to avoid string-date comparison
    limitations in ChromaDB.
    """

    anomaly_week_start = date.fromisoformat(
        str(week_of)[:10]
    )

    anomaly_week_end = (
        anomaly_week_start + timedelta(days=6)
    )

    normalized_route = route.strip().casefold()

    metadata_filter = {
        "$or": [
            {
                "applies_to_normalized": normalized_route
            },
            {
                "applies_to_normalized": "all routes"
            },
        ]
    }

    vector_store = get_vector_store()

    # Retrieve route-relevant notes. Date validation happens below.
    results = vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=k,
        filter=metadata_filter,
    )

    validated_notes = []

    for document, score in results:
        is_valid, validation_reason = (
            validate_note_for_anomaly(
                note_metadata=document.metadata,
                route=route,
                week_of=week_of,
                lookback_days=lookback_days,
            )
        )

        if not is_valid:
            continue

        validated_notes.append(
            {
                "note_id": document.metadata["note_id"],
                "date": document.metadata["date"],
                "applies_to": document.metadata["applies_to"],
                "note": document.metadata["note"],
                "similarity_score": float(score),
                "validation_reason": validation_reason,
            }
        )

    return validated_notes


if __name__ == "__main__":
    route = "Chennai-Bangalore"
    week_of = "2025-03-03"

    query = (
        f"Explain shipping cost changes for {route} "
        f"during the week starting {week_of}. "
        "Look for flooding, detours, fuel increases, "
        "transport surcharges, and truck availability."
    )

    notes = retrieve_context_notes(
        route=route,
        week_of=week_of,
        query=query,
    )

    if not notes:
        print("No eligible context notes were retrieved.")

    for note in notes:
        print(note)

