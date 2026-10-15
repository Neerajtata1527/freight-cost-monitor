from datetime import date, timedelta
from typing import Any


# Explicit event intervals for notes whose duration is known.
#
# These intervals describe when the event was active.
# They do not establish that the event caused a cost increase.
KNOWN_EVENT_INTERVALS = {
    "N001": {
        "start": date(2025, 2, 24),
        "end": date(2025, 3, 8),
        "description": "Flooding disrupted truck movement",
    },
}


def _parse_date(value: Any) -> date | None:
    """
    Convert a date-like value to a date.
    Return None when the value cannot be parsed.
    """

    if value is None:
        return None

    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _normalize_route(value: Any) -> str:
    """
    Normalize route names for matching.

    Examples:
        Delhi-Jaipur -> delhi-jaipur
        Delhi Jaipur -> delhi-jaipur
    """

    if value is None:
        return ""

    return "-".join(
        str(value)
        .strip()
        .casefold()
        .replace("_", "-")
        .replace(" ", "-")
        .split("-")
    )


def validate_note_for_anomaly(
    note_metadata: dict,
    route: str,
    week_of: str,
    lookback_days: int = 14,
) -> tuple[bool, str]:
    """
    Validate route applicability and date relevance.

    Known events use their explicit start/end intervals.
    Notes without a known interval use a conservative date fallback.

    Returns:
        (is_valid, explanation)
    """

    note_id = str(note_metadata.get("note_id", "")).strip()

    note_date = _parse_date(note_metadata.get("date"))
    week_start = _parse_date(week_of)

    if week_start is None:
        return False, "Invalid anomaly week date."

    if note_date is None:
        return False, "Context note has an invalid date."

    week_end = week_start + timedelta(days=6)

    note_route = _normalize_route(
        note_metadata.get(
            "applies_to_normalized",
            note_metadata.get("applies_to", ""),
        )
    )

    requested_route = _normalize_route(route)

    # Validate route scope first.
    if note_route not in {
        requested_route,
        "all-routes",
        "all routes",
    }:
        return (
            False,
            "The context note does not apply to this route.",
        )

    # Known event: use the explicitly defined event interval.
    if note_id in KNOWN_EVENT_INTERVALS:
        event = KNOWN_EVENT_INTERVALS[note_id]

        event_start = event["start"]
        event_end = event["end"]

        # The anomaly week and event interval must overlap.
        overlaps_event = (
            week_start <= event_end
            and week_end >= event_start
        )

        if not overlaps_event:
            return (
                False,
                (
                    f"The event described by {note_id} was active "
                    f"from {event_start.isoformat()} to "
                    f"{event_end.isoformat()}, which does not "
                    "overlap this anomaly week."
                ),
            )

        return (
            True,
            (
                f"The anomaly week overlaps the known event interval "
                f"for {note_id}: {event_start.isoformat()} to "
                f"{event_end.isoformat()}."
            ),
        )

    # Unknown event duration: use a conservative fallback.
    #
    # The note must have been recorded no earlier than the configured
    # lookback period and no later than the end of the anomaly week.
    earliest_eligible_date = (
        week_start - timedelta(days=lookback_days)
    )

    if note_date > week_end:
        return (
            False,
            "The note was recorded after the anomaly week.",
        )

    if note_date < earliest_eligible_date:
        return (
            False,
            (
                "The note is older than the permitted lookback "
                "period, and its event duration is not known."
            ),
        )

    return (
        True,
        (
            "Route and note-date checks passed, but the event "
            "duration is not explicitly known."
        ),
    )