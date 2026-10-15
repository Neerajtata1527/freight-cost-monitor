from backend.rag.evidence_validator import (
    validate_note_for_anomaly,
)


N001 = {
    "note_id": "N001",
    "date": "2025-02-24",
    "applies_to": "Chennai-Bangalore",
    "applies_to_normalized": "chennai-bangalore",
}


def test_n001_is_valid_during_flooding_week():
    is_valid, reason = validate_note_for_anomaly(
        note_metadata=N001,
        route="Chennai-Bangalore",
        week_of="2025-03-03",
    )

    assert is_valid is True


def test_n001_is_invalid_after_flooding_ended():
    is_valid, reason = validate_note_for_anomaly(
        note_metadata=N001,
        route="Chennai-Bangalore",
        week_of="2025-03-10",
    )

    assert is_valid is False
    assert "does not overlap" in reason


def test_n001_is_invalid_for_another_route():
    is_valid, reason = validate_note_for_anomaly(
        note_metadata=N001,
        route="Delhi-Jaipur",
        week_of="2025-03-03",
    )

    assert is_valid is False


def test_n001_is_valid_for_last_week_of_flooding():
    is_valid, reason = validate_note_for_anomaly(
        note_metadata=N001,
        route="Chennai-Bangalore",
        week_of="2025-03-03",
    )

    assert is_valid is True
    assert "overlaps" in reason