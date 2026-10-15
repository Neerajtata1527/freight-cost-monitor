import pandas as pd

from backend.rag.chatbot import (
    get_anomalies_without_context,
    is_missing_context_query,
)


def test_detects_missing_context_question():
    assert is_missing_context_query(
        "Which routes have potential anomalies and no eligible context notes?"
    )


def test_detects_without_context_question():
    assert is_missing_context_query(
        "Show anomalies without context notes"
    )


def test_does_not_match_general_anomaly_question():
    assert not is_missing_context_query(
        "Explain the cost anomalies for Chennai Bangalore"
    )


def test_filters_using_matched_note_id():
    report = pd.DataFrame({
        "route": [
            "Ahmedabad-Mumbai",
            "Chennai-Bangalore",
            "Chennai-Bangalore",
            "Delhi-Jaipur",
        ],
        "week_of": pd.to_datetime([
            "2025-01-20",
            "2025-02-24",
            "2025-03-10",
            "2024-11-11",
        ]),
        "flagged": [True, True, True, True],
        "matched_note_id": ["N002", "N001", "", None],
    })

    result = get_anomalies_without_context(report)

    assert len(result) == 2
    assert set(result["route"]) == {
        "Chennai-Bangalore",
        "Delhi-Jaipur",
    }

    assert set(result["week_of"].dt.strftime("%Y-%m-%d")) == {
        "2025-03-10",
        "2024-11-11",
    }


def test_excludes_unflagged_records():
    report = pd.DataFrame({
        "route": ["Delhi-Jaipur"],
        "week_of": pd.to_datetime(["2024-11-11"]),
        "flagged": [False],
        "matched_note_id": [""],
    })

    result = get_anomalies_without_context(report)

    assert result.empty