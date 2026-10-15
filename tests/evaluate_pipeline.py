from pathlib import Path

import pandas as pd

from backend.analytics.mvp_pipeline import run_pipeline
from backend.services.investigation_service import (
    run_investigation,
    OUTPUT_COLUMNS,
    OUTPUT_FILE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHIPMENT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "shipment_records.csv"
)

EXPECTED_RUNS = 3

COMPARISON_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]


def validate_output_schema(report: pd.DataFrame) -> None:
    """Verify the required output schema."""

    assert report.columns.tolist() == OUTPUT_COLUMNS, (
        f"Unexpected columns: {report.columns.tolist()}"
    )

    assert len(report) > 0, "The report is empty."

    assert not report.duplicated(
        subset=["route", "week_of"]
    ).any(), "Duplicate route-week records found."

    assert report["route"].notna().all()
    assert report["week_of"].notna().all()

    assert report["cost_per_tonne_km"].notna().all()

    assert report["flagged"].notna().all()

    print("PASS: Output schema and required fields.")


def validate_cost_calculations(
    report: pd.DataFrame,
) -> None:
    """
    Independently recalculate weekly cost per tonne-km
    from the original shipment records.
    """

    shipments = pd.read_csv(SHIPMENT_FILE)

    required_columns = [
        "origin",
        "destination",
        "route_type",
        "quantity_tonnes",
        "distance_km",
        "freight_cost_inr",
        "shipment_date",
    ]

    missing = set(required_columns) - set(shipments.columns)

    assert not missing, (
        f"Shipment CSV is missing columns: {sorted(missing)}"
    )

    shipments["shipment_date"] = pd.to_datetime(
        shipments["shipment_date"],
        errors="raise",
    )

    shipments["quantity_tonnes"] = pd.to_numeric(
        shipments["quantity_tonnes"],
        errors="raise",
    )

    shipments["distance_km"] = pd.to_numeric(
        shipments["distance_km"],
        errors="raise",
    )

    shipments["freight_cost_inr"] = pd.to_numeric(
        shipments["freight_cost_inr"],
        errors="raise",
    )

    shipments["route"] = (
        shipments["origin"].astype(str).str.strip()
        + "-"
        + shipments["destination"].astype(str).str.strip()
    )

    # Calculate the Monday of each shipment's week.
    shipments["week_of"] = (
        shipments["shipment_date"]
        - pd.to_timedelta(
            shipments["shipment_date"].dt.weekday,
            unit="D",
        )
    ).dt.strftime("%Y-%m-%d")

    shipments["tonne_km"] = (
        shipments["quantity_tonnes"]
        * shipments["distance_km"]
    )

    expected = (
        shipments.groupby(
            ["route", "route_type", "week_of"],
            as_index=False,
        )
        .agg(
            total_freight_cost=(
                "freight_cost_inr",
                "sum",
            ),
            total_tonne_km=(
                "tonne_km",
                "sum",
            ),
        )
    )

    expected["expected_cost"] = (
        expected["total_freight_cost"]
        / expected["total_tonne_km"]
    )

    actual = report[
        ["route", "week_of", "cost_per_tonne_km"]
    ].copy()

    actual["week_of"] = pd.to_datetime(
        actual["week_of"]
    ).dt.strftime("%Y-%m-%d")

    comparison = actual.merge(
        expected[
            [
                "route",
                "week_of",
                "expected_cost",
            ]
        ],
        on=["route", "week_of"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    assert comparison["_merge"].eq("both").all(), (
        "Some report records do not match source shipment data."
    )

    assert (
        comparison["expected_cost"].notna().all()
    ), "Some expected costs could not be calculated."

    assert (
        comparison["expected_cost"] > 0
    ).all(), "Expected costs should be positive."

  

    differences = (
        comparison["cost_per_tonne_km"]
        - comparison["expected_cost"]
    ).abs()

    assert differences.max() < 0.00001, (
        "Cost-per-tonne-km calculation mismatch. "
        f"Maximum difference: {differences.max()}"
    )

    print("PASS: Weekly cost calculations match source data.")


def validate_flag_consistency(
    report: pd.DataFrame,
    weekly_metrics: pd.DataFrame,
) -> None:
    """Ensure the report preserves analytics-engine decisions."""

    expected_flags = weekly_metrics[
        [
            "route",
            "week_of",
            "anomaly_candidate",
        ]
    ].copy()

    expected_flags["week_of"] = pd.to_datetime(
        expected_flags["week_of"]
    ).dt.strftime("%Y-%m-%d")

    actual_flags = report[
        ["route", "week_of", "flagged"]
    ].copy()

    actual_flags["week_of"] = pd.to_datetime(
        actual_flags["week_of"]
    ).dt.strftime("%Y-%m-%d")

    actual_flags["flagged"] = (
        actual_flags["flagged"].astype(bool)
    )

    expected_flags["flagged_expected"] = (
        expected_flags.pop("anomaly_candidate").astype(bool)
    )

    comparison = actual_flags.merge(
        expected_flags,
        on=["route", "week_of"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    assert comparison["_merge"].eq("both").all(), (
        "Report records do not match analytics records."
    )

    assert (
        comparison["flagged"]
        == comparison["flagged_expected"]
    ).all(), "Anomaly flags differ from the analytics engine."

    print("PASS: Anomaly flags match analytics decisions.")


def normalize_report(
    report: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize values for repeatability comparisons."""

    normalized = report[COMPARISON_COLUMNS].copy()

    normalized["week_of"] = pd.to_datetime(
        normalized["week_of"]
    ).dt.strftime("%Y-%m-%d")

    normalized["flagged"] = (
        normalized["flagged"].astype(bool)
    )

    normalized["matched_note_id"] = (
        normalized["matched_note_id"]
        .where(normalized["matched_note_id"].notna(), None)
    )

    normalized = normalized.sort_values(
        ["week_of", "route"]
    ).reset_index(drop=True)

    return normalized


def main() -> None:
    reports = []
    metrics_by_run = []

    print("=" * 60)
    print("FREIGHT COST MONITOR — EVALUATION")
    print("=" * 60)

    for run_number in range(1, EXPECTED_RUNS + 1):
        print(f"\nRun {run_number}/{EXPECTED_RUNS}")

        # Refresh deterministic analytics and database.
        weekly_metrics = run_pipeline()

        # Generate the investigation report.
        report = run_investigation()

        validate_output_schema(report)
        validate_cost_calculations(report)
        validate_flag_consistency(
            report,
            weekly_metrics,
        )

        # Check the persisted CSV as well.
        persisted_report = pd.read_csv(OUTPUT_FILE)

        assert persisted_report.columns.tolist() == OUTPUT_COLUMNS

        assert len(persisted_report) == len(report)

        reports.append(normalize_report(report))
        metrics_by_run.append(weekly_metrics.copy())

        print(f"Weekly records: {len(report)}")
        print(f"Flagged records: {int(report['flagged'].sum())}")
        print(
            "Matched notes:",
            int(report["matched_note_id"].notna().sum()),
        )

    baseline = reports[0]

    for index, current in enumerate(reports[1:], start=2):
        try:
            pd.testing.assert_frame_equal(
                baseline,
                current,
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as exc:
            raise AssertionError(
                f"Reproducibility failure on run {index}."
            ) from exc

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Runs completed: {EXPECTED_RUNS}")
    print("Cost calculations: PASS")
    print("Output schema: PASS")
    print("Anomaly flags: PASS")
    print("Matched note IDs: reproducible")
    print("Report reasons: reproducible")
    print("Overall reproducibility: PASS")
    print(f"Final report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()