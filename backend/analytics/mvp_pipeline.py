
from pathlib import Path

import numpy as np
import pandas as pd

from backend.database.db import save_weekly_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHIPMENT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "shipment_records.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "outputs" / "weekly_metrics.csv"
)

ANOMALY_THRESHOLD_PERCENT = 20.0
HISTORY_WINDOW_WEEKS = 8

REQUIRED_COLUMNS = [
    "origin",
    "destination",
    "route_type",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
]


def load_shipments() -> pd.DataFrame:
    if not SHIPMENT_FILE.exists():
        raise FileNotFoundError(
            f"Shipment dataset not found: {SHIPMENT_FILE}"
        )

    df = pd.read_csv(SHIPMENT_FILE)

    missing_columns = (
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df["shipment_date"] = pd.to_datetime(
        df["shipment_date"],
        errors="coerce",
    )

    numeric_columns = [
        "quantity_tonnes",
        "distance_km",
        "freight_cost_inr",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "Required columns contain missing or invalid values."
        )

    if (df["quantity_tonnes"] <= 0).any():
        raise ValueError(
            "Quantity must be greater than zero."
        )

    if (df["distance_km"] <= 0).any():
        raise ValueError(
            "Distance must be greater than zero."
        )

    if (df["freight_cost_inr"] < 0).any():
        raise ValueError(
            "Freight cost cannot be negative."
        )

    for column in [
        "origin",
        "destination",
        "route_type",
    ]:
        df[column] = df[column].astype(str).str.strip()

        if df[column].eq("").any():
            raise ValueError(
                f"Column '{column}' contains empty values."
            )

    df["route"] = (
        df["origin"] + "-" + df["destination"]
    )

    df["week_of"] = (
        df["shipment_date"].dt.normalize()
        - pd.to_timedelta(
            df["shipment_date"].dt.weekday,
            unit="D",
        )
    )

    df["tonne_km"] = (
        df["quantity_tonnes"] * df["distance_km"]
    )

    return df


def calculate_weekly_costs(
    shipments: pd.DataFrame,
) -> pd.DataFrame:

    group_columns = [
        "route",
        "route_type",
        "week_of",
    ]

    weekly = (
        shipments.groupby(
            group_columns,
            as_index=False,
        )
        .agg(
            total_freight_cost=(
                "freight_cost_inr",
                "sum",
            ),
            total_quantity_tonnes=(
                "quantity_tonnes",
                "sum",
            ),
            total_tonne_km=(
                "tonne_km",
                "sum",
            ),
            shipment_count=(
                "route",
                "size",
            ),
        )
    )

    weekly["cost_per_tonne_km"] = (
        weekly["total_freight_cost"]
        / weekly["total_tonne_km"]
    )

    return weekly.sort_values(
        ["route", "route_type", "week_of"]
    ).reset_index(drop=True)


def calculate_historical_baseline(
    weekly: pd.DataFrame,
) -> pd.DataFrame:

    if weekly.empty:
        return weekly.copy()

    all_weeks = pd.date_range(
        start=weekly["week_of"].min(),
        end=weekly["week_of"].max(),
        freq="W-MON",
    )

    route_keys = weekly[
        ["route", "route_type"]
    ].drop_duplicates()

    calendar = route_keys.merge(
        pd.DataFrame({"week_of": all_weeks}),
        how="cross",
    )

    calendar = calendar.merge(
        weekly,
        on=[
            "route",
            "route_type",
            "week_of",
        ],
        how="left",
        validate="one_to_one",
    )

    calendar = calendar.sort_values(
        ["route", "route_type", "week_of"]
    ).reset_index(drop=True)

    calendar["own_history_avg"] = (
        calendar.groupby(
            ["route", "route_type"]
        )["cost_per_tonne_km"]
        .transform(
            lambda values: (
                values.shift(1)
                .rolling(
                    window=HISTORY_WINDOW_WEEKS,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    calendar["vs_own_history"] = np.where(
        calendar["own_history_avg"] > 0,
        (
            (
                calendar["cost_per_tonne_km"]
                - calendar["own_history_avg"]
            )
            / calendar["own_history_avg"]
        ) * 100,
        np.nan,
    )

    return calendar.loc[
        calendar["total_freight_cost"].notna()
    ].reset_index(drop=True)


def calculate_peer_baseline(
    weekly: pd.DataFrame,
) -> pd.DataFrame:

    weekly = weekly.copy()

    if weekly.empty:
        weekly["peer_avg"] = pd.Series(dtype=float)
        weekly["vs_similar_routes"] = pd.Series(
            dtype=float
        )
        return weekly

    group_columns = [
        "week_of",
        "route_type",
    ]

    peer_statistics = (
        weekly.groupby(group_columns)[
            "cost_per_tonne_km"
        ]
        .agg(
            type_week_total="sum",
            type_week_count="count",
        )
        .reset_index()
    )

    weekly = weekly.merge(
        peer_statistics,
        on=group_columns,
        how="left",
        validate="many_to_one",
    )

    peer_count = (
        weekly["type_week_count"] - 1
    )

    weekly["peer_avg"] = np.where(
        peer_count > 0,
        (
            weekly["type_week_total"]
            - weekly["cost_per_tonne_km"]
        ) / peer_count,
        np.nan,
    )

    weekly["vs_similar_routes"] = np.where(
        weekly["peer_avg"] > 0,
        (
            (
                weekly["cost_per_tonne_km"]
                - weekly["peer_avg"]
            )
            / weekly["peer_avg"]
        ) * 100,
        np.nan,
    )

    return weekly.drop(
        columns=[
            "type_week_total",
            "type_week_count",
        ]
    )


def detect_anomaly_candidates(
    weekly: pd.DataFrame,
) -> pd.DataFrame:

    weekly = weekly.copy()

    own_history_anomaly = (
        (
            weekly["vs_own_history"]
            >= ANOMALY_THRESHOLD_PERCENT
        )
        & (
            weekly["own_history_avg"]
            < weekly["cost_per_tonne_km"]
        )
    )

    peer_anomaly = (
        (
            weekly["vs_similar_routes"]
            >= ANOMALY_THRESHOLD_PERCENT
        )
        & (
            weekly["peer_avg"]
            < weekly["cost_per_tonne_km"]
        )
    )

    weekly["anomaly_candidate"] = (
        own_history_anomaly | peer_anomaly
    ).fillna(False).astype(bool)

    return weekly


def run_pipeline() -> pd.DataFrame:

    shipments = load_shipments()

    weekly = calculate_weekly_costs(
        shipments
    )

    weekly = calculate_historical_baseline(
        weekly
    )

    weekly = calculate_peer_baseline(
        weekly
    )

    weekly = detect_anomaly_candidates(
        weekly
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weekly.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    save_weekly_metrics(weekly)

    print("\nAnalytics pipeline completed")
    print("-" * 40)

    print(
        f"Shipment records: {len(shipments)}"
    )

    print(
        f"Routes monitored: "
        f"{weekly['route'].nunique()}"
    )

    print(
        f"Weekly route records: {len(weekly)}"
    )

    print(
        f"Potential anomalies: "
        f"{int(weekly['anomaly_candidate'].sum())}"
    )

    print(
        f"Historical baselines unavailable: "
        f"{int(weekly['own_history_avg'].isna().sum())}"
    )

    print(
        f"Peer baselines unavailable: "
        f"{int(weekly['peer_avg'].isna().sum())}"
    )

    print(
        f"CSV output: {OUTPUT_FILE}"
    )

    print("SQLite database updated.")

    return weekly


if __name__ == "__main__":
    run_pipeline()
