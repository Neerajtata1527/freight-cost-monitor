import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "freight_monitor.db"


def get_connection():
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS weekly_route_metrics (
                route TEXT NOT NULL,
                route_type TEXT NOT NULL,
                week_of TEXT NOT NULL,
                total_freight_cost REAL,
                total_quantity_tonnes REAL,
                total_tonne_km REAL,
                cost_per_tonne_km REAL,
                own_history_avg REAL,
                vs_own_history REAL,
                peer_avg REAL,
                vs_similar_routes REAL,
                anomaly_candidate INTEGER NOT NULL,
                PRIMARY KEY (route, route_type, week_of)
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_assessments (
                route TEXT NOT NULL,
                route_type TEXT NOT NULL,
                week_of TEXT NOT NULL,
                anomaly_candidate INTEGER NOT NULL,
                matched_note_id TEXT,
                verdict TEXT NOT NULL,
                reason TEXT,
                review_status TEXT NOT NULL,
                PRIMARY KEY (route, route_type, week_of)
            )
        """)


def save_weekly_metrics(metrics: pd.DataFrame):
    initialize_database()

    columns = [
        "route",
        "route_type",
        "week_of",
        "total_freight_cost",
        "total_quantity_tonnes",
        "total_tonne_km",
        "cost_per_tonne_km",
        "own_history_avg",
        "vs_own_history",
        "peer_avg",
        "vs_similar_routes",
        "anomaly_candidate",
    ]

    records = metrics[columns].copy()

    records["week_of"] = pd.to_datetime(
        records["week_of"]
    ).dt.strftime("%Y-%m-%d")

    records["anomaly_candidate"] = (
        records["anomaly_candidate"].astype(int)
    )

    with get_connection() as connection:
        records.to_sql(
            "weekly_route_metrics",
            connection,
            if_exists="replace",
            index=False,
        )

        connection.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_weekly_route_metrics_key
            ON weekly_route_metrics (
                route,
                route_type,
                week_of
            )
        """)

        connection.execute("""
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
            SELECT
                route,
                route_type,
                week_of,
                anomaly_candidate,
                NULL,
                'pending_evidence',
                NULL,
                'pending'
            FROM weekly_route_metrics
            WHERE anomaly_candidate = 1
            ON CONFLICT (route, route_type, week_of)
            DO UPDATE SET
                anomaly_candidate = excluded.anomaly_candidate
        """)


def get_weekly_metrics(route=None):
    query = "SELECT * FROM weekly_route_metrics"
    params = ()

    if route:
        query += " WHERE route = ?"
        params = (route,)

    query += " ORDER BY week_of"

    with get_connection() as connection:
        return pd.read_sql_query(
            query,
            connection,
            params=params,
        )


def get_pending_assessments():
    query = """
        SELECT *
        FROM anomaly_assessments
        WHERE review_status = 'pending'
        ORDER BY week_of DESC
    """

    with get_connection() as connection:
        return pd.read_sql_query(
            query,
            connection,
        )