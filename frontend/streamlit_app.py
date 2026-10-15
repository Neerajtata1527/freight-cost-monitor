from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import altair as alt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.rag.chatbot import answer_question


REPORT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "final_anomaly_report.csv"
)


st.set_page_config(
    page_title="Freight Cost Monitor",
    page_icon="🚚",
    layout="wide",
)


@st.cache_data
def load_report() -> pd.DataFrame:
    """Load the final analytics report."""

    if not REPORT_FILE.exists():
        return pd.DataFrame()

    report = pd.read_csv(REPORT_FILE)

    required_columns = [
        "route",
        "week_of",
        "cost_per_tonne_km",
        "vs_own_history",
        "vs_similar_routes",
        "flagged",
        "matched_note_id",
        "reason",
    ]

    if report.empty:
        return report

    missing = set(required_columns) - set(report.columns)

    if missing:
        raise ValueError(
            f"Report is missing columns: {sorted(missing)}"
        )

    report["week_of"] = pd.to_datetime(
        report["week_of"],
        errors="coerce",
    )

    report["cost_per_tonne_km"] = pd.to_numeric(
        report["cost_per_tonne_km"],
        errors="coerce",
    )

    report["vs_own_history"] = pd.to_numeric(
        report["vs_own_history"],
        errors="coerce",
    )

    report["vs_similar_routes"] = pd.to_numeric(
        report["vs_similar_routes"],
        errors="coerce",
    )

    report["flagged"] = (
        report["flagged"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin(["true", "1"])
    )

    report["matched_note_id"] = (
        report["matched_note_id"]
        .replace(r"^\s*$", pd.NA, regex=True)
    )

    return report


def format_note_id(value) -> str:
    """Display missing note IDs without showing NaN."""

    if pd.isna(value):
        return "No eligible context note"

    value = str(value).strip()

    if not value or value.casefold() == "nan":
        return "No eligible context note"

    return value


def format_number(value, decimals: int = 2) -> str:
    if pd.isna(value):
        return "Unavailable"

    return f"{float(value):+.{decimals}f}%"


st.title("🚚 Freight Cost Monitor")

st.caption(
    "Monitor weekly shipping costs, investigate potential "
    "anomalies, and inspect validated contextual evidence."
)


report = load_report()


if report.empty:
    st.warning(
        "The final report is missing or empty. "
        "Run the analytics and investigation pipelines first."
    )

    st.code(
        "uv run python -m backend.analytics.mvp_pipeline\n"
        "uv run python -m backend.services.investigation_service"
    )

    st.stop()


# ==================================================
# Sidebar filters
# ==================================================

st.sidebar.header("Filters")

all_routes = sorted(
    report["route"].dropna().unique()
)

selected_routes = st.sidebar.multiselect(
    "Routes",
    options=all_routes,
    default=all_routes,
)

flagged_only = st.sidebar.checkbox(
    "Show potential anomalies only",
    value=False,
)


available_dates = report["week_of"].dropna()

if available_dates.empty:
    st.warning("No valid reporting dates were found.")
    st.stop()


minimum_date = available_dates.min().date()
maximum_date = available_dates.max().date()


selected_dates = st.sidebar.date_input(
    "Week starting between",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date,
)


# ==================================================
# Apply filters
# ==================================================

filtered = report[
    report["route"].isin(selected_routes)
].copy()


if flagged_only:
    filtered = filtered[
        filtered["flagged"]
    ]


if isinstance(selected_dates, (tuple, list)):
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates

        if start_date and end_date:
            filtered = filtered[
                filtered["week_of"].dt.date.between(
                    start_date,
                    end_date,
                )
            ]

    elif len(selected_dates) == 1:
        selected_date = selected_dates[0]

        filtered = filtered[
            filtered["week_of"].dt.date
            == selected_date
        ]


# ==================================================
# KPI cards
# ==================================================

total_weekly_records = len(report)

filtered_record_count = len(filtered)

total_anomalies = int(
    report["flagged"].sum()
)

filtered_anomalies = int(
    filtered["flagged"].sum()
)

total_context_matches = int(
    report.loc[
        report["flagged"],
        "matched_note_id",
    ].notna().sum()
)

filtered_context_matches = int(
    filtered.loc[
        filtered["flagged"],
        "matched_note_id",
    ].notna().sum()
)


st.subheader("Monitoring overview")

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Total weekly records",
    f"{total_weekly_records:,}",
)

col2.metric(
    "Filtered records",
    f"{filtered_record_count:,}",
)

col3.metric(
    "Potential anomalies",
    f"{filtered_anomalies:,}",
    delta=(
        f"{total_anomalies:,} in full report"
        if filtered_anomalies != total_anomalies
        else None
    ),
)

col4.metric(
    "Potential context matches",
    f"{filtered_context_matches:,}",
    delta=(
        f"{total_context_matches:,} in full report"
        if filtered_context_matches != total_context_matches
        else None
    ),
)


st.caption(
    "All flagged records remain subject to human review, "
    "including those with a potential context match."
)


# ==================================================
# Weekly shipping-cost trend
# ==================================================

st.subheader("Weekly shipping-cost trend")


if not filtered.empty:

    trend = (
        filtered.groupby(
            "week_of",
            as_index=False,
        )
        .agg(
            average_cost_per_tonne_km=(
                "cost_per_tonne_km",
                "mean",
            )
        )
        .dropna()
        .sort_values("week_of")
    )

    if not trend.empty:

        chart = (
            alt.Chart(trend)
            .mark_line(
                point=False,
            )
            .encode(
                x=alt.X(
                    "week_of:T",
                    title="Week starting",
                    axis=alt.Axis(
                        format="%b %Y",
                        labelOverlap=True,
                        tickCount=12,
                    ),
                ),
                y=alt.Y(
                    "average_cost_per_tonne_km:Q",
                    title="Average cost per tonne-km",
                ),
                tooltip=[
                    alt.Tooltip(
                        "week_of:T",
                        title="Week starting",
                        format="%Y-%m-%d",
                    ),
                    alt.Tooltip(
                        "average_cost_per_tonne_km:Q",
                        title="Average cost per tonne-km",
                        format=".4f",
                    ),
                ],
            )
            .properties(
                height=320,
            )
            .interactive()
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )

    else:
        st.info("No valid costs are available for the selected filters.")

else:
    st.info("No records match the selected filters.")


# ==================================================
# Weekly route records
# ==================================================

st.subheader("Weekly route records")


display_columns = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]


if not filtered.empty:

    display_df = filtered[
        display_columns
    ].copy()

    display_df["week_of"] = (
        display_df["week_of"]
        .dt.strftime("%Y-%m-%d")
    )

    display_df["matched_note_id"] = (
        display_df["matched_note_id"]
        .apply(format_note_id)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered[
        display_columns
    ].copy()

    csv_data["week_of"] = (
        csv_data["week_of"]
        .dt.strftime("%Y-%m-%d")
    )

    csv_data = csv_data.where(
        pd.notna(csv_data),
        None,
    )

    st.download_button(
        label="Download filtered report",
        data=csv_data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="filtered_freight_report.csv",
        mime="text/csv",
    )

else:
    st.info("No records match the selected filters.")


# ==================================================
# Anomaly investigation
# ==================================================

st.subheader("Investigate a potential anomaly")


flagged_df = filtered[
    filtered["flagged"]
].copy()


if flagged_df.empty:

    st.info(
        "No potential anomalies match the current filters."
    )

else:

    flagged_df["record_label"] = (
        flagged_df["route"]
        + " | "
        + flagged_df["week_of"].dt.strftime("%Y-%m-%d")
    )

    flagged_df = flagged_df.sort_values(
        "week_of",
        ascending=False,
    )

    selected_label = st.selectbox(
        "Choose a route-week record",
        options=flagged_df["record_label"].tolist(),
    )

    selected_record = flagged_df[
        flagged_df["record_label"] == selected_label
    ].iloc[0]


    left, right = st.columns(2)


    with left:

        st.markdown(
            f"**Route:** {selected_record['route']}"
        )

        st.markdown(
            f"**Week starting:** "
            f"{selected_record['week_of'].strftime('%Y-%m-%d')}"
        )

        st.markdown(
            f"**Cost per tonne-km:** "
            f"{selected_record['cost_per_tonne_km']:.6f}"
        )


    with right:

        st.markdown(
            "**Change versus own history:** "
            + format_number(
                selected_record["vs_own_history"]
            )
        )

        st.markdown(
            "**Change versus similar routes:** "
            + format_number(
                selected_record["vs_similar_routes"]
            )
        )

        st.markdown(
            "**Matched note ID:** "
            + format_note_id(
                selected_record["matched_note_id"]
            )
        )


    st.markdown("**Investigation reason**")

    st.write(
        selected_record["reason"]
    )


    st.warning(
        "A potential anomaly is not proof of fraud. "
        "Context notes are not proof of causation. "
        "Human review is required."
    )


# ==================================================
# Conversational assistant
# ==================================================

st.divider()

st.subheader("Ask the Freight Assistant")

st.caption(
    "Ask about shipping costs, route comparisons, "
    "potential anomalies, and validated contextual evidence."
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Example: Why did Delhi-Jaipur costs increase?"
)


if question:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    with st.chat_message("assistant"):

        with st.spinner(
            "Reviewing analytics and eligible evidence..."
        ):

            answer = answer_question(question)

        st.markdown(answer)


    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )