"""SI dashboard view."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboards.shared import (
    QUALITY_SCORE_MAP,
    add_evaluation_columns,
    calculate_ffr,
    clean_columns,
    convert_numeric_columns,
    filter_valid_rows,
    prepare_percentage_data,
    render_divider,
    render_donut_charts,
    render_global_metrics,
    render_dataframe,
    render_trend_charts,
    build_evaluation_table,
    build_summary_table,
    summarize_quality,
)

DATA_FILE = Path("data/evidence/data_procesor/qualityReportIDBTest_September.xlsx")


def _load_si_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        st.error(f"Data file not found: {DATA_FILE}")
        return pd.DataFrame()

    try:
        xls = pd.ExcelFile(DATA_FILE)
        df_si = pd.read_excel(xls, sheet_name="SI")
    except Exception as exc:  # pragma: no cover - streamlit feedback
        st.error(f"Unable to load SI sheet: {exc}")
        return pd.DataFrame()

    df_si = clean_columns(df_si)
    df_si["Month"] = "SI"
    return df_si


def render_si_dashboard() -> None:
    df_all = _load_si_data()
    if df_all.empty:
        st.warning("SI data is not available.")
        return

    project_options = sorted(df_all["Project Name"].dropna().unique().tolist())

    st.title("IDB Project Quality Dashboard - SI")
    render_divider()

    st.sidebar.subheader("SI Filters")
    selected_project = st.sidebar.selectbox("Select Project", ["All Projects"] + project_options)

    sprint_options = sorted(df_all["Sprint"].dropna().unique().tolist()) if "Sprint" in df_all.columns else []
    selected_sprint = None
    if sprint_options:
        selected_sprint = st.sidebar.selectbox("Select Sprint", ["All Sprints"] + sprint_options)

    if DATA_FILE.exists():
        with open(DATA_FILE, "rb") as file:
            st.sidebar.download_button(
                label="Download Quality Report",
                data=file,
                file_name=DATA_FILE.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    df_view = df_all.copy()

    if selected_project != "All Projects":
        df_view = df_view[df_view["Project Name"] == selected_project]

    if selected_sprint and selected_sprint != "All Sprints" and "Sprint" in df_view.columns:
        df_view = df_view[df_view["Sprint"] == selected_sprint]

    df_view = df_view.copy()

    for col in ["Comment", "Action Plan"]:
        if col not in df_view.columns:
            df_view[col] = ""

    df_view = convert_numeric_columns(df_view)
    df_view = add_evaluation_columns(df_view)

    df_valid = filter_valid_rows(df_view)

    total_workload = df_valid["Total Workload"].sum()
    completed_workload = df_valid["Completed Workload"].sum()
    ffr = calculate_ffr(completed_workload, total_workload)
    avg_quality_score = df_valid["Quality Evaluation"].map(QUALITY_SCORE_MAP).mean()
    avg_quality_label = summarize_quality(avg_quality_score)

    render_global_metrics(total_workload, completed_workload, ffr, avg_quality_label)
    render_divider()

    render_donut_charts(df_view)
    render_divider()

    st.subheader("Project Summary Data")
    df_summary = build_summary_table(df_valid, "SI")
    render_dataframe(df_summary)

    render_divider()
    st.subheader("Automatic Evaluation")
    df_eval = build_evaluation_table(df_valid, "SI")
    render_dataframe(df_eval)

    render_divider()
    st.markdown("### Trend Comparison")
    x_axis_col = "Sprint" if "Sprint" in df_view.columns else "Month"
    ffe_pivot = prepare_percentage_data(df_view, "Productivity Evaluation", x_col=x_axis_col)
    qe_pivot = prepare_percentage_data(df_view, "Quality Evaluation", x_col=x_axis_col)

    if ffe_pivot.empty or qe_pivot.empty:
        st.info("Not enough data to display trend charts.")
    else:
        render_trend_charts(ffe_pivot, qe_pivot, x_axis_col=x_axis_col)
