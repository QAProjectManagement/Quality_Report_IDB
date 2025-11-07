"""SM dashboard view."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from dashboards.shared import (
    MONTH_ORDER,
    QUALITY_SCORE_MAP,
    add_evaluation_columns,
    apply_month_order,
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


def _load_monthly_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        st.error(f"Data file not found: {DATA_FILE}")
        return pd.DataFrame()

    try:
        xls = pd.ExcelFile(DATA_FILE)
    except Exception as exc:  # pragma: no cover - streamlit feedback
        st.error(f"Unable to load Excel file: {exc}")
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for sheet in xls.sheet_names:
        if sheet.lower() == "si":
            continue
        try:
            temp_df = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            continue
        temp_df = clean_columns(temp_df)
        temp_df["Month"] = sheet
        frames.append(temp_df)

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    df_all = clean_columns(df_all)
    df_all = apply_month_order(df_all, MONTH_ORDER)
    return df_all


def render_sm_dashboard() -> None:
    df_all = _load_monthly_data()
    if df_all.empty:
        st.warning("SM data is not available.")
        return

    project_options = sorted(df_all["Project Name"].dropna().unique().tolist())
    month_choices = [month for month in MONTH_ORDER if month in df_all["Month"].dropna().unique()]

    st.title("IDB Project Quality Dashboard - SM")
    render_divider()

    st.sidebar.subheader("SM Filters")
    selected_project = st.sidebar.selectbox("Select Project", ["All Projects"] + project_options)

    default_month_index = len(month_choices) if month_choices else 0
    selected_month = st.sidebar.selectbox(
        "Select Month",
        ["All Months"] + month_choices,
        index=default_month_index,
    )

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
    df_trend = df_all.copy()

    if selected_project != "All Projects":
        df_view = df_view[df_view["Project Name"] == selected_project]
        df_trend = df_trend[df_trend["Project Name"] == selected_project]

    if selected_month != "All Months":
        df_view = df_view[df_view["Month"] == selected_month]

    if selected_sprint and selected_sprint != "All Sprints" and "Sprint" in df_view.columns:
        df_view = df_view[df_view["Sprint"] == selected_sprint]
        df_trend = df_trend[df_trend["Sprint"] == selected_sprint]

    df_view = df_view.copy()
    df_trend = df_trend.copy()

    for col in ["Comment", "Action Plan"]:
        if col not in df_view.columns:
            df_view[col] = ""
        if col not in df_trend.columns:
            df_trend[col] = ""

    df_view = convert_numeric_columns(df_view)
    df_view = add_evaluation_columns(df_view)

    df_trend = convert_numeric_columns(df_trend)
    df_trend = add_evaluation_columns(df_trend)

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
    df_summary = build_summary_table(df_valid, "SM")
    render_dataframe(df_summary)

    render_divider()
    st.subheader("Automatic Evaluation")
    df_eval = build_evaluation_table(df_valid, "SM")
    render_dataframe(df_eval)

    render_divider()
    st.markdown("### Monthly Trend Comparison")
    df_trend = apply_month_order(df_trend, month_choices or MONTH_ORDER)
    ffe_pivot = prepare_percentage_data(
        df_trend,
        "Productivity Evaluation",
        x_col="Month",
        month_order=month_choices or MONTH_ORDER,
    )
    qe_pivot = prepare_percentage_data(
        df_trend,
        "Quality Evaluation",
        x_col="Month",
        month_order=month_choices or MONTH_ORDER,
    )

    if ffe_pivot.empty or qe_pivot.empty:
        st.info("Not enough data to display trend charts.")
    else:
        render_trend_charts(ffe_pivot, qe_pivot, x_axis_col="Month")
