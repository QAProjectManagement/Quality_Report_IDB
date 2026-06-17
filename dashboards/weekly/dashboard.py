"""Weekly dashboard view."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

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
from dashboard_components.weekly import (
    infer_min_weekly_start,
    is_weekly_sheet,
    parse_weekly_start,
    prepare_weekly_dataframe,
)


def _sort_weekly_labels(labels: Iterable[str]) -> List[str]:
    pairs: List[Tuple[datetime.date, str]] = []
    cleaned = [label for label in labels if isinstance(label, str)]
    for label in cleaned:
        parsed = parse_weekly_start(label)
        if parsed is None:
            continue
        pairs.append((parsed, label))
    if pairs:
        pairs.sort(key=lambda item: item[0], reverse=True)
        return [label for _, label in pairs]
    return cleaned

WEEKLY_FILE = Path("data/evidence/data_procesor/Weekly/All Project - Weekly Report.xlsx")


def _load_weekly_data() -> Tuple[pd.DataFrame, List[str], List[str]]:
    if not WEEKLY_FILE.exists():
        st.error(f"Weekly file not found: {WEEKLY_FILE}")
        return pd.DataFrame(), [], []

    try:
        weekly_xls = pd.ExcelFile(WEEKLY_FILE)
    except Exception as exc:  # pragma: no cover - streamlit feedback
        st.error(f"Unable to load weekly workbook: {exc}")
        return pd.DataFrame(), [], []

    weekly_sheet_names = [
        sheet for sheet in weekly_xls.sheet_names if is_weekly_sheet(sheet)
    ]
    weekly_sheet_names = _sort_weekly_labels(weekly_sheet_names)
    min_weekly_start = infer_min_weekly_start(weekly_sheet_names)

    frames: List[pd.DataFrame] = []
    for sheet in weekly_sheet_names:
        try:
            raw = pd.read_excel(weekly_xls, sheet_name=sheet, header=None)
        except Exception:
            continue
        prepared = prepare_weekly_dataframe(raw, sheet)
        if not prepared.empty:
            frames.append(prepared)

    if not frames:
        return pd.DataFrame(), weekly_sheet_names, []

    weekly_df = pd.concat(frames, ignore_index=True)
    weekly_df = clean_columns(weekly_df)

    if not weekly_df.empty and "Weekly Report" in weekly_df.columns:
        weekly_df["__parsed"] = weekly_df["Weekly Report"].apply(parse_weekly_start)
        mask = weekly_df["__parsed"].notna() & (weekly_df["__parsed"] >= min_weekly_start)
        weekly_df = weekly_df[mask].copy()
        valid_reports = weekly_df["Weekly Report"].dropna().unique().tolist()
        weekly_sheet_names = [name for name in weekly_sheet_names if name in valid_reports]
        weekly_df = weekly_df.drop(columns="__parsed", errors="ignore")

    timeframe_order: List[str] = []
    if "Month" in weekly_df.columns:
        for value in weekly_df["Month"].dropna():
            if value not in timeframe_order:
                timeframe_order.append(value)

    weekly_sheet_names = _sort_weekly_labels(weekly_sheet_names)

    return weekly_df, weekly_sheet_names, timeframe_order


def render_weekly_dashboard() -> None:
    df_all, sheet_names, timeframe_order = _load_weekly_data()
    if df_all.empty:
        st.warning("Weekly data is not available.")
        return

    month_order = timeframe_order or [month for month in MONTH_ORDER if month in df_all["Month"].dropna().unique()]
    df_all = apply_month_order(df_all, month_order)

    project_options = sorted(df_all["Project Name"].dropna().unique().tolist())

    st.title("IDB Project Quality Dashboard - Weekly")
    render_divider()

    st.sidebar.subheader("Weekly Filters")
    selected_project = st.sidebar.selectbox("Select Project", ["All Projects"] + project_options)

    project_filtered = (
        df_all if selected_project == "All Projects" else df_all[df_all["Project Name"] == selected_project]
    )
    available_weekly_names: List[str] = []
    if (
        not project_filtered.empty
        and "Weekly Report" in project_filtered.columns
    ):
        available_weekly_names = _sort_weekly_labels(
            project_filtered["Weekly Report"].dropna().unique().tolist()
        )
    else:
        available_weekly_names = sheet_names

    weekly_options = ["All Weekly Reports"] + available_weekly_names if available_weekly_names else ["All Weekly Reports"]
    default_weekly_index = 1 if available_weekly_names else 0
    selected_weekly = st.sidebar.selectbox(
        "Select Weekly Report",
        weekly_options,
        index=default_weekly_index,
    )

    sprint_options = sorted(df_all["Sprint"].dropna().unique().tolist()) if "Sprint" in df_all.columns else []
    selected_sprint = None
    if sprint_options:
        selected_sprint = st.sidebar.selectbox("Select Sprint", ["All Sprints"] + sprint_options)

    if WEEKLY_FILE.exists():
        with open(WEEKLY_FILE, "rb") as file:
            st.sidebar.download_button(
                label="Download Weekly Report",
                data=file,
                file_name=WEEKLY_FILE.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    df_view = df_all.copy()
    df_trend = df_all.copy()

    if selected_weekly != "All Weekly Reports" and "Weekly Report" in df_view.columns:
        df_view = df_view[df_view["Weekly Report"] == selected_weekly]

    if selected_project != "All Projects":
        df_view = df_view[df_view["Project Name"] == selected_project]
        df_trend = df_trend[df_trend["Project Name"] == selected_project]

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
    df_summary = build_summary_table(df_valid, "Weekly")
    if "Month" in df_summary.columns:
        df_summary = df_summary.rename(columns={"Month": "Period"})
    render_dataframe(df_summary, max_height=None)

    render_divider()
    st.subheader("Automatic Evaluation")
    df_eval = build_evaluation_table(df_valid, "Weekly")
    if "Month" in df_eval.columns:
        df_eval = df_eval.rename(columns={"Month": "Period"})
    render_dataframe(df_eval, max_height=None)
