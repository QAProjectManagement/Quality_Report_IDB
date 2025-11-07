from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from . import weekly


def _load_monthly_frames(xls: pd.ExcelFile, sheet_names: List[str]) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    bulan_sheets = [s for s in sheet_names if isinstance(s, str) and s.lower() != "si"]
    frames: List[pd.DataFrame] = []

    for sheet in bulan_sheets:
        try:
            temp_df = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            continue
        temp_df.columns = temp_df.columns.str.strip()
        temp_df["Month"] = sheet
        frames.append(temp_df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return frames, combined


def load_component(context: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare SM dashboard data and capture sidebar selections."""
    xls: pd.ExcelFile = context["xls"]
    sheet_names: List[str] = list(context["sheet_names"])
    bulan_order: List[str] = list(context["bulan_order"])
    file_path: str = context["file_path"]
    weekly_file_path: str = context["weekly_file_path"]

    _, df_all_monthly = _load_monthly_frames(xls, sheet_names)
    df = df_all_monthly.copy()

    selected_project: Optional[str] = None
    if "Project Name" in df.columns:
        project_options = sorted(df["Project Name"].dropna().unique().tolist())
        if project_options:
            selected_project = st.sidebar.selectbox(
                "Select Project", ["All Projects"] + project_options
            )
            if selected_project != "All Projects":
                df = df[df["Project Name"] == selected_project]

    selected_month: Optional[str] = None
    month_options = ["All Months"]
    if "Month" in df.columns:
        available_months = [m for m in bulan_order if m in df["Month"].dropna().unique()]
        default_index = len(available_months) - 1 if available_months else 0
        month_options += available_months
        selected_month = st.sidebar.selectbox(
            "Select Month", month_options, index=default_index + 1 if available_months else 0
        )
        if selected_month != "All Months":
            df = df[df["Month"] == selected_month]

    selected_weekly: Optional[str] = None
    is_weekly_view = False
    timeframe_order = list(bulan_order)
    weekly_df_all = pd.DataFrame()

    selected_sprint: Optional[str] = None
    if "Sprint" in df.columns:
        sprint_options = sorted(df["Sprint"].dropna().unique().tolist())
        if sprint_options:
            selected_sprint = st.sidebar.selectbox(
                "Select Sprint", ["All Sprints"] + sprint_options
            )
            if selected_sprint != "All Sprints":
                df = df[df["Sprint"] == selected_sprint]

    download_label = "💾 Download Quality Report"
    download_source = file_path
    download_name = Path(file_path).name

    df_all = df_all_monthly.copy()

    return {
        "df": df,
        "df_all": df_all,
        "bulan_order": timeframe_order if is_weekly_view else bulan_order,
        "download": {
            "label": download_label,
            "source": download_source,
            "name": download_name,
        },
        "filters": {
            "selected_project": selected_project,
            "selected_month": selected_month or "All Months",
            "selected_weekly": selected_weekly,
            "selected_sprint": selected_sprint,
        },
        "is_weekly_view": is_weekly_view,
        "x_axis_col": "Month",
        "project_type": "SM",
    }
