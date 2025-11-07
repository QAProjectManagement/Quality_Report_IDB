from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def load_component(context: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare SI dashboard data and capture sidebar selections."""
    xls: pd.ExcelFile = context["xls"]
    bulan_order: List[str] = list(context["bulan_order"])
    file_path: str = context["file_path"]

    try:
        df = pd.read_excel(xls, sheet_name="SI")
    except Exception:
        df = pd.DataFrame()

    df.columns = df.columns.str.strip()
    df["Month"] = "SI"

    df_all = df.copy()

    selected_project: Optional[str] = None
    if "Project Name" in df.columns:
        project_options = sorted(df["Project Name"].dropna().unique().tolist())
        if project_options:
            selected_project = st.sidebar.selectbox(
                "Select Project", ["All Projects"] + project_options
            )
            if selected_project != "All Projects":
                df = df[df["Project Name"] == selected_project]

    selected_sprint: Optional[str] = None
    if "Sprint" in df.columns:
        sprint_options = sorted(df["Sprint"].dropna().unique().tolist())
        if sprint_options:
            selected_sprint = st.sidebar.selectbox(
                "Select Sprint", ["All Sprints"] + sprint_options
            )
            if selected_sprint != "All Sprints":
                df = df[df["Sprint"] == selected_sprint]

    download_info = {
        "label": "💾 Download Quality Report",
        "source": file_path,
        "name": Path(file_path).name,
    }

    x_axis_col = "Sprint" if "Sprint" in df_all.columns else "Month"

    return {
        "df": df,
        "df_all": df_all,
        "bulan_order": bulan_order,
        "download": download_info,
        "filters": {
            "selected_project": selected_project,
            "selected_month": "SI",
            "selected_weekly": None,
            "selected_sprint": selected_sprint,
        },
        "is_weekly_view": False,
        "x_axis_col": x_axis_col,
        "project_type": "SI",
    }

