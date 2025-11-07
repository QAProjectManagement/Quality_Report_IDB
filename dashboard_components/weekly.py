from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st


def _find_header_row(df: pd.DataFrame) -> Optional[int]:
    """Return the index of a row that likely contains column headers."""
    for idx, row in df.iterrows():
        values = [
            str(value).strip().lower()
            for value in row.tolist()
            if pd.notna(value) and str(value).strip()
        ]
        if "project name" in values:
            return idx
    return None


def prepare_weekly_dataframe(raw_df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Normalize weekly report sheets so downstream logic can reuse SM layout."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df_week = raw_df.copy()
    df_week = df_week.dropna(how="all").reset_index(drop=True)
    df_week.columns = df_week.columns.map(lambda col: str(col).strip())

    if "Project Name" not in df_week.columns:
        header_idx = _find_header_row(df_week)
        if header_idx is not None:
            header_values = df_week.iloc[header_idx].tolist()
            new_columns: List[str] = []
            for position, value in enumerate(header_values):
                if pd.isna(value):
                    new_columns.append(f"Column_{position}")
                else:
                    text = str(value).strip()
                    new_columns.append(text if text else f"Column_{position}")

            df_week = df_week.iloc[header_idx + 1 :].reset_index(drop=True)
            df_week.columns = new_columns
            df_week.columns = df_week.columns.map(lambda col: str(col).strip())

            drop_candidates = [
                col
                for col in df_week.columns
                if col.lower().startswith("column_") and df_week[col].isna().all()
            ]
            if drop_candidates:
                df_week = df_week.drop(columns=drop_candidates)

    rename_map = {
        "Total Workload (Planned)": "Total Workload",
        "Completed Workload (Actual)": "Completed Workload",
        "Defect Found": "Defect",
        "Defect Fixed": "Fixed defect",
        "Issue/ Risk/ Note": "Comment",
    }
    df_week = df_week.rename(columns=rename_map)

    unnamed_cols = [c for c in df_week.columns if str(c).lower().startswith("unnamed")]
    df_week = df_week.drop(columns=unnamed_cols, errors="ignore")

    if "Project Name" not in df_week.columns:
        return pd.DataFrame()

    df_week = df_week[df_week["Project Name"].notna()]
    df_week = df_week[
        df_week["Project Name"].astype(str).str.strip().str.lower() != "project name"
    ]
    df_week["Project Name"] = df_week["Project Name"].astype(str).str.strip()
    df_week = df_week[~df_week["Project Name"].str.fullmatch("Total", case=False, na=False)]

    if "Period" in df_week.columns:
        df_week["Month"] = df_week["Period"]
    else:
        df_week["Month"] = sheet_name

    df_week["Weekly Report"] = sheet_name

    if "Comment" in df_week.columns:
        df_week["Comment"] = df_week["Comment"].fillna("")
    else:
        df_week["Comment"] = ""

    for numeric_col in ["Total Workload", "Completed Workload", "Defect", "Fixed defect"]:
        if numeric_col not in df_week.columns:
            df_week[numeric_col] = pd.NA

    return df_week

def get_weekly_workbook(file_path: str) -> Tuple[pd.ExcelFile | None, List[str]]:
    """Return the Excel workbook and filtered weekly sheet names, if available."""
    try:
        weekly_xls = pd.ExcelFile(file_path)
    except FileNotFoundError:
        return None, []
    except Exception:
        return None, []

    sheet_names: List[str] = [
        sheet for sheet in weekly_xls.sheet_names if is_weekly_sheet(sheet)
    ]
    return weekly_xls, sheet_names


def build_weekly_dataset(
    weekly_xls: pd.ExcelFile | None, sheet_names: Iterable[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and normalize all weekly sheets into a single DataFrame."""
    if weekly_xls is None:
        return pd.DataFrame(), []

    frames: List[pd.DataFrame] = []
    for sheet in sheet_names:
        try:
            weekly_raw = pd.read_excel(weekly_xls, sheet_name=sheet, header=None)
        except Exception:
            continue
        weekly_prepared = prepare_weekly_dataframe(weekly_raw, sheet)
        if not weekly_prepared.empty:
            frames.append(weekly_prepared)

    if not frames:
        return pd.DataFrame(), []

    weekly_df_all = pd.concat(frames, ignore_index=True)
    weekly_df_all.columns = weekly_df_all.columns.str.strip().str.replace("*", "", regex=False)

    timeframe_order: List[str] = []
    if "Month" in weekly_df_all.columns:
        for value in weekly_df_all["Month"].dropna().tolist():
            if value not in timeframe_order:
                timeframe_order.append(value)

    if not timeframe_order and "Month" in weekly_df_all.columns:
        timeframe_order = list(weekly_df_all["Month"].dropna().unique())

    if "Month" in weekly_df_all.columns and timeframe_order:
        weekly_df_all["Month"] = pd.Categorical(
            weekly_df_all["Month"], categories=timeframe_order, ordered=True
        )

    return weekly_df_all, timeframe_order


DEFAULT_MIN_WEEKLY_START = datetime.strptime("01Jan2000", "%d%b%Y").date()


def infer_min_weekly_start(
    labels: Iterable[str], *, default: Optional[date] = None
) -> date:
    """Return the earliest parsed weekly start date from available labels."""
    parsed_dates: List[date] = []
    for label in labels:
        parsed = parse_weekly_start(label)
        if parsed is not None:
            parsed_dates.append(parsed)

    if parsed_dates:
        return min(parsed_dates)

    return default or DEFAULT_MIN_WEEKLY_START

DATE_FORMATS = [
    "%d%b%y",
    "%d%b%Y",
    "%d %b %y",
    "%d %b %Y",
    "%d %B %y",
    "%d %B %Y",
]

DATE_PATTERNS = [
    r"\b\d{1,2}[A-Za-z]{3}\d{2,4}\b",
    r"\b\d{1,2}\s+[A-Za-z]{3}\s+\d{2,4}\b",
    r"\b\d{1,2}\s+[A-Za-z]{4,9}\s+\d{2,4}\b",
]


def parse_weekly_start(label: Any) -> Optional[datetime.date]:
    if not isinstance(label, str):
        return None

    cleaned = re.sub(r"\(.*?\)", "", label)
    cleaned = cleaned.replace("_", " ").replace("-", " ")

    candidates: List[str] = []
    for pattern in DATE_PATTERNS:
        candidates.extend(re.findall(pattern, cleaned))

    if not candidates:
        parts = cleaned.split()
        if len(parts) >= 3:
            candidates.append(" ".join(parts[:3]))

    for candidate in candidates:
        candidate = candidate.strip()
        for fmt in DATE_FORMATS:
            try:
                parsed = datetime.strptime(candidate, fmt)
                return parsed.date()
            except ValueError:
                continue

    return None


def is_weekly_sheet(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    if "weekly" in name.lower():
        return True
    return parse_weekly_start(name) is not None


def _sort_weekly_labels(labels: Iterable[str]) -> List[str]:
    pairs: List[Tuple[datetime.date, str]] = []
    labels_list = list(labels)
    for label in labels_list:
        parsed = parse_weekly_start(label)
        if parsed is None:
            continue
        pairs.append((parsed, label))
    if pairs:
        pairs.sort(key=lambda item: item[0], reverse=True)
        return [label for _, label in pairs]
    return labels_list


def load_component(context: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare Weekly dashboard data and capture sidebar selections."""
    weekly_file_path: str = context["weekly_file_path"]
    bulan_order: List[str] = list(context.get("bulan_order", []))

    weekly_xls, weekly_sheet_names = get_weekly_workbook(weekly_file_path)
    min_weekly_start = infer_min_weekly_start(weekly_sheet_names)
    weekly_df_all, timeframe_order = build_weekly_dataset(weekly_xls, weekly_sheet_names)

    df = weekly_df_all.copy()
    selected_project: Optional[str] = None
    selected_weekly: Optional[str] = None
    selected_sprint: Optional[str] = None

    if not df.empty and "Weekly Report" in df.columns:
        mask = df["Weekly Report"].apply(
            lambda value: (
                (parsed := parse_weekly_start(value)) is not None
                and parsed >= min_weekly_start
            )
        )
        df = df[mask].copy()
        weekly_df_all = weekly_df_all[weekly_df_all["Weekly Report"].isin(df["Weekly Report"].unique())].copy()
        if timeframe_order:
            valid_months = df["Month"].dropna().unique().tolist() if "Month" in df.columns else []
            timeframe_order = [month for month in timeframe_order if month in valid_months]
        if weekly_sheet_names:
            filtered_reports = _sort_weekly_labels(df["Weekly Report"].dropna().unique().tolist())
            weekly_sheet_names = [name for name in filtered_reports if name in weekly_sheet_names]

    if df.empty:
        st.warning("Weekly data is not available.")
    else:
        if "Project Name" in df.columns:
            project_options = sorted(df["Project Name"].dropna().unique().tolist())
            if project_options:
                selected_project = st.sidebar.selectbox(
                    "Select Project", ["All Projects"] + project_options
                )
                if selected_project != "All Projects":
                    df = df[df["Project Name"] == selected_project]

        if weekly_sheet_names:
            project_weeklies: List[str] = []
            if not df.empty and "Weekly Report" in df.columns:
                project_weeklies = _sort_weekly_labels(
                    df["Weekly Report"].dropna().unique().tolist()
                )
            available_weeklies = project_weeklies or _sort_weekly_labels(weekly_sheet_names)
            weekly_options = (
                ["All Weekly Reports"] + available_weeklies if available_weeklies else ["All Weekly Reports"]
            )
            default_weekly_index = 1 if available_weeklies else 0
            selected_weekly = st.sidebar.selectbox(
                "Select Weekly Report", weekly_options, index=default_weekly_index
            )
            if selected_weekly != weekly_options[0]:
                df = df[df.get("Weekly Report") == selected_weekly]

        if "Sprint" in df.columns:
            sprint_options = sorted(df["Sprint"].dropna().unique().tolist())
            if sprint_options:
                selected_sprint = st.sidebar.selectbox(
                    "Select Sprint", ["All Sprints"] + sprint_options
                )
                if selected_sprint != "All Sprints":
                    df = df[df["Sprint"] == selected_sprint]

    download_info = {
        "label": "💾 Download Weekly Report",
        "source": weekly_file_path,
        "name": Path(weekly_file_path).name,
    }

    return {
        "df": df,
        "df_all": weekly_df_all,
        "bulan_order": timeframe_order if timeframe_order else bulan_order,
        "download": download_info,
        "filters": {
            "selected_project": selected_project,
            "selected_month": "All Months",
            "selected_weekly": selected_weekly,
            "selected_sprint": selected_sprint,
        },
        "is_weekly_view": True,
        "x_axis_col": "Month",
        "project_type": "Weekly",
    }
