from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import re
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from auth import show_login
from streamlit_elements import elements, mui, nivo
 
from dashboard_components.si import load_component as load_si_component
from dashboard_components.sm import load_component as load_sm_component
from dashboard_components.weekly import load_component as load_weekly_component
from dashboards.shared import MONTH_ABBREVIATIONS, render_dataframe

st.set_page_config(
    page_title="IDB Quality Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# Column name constants
COL_TOTAL_WORKLOAD = "Total Workload"
COL_COMPLETED_WORKLOAD = "Completed Workload"
COL_FIXED_DEFECT = "Fixed defect"
COL_PROJECT_NAME = "Project Name"
COL_IDB_TEAM = "IDB Team"
COL_PIC = "Project Leader or PIC"
COL_NO_OF_EMPLOYEE = "No of Employee"
COL_SPRINT_SCHEDULE = "Sprint Schedule"
COL_PRODUCTIVITY_RATE = "Productivity Rate"
COL_DEFECT_DENSITY = "Defect Density"
COL_DEFECT_CORRECTION_RATE = "Defect Correction Rate"
COL_QUALITY_EVALUATION = "Quality Evaluation"
COL_PRODUCTIVITY_EVALUATION = "Productivity Evaluation"
COL_QUALITY_EVALUATION_NUMERIC = "Quality Evaluation Numeric"
COL_PERCENTAGE = "Percentage %"

# Evaluation label constants
LABEL_STANDARD = "✅ Standard"
LABEL_ISSUE = "⚠️ Issue"
LABEL_RISK = "❌ Risk"
LABEL_NO_TASKS = "❓ No Tasks"

NUMERIC_COLUMNS = [COL_TOTAL_WORKLOAD, COL_COMPLETED_WORKLOAD, "Defect", COL_FIXED_DEFECT]

NIVO_COLOR_MAP: Dict[str, str] = {
    LABEL_STANDARD: "#4CAF50",
    LABEL_ISSUE: "#E0BC00",
    LABEL_RISK: "#F44336",
    LABEL_NO_TASKS: "#808080",
}

LINE_COLOR_MAP: Dict[str, str] = {
    LABEL_STANDARD: "#4CAF50",
    LABEL_ISSUE: "#E0BC00",
    LABEL_RISK: "#F44336",
}

QUALITY_SCORE_MAP: Dict[str, int] = {
    LABEL_STANDARD: 3,
    LABEL_ISSUE: 2,
    LABEL_RISK: 1,
    LABEL_NO_TASKS: 0,
}

PADDING_STYLE = "8px 12px"
HR_STYLE = "<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1rem 0;'>"

DASHBOARD_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Dark App Background ── */
.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 50%, #0f1117 100%) !important;
    background-attachment: fixed !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13162b 0%, #0e1020 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.2) !important;
}
[data-testid="stSidebar"] * {
    color: #c8d0e8 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #8892b0 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #a5b4fc !important;
    font-weight: 600 !important;
}

/* ── Main Content Area ── */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* ── Page Title ── */
.stApp h1 {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #a5b4fc, #818cf8, #6366f1) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0.5rem !important;
}

/* ── Section Headers (subheader / markdown h3) ── */
.stApp h2 {
    color: #c7d2fe !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    margin-top: 0.5rem !important;
}
.stApp h3 {
    color: #a5b4fc !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    margin-top: 0.2rem !important;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(139, 92, 246, 0.08) 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 14px !important;
    padding: 1.2rem 1.4rem !important;
    backdrop-filter: blur(10px) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2) !important;
}
[data-testid="stMetricLabel"] {
    color: #8892b0 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
}

/* ── DataFrames / Tables ── */
[data-testid="stDataFrame"],
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}

/* ── Selectbox / Input ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(30, 35, 60, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 8px !important;
    color: #c8d0e8 !important;
}

/* ── Plotly Chart Containers ── */
.stPlotlyChart {
    background: rgba(30, 35, 60, 0.5) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Nivo/Elements Container ── */
.element-container iframe {
    border-radius: 12px !important;
}

/* ── Download Button ── */
.stDownloadButton button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}

/* ── Alert / Info Boxes ── */
.stAlert {
    border-radius: 10px !important;
    border-left: 4px solid #6366f1 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.5); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.8); }
</style>
"""


def describe_filters(
    project_type: str,
    selected_project: Optional[str],
    selected_month: Optional[str],
    selected_sprint: Optional[str],
    selected_weekly: Optional[str],
) -> str:
    parts: List[str] = []
    if project_type:
        parts.append(project_type.upper())
    if selected_project and selected_project != "All Projects":
        parts.append(f"project **{selected_project}**")
    if selected_weekly and selected_weekly != "All Weekly Reports":
        parts.append(f"weekly report **{selected_weekly}**")
    if selected_month and selected_month not in ("All Months", "SI"):
        parts.append(f"period **{selected_month}**")
    if selected_sprint and selected_sprint != "All Sprints":
        parts.append(f"sprint **{selected_sprint}**")
    return ", ".join(parts) if parts else "current selection"


def format_percent(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def answer_dashboard_question(
    question: str,
    df_current: pd.DataFrame,
    project_type: str,
    filters: Dict[str, Optional[str]],
) -> str:
    if df_current is None or df_current.empty:
        return "Data untuk filter yang dipilih belum tersedia."

    selected_project = filters.get("selected_project")
    selected_month = filters.get("selected_month")
    selected_sprint = filters.get("selected_sprint")
    selected_weekly = filters.get("selected_weekly")

    context_text = describe_filters(
        project_type, selected_project, selected_month, selected_sprint, selected_weekly
    )

    total_workload = (
        df_current[COL_TOTAL_WORKLOAD].sum() if COL_TOTAL_WORKLOAD in df_current.columns else 0
    )
    completed_workload = (
        df_current[COL_COMPLETED_WORKLOAD].sum()
        if COL_COMPLETED_WORKLOAD in df_current.columns
        else 0
    )
    avg_productivity = (
        df_current[COL_PRODUCTIVITY_RATE].mean()
        if COL_PRODUCTIVITY_RATE in df_current.columns
        else float("nan")
    )
    avg_correction = (
        df_current[COL_DEFECT_CORRECTION_RATE].mean()
        if COL_DEFECT_CORRECTION_RATE in df_current.columns
        else float("nan")
    )

    quality_counts = (
        df_current[COL_QUALITY_EVALUATION].value_counts().to_dict()
        if COL_QUALITY_EVALUATION in df_current.columns
        else {}
    )

    q = question.lower()

    if "total workload" in q or ("total" in q and "workload" in q):
        return (
            f"Total workload pada {context_text} adalah **{total_workload:,.0f}** dengan pekerjaan selesai "
            f"**{completed_workload:,.0f}** unit."
        )

    if "completed workload" in q or ("workload" in q and "selesai" in q):
        return (
            f"Pekerjaan selesai untuk {context_text} mencapai **{completed_workload:,.0f}** "
            f"dari total **{total_workload:,.0f}** unit."
        )

    if "productivity" in q or "ffr" in q:
        return (
            f"Rata-rata productivity rate pada {context_text} adalah {format_percent(avg_productivity)} "
            f"dengan total workload {total_workload:,.0f}."
        )

    if "defect" in q and "correction" in q:
        return (
            f"Rata-rata defect correction rate untuk {context_text} adalah {format_percent(avg_correction)}."
        )

    if "quality" in q:
        if not quality_counts:
            return f"Belum ada data evaluasi kualitas untuk {context_text}."
        details = ", ".join(
            f"{label}: {count}" for label, count in sorted(quality_counts.items(), key=lambda x: (-x[1], x[0]))
        )
        return f"Distribusi evaluasi kualitas pada {context_text}: {details}."

    if "risk" in q:
        risk_rows = df_current[
            df_current.get(COL_QUALITY_EVALUATION, pd.Series(dtype=str)).str.contains("Risk", na=False)
            | df_current.get(COL_PRODUCTIVITY_EVALUATION, pd.Series(dtype=str)).str.contains("Risk", na=False)
        ]
        if risk_rows.empty:
            return f"Tidak ada proyek dengan status risiko pada {context_text}."
        projects = ", ".join(sorted(risk_rows[COL_PROJECT_NAME].astype(str).unique().tolist()))
        return f"Proyek dengan status risiko pada {context_text}: {projects}."

    summary = [
        f"Ringkasan untuk {context_text}:",
        f"- Total workload: **{total_workload:,.0f}** | Completed: **{completed_workload:,.0f}**",
        f"- Rata-rata productivity rate: {format_percent(avg_productivity)}",
        f"- Rata-rata defect correction rate: {format_percent(avg_correction)}",
    ]
    if quality_counts:
        summary.append(
            "- Evaluasi kualitas: "
            + ", ".join(
                f"{label}: {count}" for label, count in sorted(quality_counts.items(), key=lambda x: (-x[1], x[0]))
            )
        )
    if "Defect" in df_current.columns:
        summary.append(f"- Total defect: **{int(df_current['Defect'].sum())}**")
    return "\n".join(summary)
def calculate_ffr(completed: Any, total: Any) -> float:
    if total in (0, None) or pd.isna(total) or pd.isna(completed):
        return 0.0
    return round((completed / total) * 100, 2)


def calculate_dcr(fixed: Any, total_defect: Any) -> float:
    if total_defect in (0, None) or pd.isna(total_defect) or pd.isna(fixed):
        return 0.0
    return round((fixed / total_defect) * 100, 2)


def evaluate_productivity(rate: float, total_workload: Any, completed_workload: Any) -> str:
    if (total_workload in (0, None) or pd.isna(total_workload)) and (
        completed_workload in (0, None) or pd.isna(completed_workload)
    ):
        return LABEL_NO_TASKS
    if rate <= 70:
        return LABEL_RISK
    if 70 < rate < 80:
        return LABEL_ISSUE
    return LABEL_STANDARD


def evaluate_quality(density: Any, correction_rate: Any, defect_count: Any) -> str:
    if pd.isna(density) or pd.isna(correction_rate):
        return LABEL_NO_TASKS
    if defect_count in (0, None) or pd.isna(defect_count):
        return LABEL_STANDARD
    if correction_rate in (0, None) or pd.isna(correction_rate):
        return LABEL_RISK
    if density >= 3 or correction_rate <= 70:
        return LABEL_RISK
    if 2 <= density < 3 or (70 < correction_rate <= 80):
        return LABEL_ISSUE
    return LABEL_STANDARD


def evaluate_quality_with_check(row: pd.Series) -> str:
    productivity_eval = row.get(COL_PRODUCTIVITY_EVALUATION)
    if productivity_eval == LABEL_NO_TASKS:
        return LABEL_NO_TASKS
    return evaluate_quality(
        row.get(COL_DEFECT_DENSITY),
        row.get(COL_DEFECT_CORRECTION_RATE),
        row.get("Defect"),
    )


def sanitize_dataframe(df: Optional[pd.DataFrame], month_order: List[str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    result = df.copy()
    result.columns = result.columns.astype(str).str.strip().str.replace("*", "", regex=False)
    if month_order and "Month" in result.columns:
        result["Month"] = pd.Categorical(result["Month"], categories=month_order, ordered=True)
    return result


def convert_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def add_calculated_columns(df: pd.DataFrame) -> pd.DataFrame:
    calculated_columns = [
        COL_PRODUCTIVITY_RATE,
        COL_DEFECT_CORRECTION_RATE,
        COL_DEFECT_DENSITY,
        COL_PRODUCTIVITY_EVALUATION,
        COL_QUALITY_EVALUATION,
    ]

    if df.empty:
        result = df.copy()
        for col in calculated_columns:
            if col not in result.columns:
                result[col] = []
        return result

    result = df.copy()
    result[COL_PRODUCTIVITY_RATE] = result.apply(
        lambda r: calculate_ffr(r.get(COL_COMPLETED_WORKLOAD), r.get(COL_TOTAL_WORKLOAD)), axis=1
    )
    result[COL_DEFECT_CORRECTION_RATE] = result.apply(
        lambda r: calculate_dcr(r.get(COL_FIXED_DEFECT), r.get("Defect")), axis=1
    )
    result[COL_DEFECT_DENSITY] = result.apply(
        lambda r: (r.get("Defect") / r.get(COL_TOTAL_WORKLOAD))
        if r.get(COL_TOTAL_WORKLOAD) not in (None, 0) and not pd.isna(r.get(COL_TOTAL_WORKLOAD))
        else 0,
        axis=1,
    )
    result[COL_PRODUCTIVITY_EVALUATION] = result.apply(
        lambda r: evaluate_productivity(
            r.get(COL_PRODUCTIVITY_RATE), r.get(COL_TOTAL_WORKLOAD), r.get(COL_COMPLETED_WORKLOAD)
        ),
        axis=1,
    )
    result[COL_QUALITY_EVALUATION] = result.apply(evaluate_quality_with_check, axis=1)
    return result


def render_download_button(info: Dict[str, Optional[str]]) -> None:
    label = info.get("label")
    source = info.get("source")
    file_name = info.get("name")

    if not label or not source or not file_name:
        return

    try:
        with open(source, "rb") as file_handle:
            st.sidebar.download_button(
                label=label,
                data=file_handle,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except FileNotFoundError:
        st.sidebar.warning(f"File not found: {file_name}")
    except Exception:
        st.sidebar.warning("Unable to provide download for the selected report.")


def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    df_cleaned = df.dropna(how="all").copy()
    # Also drop rows where all columns are empty strings
    mask = df_cleaned.astype(str).apply(lambda x: x.str.strip() == '').all(axis=1)
    return df_cleaned[~mask]

def get_valid_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = remove_empty_rows(df)
    if df_clean.empty or not {COL_PROJECT_NAME, COL_TOTAL_WORKLOAD}.issubset(df_clean.columns):
        return df_clean.iloc[0:0].copy()
    return df_clean[df_clean[COL_PROJECT_NAME].notna() & df_clean[COL_TOTAL_WORKLOAD].notna()].copy()


def attach_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if COL_QUALITY_EVALUATION not in result.columns:
        result[COL_QUALITY_EVALUATION_NUMERIC] = pd.NA
    else:
        result[COL_QUALITY_EVALUATION_NUMERIC] = result[COL_QUALITY_EVALUATION].map(QUALITY_SCORE_MAP)
    return result


def score_to_label(avg_quality_score: float) -> str:
    if pd.isna(avg_quality_score):
        return "N/A"
    if avg_quality_score >= 2.5:
        return "Standard"
    if avg_quality_score >= 1.5:
        return "Issue"
    if avg_quality_score > 0:
        return "Risk"
    return "No Tasks"


def render_nivo_pie(chart_id: str, title: str, df_source: pd.DataFrame, column_name: str) -> None:
    if column_name not in df_source.columns or df_source.empty:
        data: List[Dict[str, Any]] = []
    else:
        counts = df_source[column_name].value_counts().reset_index()
        counts.columns = ["id", "value"]
        data = [
            {
                "id": row["id"],
                "label": row["id"],
                "value": int(row["value"]),
                "color": NIVO_COLOR_MAP.get(row["id"], "#888888"),
            }
            for _, row in counts.iterrows()
        ]

    with elements(chart_id):
        with mui.Box(sx={"height": 500}):
            mui.Typography(
                title,
                variant="h6",
                sx={"mb": 2, "fontWeight": "bold", "color": "#333"},
            )
            nivo.Pie(
                data=data,
                margin={"top": 40, "right": 160, "bottom": 100, "left": 110},
                innerRadius=0.5,
                padAngle=0.7,
                cornerRadius=3,
                activeOuterRadiusOffset=8,
                borderWidth=1,
                borderColor={"from": "color", "modifiers": [["darker", 0.8]]},
                arcLinkLabelsSkipAngle=10,
                arcLinkLabelsTextColor="grey",
                arcLinkLabelsColor={"from": "color"},
                arcLabelsSkipAngle=10,
                arcLabelsTextColor={"from": "color", "modifiers": [["darker", 2]]},
                legends=[
                    {
                        "anchor": "bottom",
                        "direction": "row",
                        "translateY": 50,
                        "itemWidth": 100,
                        "itemHeight": 18,
                        "symbolSize": 18,
                        "symbolShape": "circle",
                        "effects": [{"on": "hover", "style": {"itemTextColor": "#000"}}],
                    }
                ],
                theme={
                    "tooltip": {
                        "container": {
                            "background": "#333",
                            "color": "#fff",
                            "fontSize": "14px",
                            "borderRadius": "4px",
                            "padding": PADDING_STYLE,
                        }
                    }
                },
            )


def prepare_ffr_trend_data(
    df: pd.DataFrame,
    x_col: str,
    order: List[str],
) -> pd.DataFrame:
    if df is None or df.empty or COL_TOTAL_WORKLOAD not in df.columns or COL_COMPLETED_WORKLOAD not in df.columns or x_col not in df.columns:
        return pd.DataFrame(columns=[x_col, COL_PRODUCTIVITY_RATE])

    grouped = df.groupby(x_col, observed=True).agg(
        total=(COL_TOTAL_WORKLOAD, "sum"),
        completed=(COL_COMPLETED_WORKLOAD, "sum")
    ).reset_index()

    grouped[COL_PRODUCTIVITY_RATE] = grouped.apply(
        lambda row: (row["completed"] / row["total"] * 100) if row["total"] > 0 else 0, 
        axis=1
    ).round(2)

    if x_col == "Month" and order:
        grouped = grouped[grouped[x_col].isin(order)]
        grouped[x_col] = pd.Categorical(grouped[x_col], categories=order, ordered=True)

    grouped = grouped.sort_values(x_col)

    return grouped


def prepare_percentage_data(
    df: pd.DataFrame,
    col_label: str,
    order: List[str],
    x_col: str = "Month",
    color_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    if (
        df is None
        or df.empty
        or col_label not in df.columns
        or x_col not in df.columns
    ):
        if color_map:
            return pd.DataFrame(columns=[x_col, *color_map.keys()])
        return pd.DataFrame(columns=[x_col])

    count_df = (
        df.groupby([x_col, col_label], observed=True)
        .size()
        .reset_index(name="count")
    )
    total_df = df.groupby(x_col, observed=True).size().reset_index(name="total")

    merged_df = pd.merge(count_df, total_df, on=x_col, how="left", validate="many_to_one")
    merged_df["percentage"] = (merged_df["count"] / merged_df["total"] * 100).round(2)

    if x_col == "Month":
        if order:
            merged_df = merged_df[merged_df[x_col].isin(order)]
            merged_df[x_col] = pd.Categorical(merged_df[x_col], categories=order, ordered=True)
            valid_x = [month for month in order if month in merged_df[x_col].unique()]
        else:
            valid_x = merged_df[x_col].dropna().unique().tolist()
    else:
        valid_x = sorted(merged_df[x_col].dropna().unique())

    if valid_x:
        merged_df = merged_df[merged_df[x_col].isin(valid_x)]

    merged_df = merged_df.sort_values(x_col)

    pivot_df = merged_df.pivot(index=x_col, columns=col_label, values="percentage").fillna(0)

    if color_map:
        pivot_df = pivot_df.reindex(columns=color_map.keys(), fill_value=0)

    pivot_df = pivot_df.replace(0, 0.01).reset_index()

    if color_map:
        return pivot_df[[x_col, *color_map.keys()]]

    remaining_cols = [col for col in pivot_df.columns if col != x_col]
    return pivot_df[[x_col, *remaining_cols]]


def normalize_project_name(name: str) -> str:
    name = str(name).strip().upper()
    name = re.sub(r'\b202[56]\b', '', name)
    name = name.replace(' DX ', ' AX ')
    name = re.sub(r'\s+', ' ', name).strip()
    return name


@st.cache_data
def load_project_mapping() -> Dict[str, Dict[str, str]]:
    path_2025 = r"data/evidence/data_procesor/qualityReportIDBTest_December_2025.xlsx"
    mapping: Dict[str, Dict[str, str]] = {}
    try:
        xls = pd.ExcelFile(path_2025)
        for sheet in xls.sheet_names:
            df_sheet = pd.read_excel(xls, sheet_name=sheet)
            df_sheet.columns = df_sheet.columns.astype(str).str.strip().str.replace("*", "", regex=False)
            if COL_PROJECT_NAME in df_sheet.columns:
                for _, row in df_sheet.iterrows():
                    proj = row.get(COL_PROJECT_NAME)
                    if pd.notna(proj) and str(proj).strip():
                        proj_name = str(proj).strip()
                        norm_key = normalize_project_name(proj_name)
                        if norm_key not in mapping:
                            mapping[norm_key] = {}
                        if COL_IDB_TEAM in df_sheet.columns and pd.notna(row[COL_IDB_TEAM]) and str(row[COL_IDB_TEAM]).strip() not in ("", "None", "nan"):
                            mapping[norm_key][COL_IDB_TEAM] = row[COL_IDB_TEAM]
                        if COL_PIC in df_sheet.columns and pd.notna(row[COL_PIC]) and str(row[COL_PIC]).strip() not in ("", "None", "nan"):
                            mapping[norm_key][COL_PIC] = row[COL_PIC]
    except Exception:
        pass
    return mapping


def apply_project_mapping(df: pd.DataFrame, mapping: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    if df.empty or COL_PROJECT_NAME not in df.columns:
        return df
    df_out = df.copy()
    
    if COL_IDB_TEAM not in df_out.columns:
        df_out[COL_IDB_TEAM] = pd.NA
    df_out[COL_IDB_TEAM] = df_out[COL_IDB_TEAM].astype(object)
    
    if COL_PIC not in df_out.columns:
        df_out[COL_PIC] = pd.NA
    df_out[COL_PIC] = df_out[COL_PIC].astype(object)
        
    for idx, row in df_out.iterrows():
        proj = row.get(COL_PROJECT_NAME)
        if pd.notna(proj):
            proj_name = str(proj).strip()
            norm_key = normalize_project_name(proj_name)
            
            # Additional fallback mappings
            fallback_map = {
                "SF MANUFACTURING QUALITY AX RMS/MIMS OPERATION & DEVELOPMENT TEAM": "SF MANUFACTURE AX TEAM",
                "SF MANUFACTURING QUALITY AX STANDARD REPORT UTILIZATION TEAM": "SF MANUFACTURE AX TEAM",
                "LGUS AWS MSP PROJECT": "SALESFORCE MAINTENANCE & ENHANCEMENT PROJECT"
            }
            if norm_key not in mapping and norm_key in fallback_map:
                norm_key = fallback_map[norm_key]
                
            if norm_key in mapping:
                idb_team = row.get(COL_IDB_TEAM)
                if pd.isna(idb_team) or str(idb_team).strip() in ("", "None", "nan"):
                    if COL_IDB_TEAM in mapping[norm_key]:
                        df_out.at[idx, COL_IDB_TEAM] = mapping[norm_key][COL_IDB_TEAM]
                
                pic = row.get(COL_PIC)
                if pd.isna(pic) or str(pic).strip() in ("", "None", "nan"):
                    if proj_name.upper() == "LGUS AWS MSP PROJECT":
                        df_out.at[idx, COL_PIC] = "Kurniawan"
                    elif COL_PIC in mapping[norm_key]:
                        df_out.at[idx, COL_PIC] = mapping[norm_key][COL_PIC]
    return df_out


def main() -> None:
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        if not show_login():
            st.stop()
        return

    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    st.sidebar.image("lgsm_logo.png", width=180)
    st.sidebar.markdown("### 🗂 Navigation")

    view_options = [
        ("SM", "SM Projects"),
        ("SI", "SI Projects"),
        ("Weekly", "Weekly Reports"),
    ]
    selected_view = st.sidebar.selectbox(
        "Select Dashboard",
        options=view_options,
        format_func=lambda item: item[1],
    )[0]

    selected_year = st.sidebar.selectbox(
        "Select Year",
        options=["2025", "2026"],
        index=1,
    )

    if selected_year == "2025":
        file_path = r"data/evidence/data_procesor/qualityReportIDBTest_December_2025.xlsx"
    else:
        file_path = r"data/evidence/data_procesor/qualityReportIDBTest_May_2026.xlsx"
    
    weekly_file_path = r"data/evidence/data_procesor/Weekly/All Project - Weekly Report.xlsx"

    context = {
        "bulan_order": BASE_MONTH_ORDER,
        "file_path": file_path,
        "weekly_file_path": weekly_file_path,
    }

    if selected_view == "SI":
        payload = load_si_component(context)
    elif selected_view == "Weekly":
        payload = load_weekly_component(context)
    else:
        payload = load_sm_component(context)

    bulan_order = payload.get("bulan_order", BASE_MONTH_ORDER)

    df = sanitize_dataframe(payload.get("df"), bulan_order)
    df = convert_numeric_columns(df, NUMERIC_COLUMNS)
    df = add_calculated_columns(df)

    df_all = sanitize_dataframe(payload.get("df_all"), bulan_order)
    df_all = convert_numeric_columns(df_all, NUMERIC_COLUMNS)

    # Apply 2025 mapping to fill missing IDB Team and PIC
    mapping = load_project_mapping()
    df = apply_project_mapping(df, mapping)
    df_all = apply_project_mapping(df_all, mapping)

    filters = payload.get("filters", {})
    project_type_label = payload.get("project_type", selected_view)
    is_weekly_view = payload.get("is_weekly_view", False)
    x_axis_col = payload.get("x_axis_col", "Month")

    render_download_button(payload.get("download", {}))

    if not df_all.empty:
        selected_project = filters.get("selected_project")
        if (
            selected_project
            and selected_project != "All Projects"
            and COL_PROJECT_NAME in df_all.columns
        ):
            df_all = df_all[df_all[COL_PROJECT_NAME] == selected_project]

        selected_sprint = filters.get("selected_sprint")
        if (
            selected_sprint
            and selected_sprint != "All Sprints"
            and "Sprint" in df_all.columns
        ):
            df_all = df_all[df_all["Sprint"] == selected_sprint]

        if (
            project_type_label != "SI"
            and not is_weekly_view
            and "Month" in df_all.columns
            and bulan_order
        ):
            available_months = [
                month
                for month in bulan_order
                if month in df_all["Month"].dropna().unique()
            ]
            if available_months:
                df_all = df_all[df_all["Month"].isin(available_months)]

    df_all = add_calculated_columns(df_all)

    st.title(f"📊 IDB Project Quality Dashboard - {project_type_label}")
    st.markdown(HR_STYLE, unsafe_allow_html=True)

    df_valid = get_valid_dataframe(df)
    
    if COL_PRODUCTIVITY_EVALUATION in df_valid.columns:
        df_valid = df_valid[df_valid[COL_PRODUCTIVITY_EVALUATION] != LABEL_NO_TASKS]
        
    df_valid = attach_quality_score(df_valid)

    total_workload = (
        df_valid[COL_TOTAL_WORKLOAD].sum() if COL_TOTAL_WORKLOAD in df_valid.columns else 0
    )
    completed_workload = (
        df_valid[COL_COMPLETED_WORKLOAD].sum()
        if COL_COMPLETED_WORKLOAD in df_valid.columns
        else 0
    )
    ffr = calculate_ffr(completed_workload, total_workload)

    avg_quality_score = (
        df_valid[COL_QUALITY_EVALUATION_NUMERIC].mean()
        if COL_QUALITY_EVALUATION_NUMERIC in df_valid.columns
        else float("nan")
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📌 Total Workload", f"{int(total_workload)}")
    col2.metric("✅ Completed Workload", f"{int(completed_workload)}")
    col3.metric("🎯 Productivity Rate", f"{ffr:.1f}%" if total_workload > 0 else "N/A")
    col4.metric("🧪 Avg. Quality Evaluation", score_to_label(avg_quality_score))

    st.markdown(HR_STYLE, unsafe_allow_html=True)

    summary_cols = [
        COL_PROJECT_NAME,
        COL_IDB_TEAM,
        COL_PIC,
        "Month",
        "Sprint",
        COL_NO_OF_EMPLOYEE,
        COL_TOTAL_WORKLOAD,
        COL_COMPLETED_WORKLOAD,
        "Defect",
        COL_FIXED_DEFECT,
        COL_PRODUCTIVITY_RATE,
        COL_DEFECT_DENSITY,
        COL_DEFECT_CORRECTION_RATE,
    ]

    if project_type_label == "SI" and COL_SPRINT_SCHEDULE in df_valid.columns:
        summary_cols.insert(4, COL_SPRINT_SCHEDULE)

    df_summary = df_valid.copy()
    if COL_PRODUCTIVITY_RATE in df_summary.columns:
        df_summary[COL_PRODUCTIVITY_RATE] = df_summary[COL_PRODUCTIVITY_RATE].apply(
            lambda x: f"{int(round(x))}%" if pd.notna(x) else ""
        )
    if COL_DEFECT_CORRECTION_RATE in df_summary.columns:
        df_summary[COL_DEFECT_CORRECTION_RATE] = df_summary[COL_DEFECT_CORRECTION_RATE].apply(
            lambda x: f"{int(round(x))}%" if pd.notna(x) else ""
        )
    if COL_DEFECT_DENSITY in df_summary.columns:
        df_summary[COL_DEFECT_DENSITY] = df_summary[COL_DEFECT_DENSITY].round(2)

    available_cols = [col for col in summary_cols if col in df_summary.columns]
    if project_type_label == "SI" and "Month" in available_cols:
        available_cols.remove("Month")
        
    if COL_NO_OF_EMPLOYEE in available_cols:
        is_empty = df_summary[COL_NO_OF_EMPLOYEE].isna() | df_summary[COL_NO_OF_EMPLOYEE].astype(str).str.strip().str.lower().isin(["", "none", "nan", "null"])
        if is_empty.all():
            available_cols.remove(COL_NO_OF_EMPLOYEE)

    df_summary = df_summary[available_cols]
    df_summary.index = range(1, len(df_summary) + 1)

    col_pie_1, col_pie_2 = st.columns(2)
    with col_pie_1:
        render_nivo_pie("nivo_function_productivity", "🎯 Productivity", df_valid, COL_PRODUCTIVITY_EVALUATION)
    with col_pie_2:
        render_nivo_pie("nivo_quality_evaluation", "🧪 Quality", df_valid, COL_QUALITY_EVALUATION)

    st.markdown(HR_STYLE, unsafe_allow_html=True)
    st.subheader("📄 Project Summary Data")
    summary_df_kwargs = {"min_rows": 1}
    if project_type_label == "Weekly":
        summary_df_kwargs["max_height"] = None
    render_dataframe(df_summary, **summary_df_kwargs)

    st.markdown(HR_STYLE, unsafe_allow_html=True)
    st.subheader("🧠 Automatic Evaluation")

    eval_cols = [
        COL_PROJECT_NAME,
        "Month",
        "Sprint",
        COL_TOTAL_WORKLOAD,
        COL_PRODUCTIVITY_EVALUATION,
        COL_QUALITY_EVALUATION,
        "Comment",
        "Action Plan",
    ]

    if project_type_label == "SI" and COL_SPRINT_SCHEDULE in df_valid.columns:
        eval_cols.insert(3, COL_SPRINT_SCHEDULE)

    if project_type_label == "SI" and "Month" in eval_cols:
        eval_cols.remove("Month")

    df_eval = df_valid.copy()
    df_eval = df_eval[[col for col in eval_cols if col in df_eval.columns]]
    df_eval.index = range(1, len(df_eval) + 1)
    eval_df_kwargs = {"min_rows": 1}
    if project_type_label == "Weekly":
        eval_df_kwargs["max_height"] = None
    render_dataframe(df_eval, **eval_df_kwargs)

    if project_type_label != "Weekly":
        st.markdown(HR_STYLE, unsafe_allow_html=True)
        trend_heading = "### 📈 Monthly Trend Comparison"
        st.markdown(trend_heading)

        # Filter out "No Tasks" rows, sama seperti df_valid yang dipakai di metric atas
        df_all_trend = df_all.copy()
        if COL_PRODUCTIVITY_EVALUATION in df_all_trend.columns:
            df_all_trend = df_all_trend[df_all_trend[COL_PRODUCTIVITY_EVALUATION] != LABEL_NO_TASKS]
        if COL_QUALITY_EVALUATION in df_all_trend.columns:
            df_all_trend = df_all_trend[df_all_trend[COL_QUALITY_EVALUATION] != LABEL_NO_TASKS]

        order_for_trend = bulan_order if x_axis_col == "Month" else []
        ffe_pivot = prepare_percentage_data(
            df_all_trend, COL_PRODUCTIVITY_EVALUATION, order_for_trend, x_axis_col, LINE_COLOR_MAP
        )
        qe_pivot = prepare_percentage_data(
            df_all_trend, COL_QUALITY_EVALUATION, order_for_trend, x_axis_col, LINE_COLOR_MAP
        )

        trend_col1, trend_col2 = st.columns(2)

        with trend_col1:
            fig_ffe = go.Figure()
            for label, color in LINE_COLOR_MAP.items():
                fig_ffe.add_trace(
                    go.Scatter(
                        x=ffe_pivot[x_axis_col] if x_axis_col in ffe_pivot.columns else [],
                        y=ffe_pivot[label] if label in ffe_pivot.columns else [],
                        mode="lines+markers",
                        name=label,
                        line={"color": color, "width": 3},
                        marker={"size": 8},
                    )
                )
            fig_ffe.update_layout(
                title="🎯 Productivity Evaluation",
                plot_bgcolor="white",
                margin={"l": 40, "r": 20, "t": 60, "b": 40},
                title_x=0.05,
                xaxis={"title": x_axis_col, "showgrid": False},
                yaxis={
                    "title": COL_PERCENTAGE,
                    "showgrid": True,
                    "gridcolor": "lightgray",
                    "ticksuffix": " %",
                },
            )
            if x_axis_col == "Month" and x_axis_col in ffe_pivot.columns:
                tick_vals = ffe_pivot[x_axis_col].tolist()
                tick_text = [MONTH_ABBREVIATIONS.get(str(val), str(val)) for val in tick_vals]
                fig_ffe.update_xaxes(
                    showgrid=False,
                    tickangle=0,
                    tickmode="array",
                    tickvals=tick_vals,
                    ticktext=tick_text,
                )
            else:
                fig_ffe.update_xaxes(showgrid=False, tickangle=0)
            fig_ffe.update_yaxes(
                title_text=COL_PERCENTAGE,
                showgrid=True,
                gridcolor="lightgray",
                ticksuffix=" %",
            )
            st.plotly_chart(fig_ffe, use_container_width=True)

        with trend_col2:
            fig_qe = go.Figure()
            for label, color in LINE_COLOR_MAP.items():
                fig_qe.add_trace(
                    go.Scatter(
                        x=qe_pivot[x_axis_col] if x_axis_col in qe_pivot.columns else [],
                        y=qe_pivot[label] if label in qe_pivot.columns else [],
                        mode="lines+markers",
                        name=label,
                        line={"color": color, "width": 3},
                        marker={"size": 8},
                    )
                )
            fig_qe.update_layout(
                title="🧪 Quality Evaluation per Month",
                plot_bgcolor="white",
                margin={"l": 40, "r": 20, "t": 60, "b": 40},
                title_x=0.05,
                xaxis={"title": x_axis_col, "showgrid": False},
                yaxis={
                    "title": COL_PERCENTAGE,
                    "showgrid": True,
                    "gridcolor": "lightgray",
                    "ticksuffix": " %",
                },
            )
            if x_axis_col == "Month" and x_axis_col in qe_pivot.columns:
                tick_vals = qe_pivot[x_axis_col].tolist()
                tick_text = [MONTH_ABBREVIATIONS.get(str(val), str(val)) for val in tick_vals]
                fig_qe.update_xaxes(
                    showgrid=False,
                    tickangle=0,
                    tickmode="array",
                    tickvals=tick_vals,
                    ticktext=tick_text,
                )
            else:
                fig_qe.update_xaxes(showgrid=False, tickangle=0)
            fig_qe.update_yaxes(
                title_text=COL_PERCENTAGE,
                showgrid=True,
                gridcolor="lightgray",
                ticksuffix=" %",
            )
            st.plotly_chart(fig_qe, use_container_width=True)


if __name__ == "__main__":
    main()
