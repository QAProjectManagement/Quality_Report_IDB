"""Shared helpers for the IDB quality dashboards."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_elements import elements, mui, nivo

MONTH_ORDER: Tuple[str, ...] = (
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
)

MONTH_ABBREVIATIONS: Dict[str, str] = {
    "January": "Jan",
    "February": "Feb",
    "March": "Mar",
    "April": "Apr",
    "May": "May",
    "June": "Jun",
    "July": "Jul",
    "August": "Aug",
    "September": "Sep",
    "October": "Oct",
    "November": "Nov",
    "December": "Dec",
}

NUMERIC_COLUMNS: Tuple[str, ...] = (
    "Total Workload",
    "Completed Workload",
    "Defect",
    "Fixed defect",
)

QUALITY_SCORE_MAP = {
    "✅ Standard": 3,
    "⚠️ Issue": 2,
    "❌ Risk": 1,
    "❓ No Tasks": 0,
}

NIVO_COLOR_MAP = {
    "✅ Standard": "#4CAF50",
    "⚠️ Issue": "#E0BC00",
    "❌ Risk": "#F44336",
    "❓ No Tasks": "#808080",
}

TREND_COLOR_MAP = {
    "✅ Standard": "#4CAF50",
    "⚠️ Issue": "#E0BC00",
    "❌ Risk": "#F44336",
}

PADDING_STYLE = "8px 12px"
HR_STYLE = "<hr style='border: 1px solid #808080;'>"

DATAFRAME_HEADER_HEIGHT = 38
DATAFRAME_ROW_HEIGHT = 35
DATAFRAME_BASE_PADDING = 16
DATAFRAME_MIN_ROWS = 3
DATAFRAME_MAX_HEIGHT = 900


def calculate_dataframe_height(
    df: pd.DataFrame | None,
    *,
    row_height: int = DATAFRAME_ROW_HEIGHT,
    header_height: int = DATAFRAME_HEADER_HEIGHT,
    base_padding: int = DATAFRAME_BASE_PADDING,
    min_rows: int = DATAFRAME_MIN_ROWS,
    max_rows: int | None = None,
    max_height: int | None = DATAFRAME_MAX_HEIGHT,
) -> int:
    """Estimate a reasonable height for Streamlit dataframes based on row count."""
    if row_height <= 0:
        row_height = DATAFRAME_ROW_HEIGHT
    if header_height <= 0:
        header_height = DATAFRAME_HEADER_HEIGHT
    if base_padding < 0:
        base_padding = DATAFRAME_BASE_PADDING
    if min_rows < 1:
        min_rows = DATAFRAME_MIN_ROWS

    row_count = 0 if df is None or df.empty else len(df)
    if max_rows is not None:
        row_count = min(row_count, max_rows)

    effective_rows = max(row_count, min_rows)
    height = header_height + base_padding + (effective_rows * row_height)

    if max_height is not None:
        height = min(height, max_height)

    return int(height)


def render_dataframe(
    df: pd.DataFrame,
    *,
    hide_index: bool = False,
    column_config: Dict[str, Any] | None = None,
    row_height: int = DATAFRAME_ROW_HEIGHT,
    header_height: int = DATAFRAME_HEADER_HEIGHT,
    base_padding: int = DATAFRAME_BASE_PADDING,
    min_rows: int = DATAFRAME_MIN_ROWS,
    max_rows: int | None = None,
    max_height: int | None = DATAFRAME_MAX_HEIGHT,
    **st_kwargs: Any,
):
    """Render a dataframe with an auto-calculated height to reduce excessive scrolling."""
    height = calculate_dataframe_height(
        df,
        row_height=row_height,
        header_height=header_height,
        base_padding=base_padding,
        min_rows=min_rows,
        max_rows=max_rows,
        max_height=max_height,
    )

    return st.dataframe(
        df,
        use_container_width=True,
        height=height,
        hide_index=hide_index,
        column_config=column_config,
        **st_kwargs,
    )


def render_divider() -> None:
    """Render a horizontal divider to keep the layout consistent."""
    st.markdown(HR_STYLE, unsafe_allow_html=True)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and remove literal asterisks from column names."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace("*", "", regex=False)
    return df


def apply_month_order(df: pd.DataFrame, month_order: Iterable[str]) -> pd.DataFrame:
    """Apply month ordering when available."""
    if df is None or df.empty:
        return df
    df = df.copy()
    months = list(month_order)
    if "Month" in df.columns and months:
        df["Month"] = pd.Categorical(df["Month"], categories=months, ordered=True)
    return df


def convert_numeric_columns(df: pd.DataFrame, numeric_columns: Iterable[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    """Coerce numeric columns to floats, ignoring conversion errors."""
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def calculate_ffr(completed: float, total: float) -> float:
    if total == 0 or pd.isna(total) or pd.isna(completed):
        return 0.0
    return round((completed / total) * 100, 2)


def calculate_dcr(fixed: float, total_defect: float) -> float:
    if total_defect == 0 or pd.isna(fixed) or pd.isna(total_defect):
        return 0.0
    return round((fixed / total_defect) * 100, 2)


def evaluate_productivity(rate: float, total_workload: float, completed_workload: float) -> str:
    if (total_workload or 0) == 0 and (completed_workload or 0) == 0:
        return "❓ No Tasks"
    if rate <= 70:
        return "❌ Risk"
    if 70 < rate < 80:
        return "⚠️ Issue"
    return "✅ Standard"


def evaluate_quality(density: float, correction_rate: float, defect_count: float) -> str:
    if pd.isna(density) or pd.isna(correction_rate):
        return "❓ No Tasks"
    if (defect_count or 0) == 0:
        return "✅ Standard"
    if correction_rate == 0:
        return "❌ Risk"
    if density >= 3 or correction_rate <= 70:
        return "❌ Risk"
    if 2 <= density < 3 or 70 < correction_rate <= 80:
        return "⚠️ Issue"
    return "✅ Standard"


def evaluate_quality_with_check(row: pd.Series) -> str:
    if row.get("Productivity Evaluation") == "❓ No Tasks":
        return "❓ No Tasks"
    return evaluate_quality(row.get("Defect Density"), row.get("Defect Correction Rate"), row.get("Defect"))


def add_evaluation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add productivity/quality metrics and evaluation columns."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df["Productivity Rate"] = df.apply(
        lambda row: calculate_ffr(row.get("Completed Workload"), row.get("Total Workload")),
        axis=1,
    )
    df["Defect Correction Rate"] = df.apply(
        lambda row: calculate_dcr(row.get("Fixed defect"), row.get("Defect")),
        axis=1,
    )
    df["Defect Density"] = df.apply(
        lambda row: (row.get("Defect", 0) / row.get("Total Workload"))
        if row.get("Total Workload") and row.get("Total Workload") > 0
        else 0,
        axis=1,
    )
    df["Productivity Evaluation"] = df.apply(
        lambda row: evaluate_productivity(
            row.get("Productivity Rate"),
            row.get("Total Workload"),
            row.get("Completed Workload"),
        ),
        axis=1,
    )
    df["Quality Evaluation"] = df.apply(evaluate_quality_with_check, axis=1)
    return df


def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df.get("Project Name").notna() & df.get("Total Workload").notna()].copy()


def summarize_quality(avg_quality_score: float) -> str:
    if pd.isna(avg_quality_score):
        return "N/A"
    if avg_quality_score >= 2.5:
        return "Standard"
    if avg_quality_score >= 1.5:
        return "Issue"
    if avg_quality_score > 0:
        return "Risk"
    return "No Tasks"


def build_summary_table(df_valid: pd.DataFrame, project_type: str) -> pd.DataFrame:
    summary_cols: List[str] = [
        "Project Name",
        "IDB Team",
        "Project Leader or PIC",
        "Month",
        "Sprint",
        "No of Employee",
        "Total Workload",
        "Completed Workload",
        "Defect",
        "Fixed defect",
        "Productivity Rate",
        "Defect Density",
        "Defect Correction Rate",
    ]

    df_summary = df_valid.copy()
    if df_summary.empty:
        return df_summary

    if project_type == "SI" and "Sprint Schedule" in df_summary.columns:
        insert_at = summary_cols.index("Sprint") + 1
        summary_cols.insert(insert_at, "Sprint Schedule")

    available_cols = [col for col in summary_cols if col in df_summary.columns]

    if project_type == "SI" and "Month" in available_cols:
        available_cols.remove("Month")

    df_summary = df_summary[available_cols]

    if project_type.lower() == "weekly" and "Month" in df_summary.columns:
        df_summary = df_summary.rename(columns={"Month": "Period"})

    if "Productivity Rate" in df_summary.columns:
        df_summary["Productivity Rate"] = df_summary["Productivity Rate"].apply(
            lambda value: f"{int(round(value))}%" if pd.notna(value) else ""
        )
    if "Defect Correction Rate" in df_summary.columns:
        df_summary["Defect Correction Rate"] = df_summary["Defect Correction Rate"].apply(
            lambda value: f"{int(round(value))}%" if pd.notna(value) else ""
        )
    if "Defect Density" in df_summary.columns:
        df_summary["Defect Density"] = df_summary["Defect Density"].round(2)

    df_summary.index = range(1, len(df_summary) + 1)
    return df_summary


def build_evaluation_table(df_valid: pd.DataFrame, project_type: str) -> pd.DataFrame:
    eval_cols: List[str] = [
        "Project Name",
        "Month",
        "Sprint",
        "Total Workload",
        "Productivity Evaluation",
        "Quality Evaluation",
        "Comment",
        "Action Plan",
    ]

    df_eval = df_valid.copy()
    if df_eval.empty:
        return df_eval

    if project_type == "SI" and "Sprint Schedule" in df_eval.columns:
        insert_at = eval_cols.index("Sprint") + 1
        eval_cols.insert(insert_at, "Sprint Schedule")

    if project_type == "SI" and "Month" in eval_cols:
        eval_cols.remove("Month")

    available_eval_cols = [col for col in eval_cols if col in df_eval.columns]
    df_eval = df_eval[available_eval_cols]

    if project_type.lower() == "weekly" and "Month" in df_eval.columns:
        df_eval = df_eval.rename(columns={"Month": "Period"})

    df_eval.index = range(1, len(df_eval) + 1)
    return df_eval


def render_global_metrics(total_workload: float, completed_workload: float, ffr: float, avg_label: str) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Workload", f"{int(total_workload)}")
    col2.metric("Completed Workload", f"{int(completed_workload)}")
    col3.metric("Productivity Rate", f"{ffr:.1f}%" if total_workload > 0 else "N/A")
    col4.metric("Avg. Quality Evaluation", avg_label)


def render_donut_charts(df: pd.DataFrame) -> None:
    df = df.copy()
    df = df[df.get("Project Name").notna() & df.get("Total Workload").notna()]
    col1, col2 = st.columns(2)

    with col1:
        with elements("nivo_function_productivity"):
            with mui.Box(sx={"height": 500}):
                mui.Typography(
                    "Productivity Overview",
                    variant="h6",
                    sx={"mb": 2, "fontWeight": "bold", "color": "#333"},
                )
                ffe_counts = df["Productivity Evaluation"].value_counts().reset_index()
                ffe_counts.columns = ["id", "value"]
                nivo_data = [
                    {
                        "id": row["id"],
                        "label": row["id"],
                        "value": int(row["value"]),
                        "color": NIVO_COLOR_MAP.get(row["id"], "#888888"),
                    }
                    for _, row in ffe_counts.iterrows()
                ]
                nivo.Pie(
                    data=nivo_data,
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

    with col2:
        with elements("nivo_quality_evaluation"):
            with mui.Box(sx={"height": 500}):
                mui.Typography(
                    "Quality Overview",
                    variant="h6",
                    sx={"mb": 2, "fontWeight": "bold", "color": "#333"},
                )
                quality_counts = df["Quality Evaluation"].value_counts().reset_index()
                quality_counts.columns = ["id", "value"]
                nivo_data_quality = [
                    {
                        "id": row["id"],
                        "label": row["id"],
                        "value": int(row["value"]),
                        "color": NIVO_COLOR_MAP.get(row["id"], "#888888"),
                    }
                    for _, row in quality_counts.iterrows()
                ]
                nivo.Pie(
                    data=nivo_data_quality,
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


def prepare_percentage_data(
    df: pd.DataFrame,
    col_label: str,
    x_col: str = "Month",
    month_order: Iterable[str] = MONTH_ORDER,
) -> pd.DataFrame:
    if df is None or df.empty or col_label not in df.columns or x_col not in df.columns:
        return pd.DataFrame(columns=[x_col, *TREND_COLOR_MAP.keys()])

    count_df = df.groupby([x_col, col_label], observed=True).size().reset_index(name="count")
    total_df = df.groupby(x_col, observed=True).size().reset_index(name="total")
    merged_df = pd.merge(count_df, total_df, on=x_col)
    merged_df["percentage"] = (merged_df["count"] / merged_df["total"] * 100).round(2)

    if x_col == "Month":
        valid_x = [month for month in month_order if month in merged_df[x_col].unique()]
    else:
        valid_x = sorted(merged_df[x_col].dropna().unique())

    merged_df = merged_df[merged_df[x_col].isin(valid_x)]
    merged_df = merged_df.sort_values(x_col)

    pivot_df = merged_df.pivot(index=x_col, columns=col_label, values="percentage").fillna(0)

    for label in TREND_COLOR_MAP:
        if label not in pivot_df.columns:
            pivot_df[label] = 0

    pivot_df = pivot_df[list(TREND_COLOR_MAP.keys())]
    if valid_x:
        pivot_df = pivot_df.loc[valid_x]
    pivot_df = pivot_df.applymap(lambda value: 0.01 if value == 0 else value)
    return pivot_df.reset_index()


def render_trend_charts(ffe_pivot: pd.DataFrame, qe_pivot: pd.DataFrame, x_axis_col: str) -> None:
    col1, col2 = st.columns(2)

    with col1:
        fig_ffe = go.Figure()
        for label, color in TREND_COLOR_MAP.items():
            if label in ffe_pivot:
                fig_ffe.add_trace(
                    go.Scatter(
                        x=ffe_pivot[x_axis_col],
                        y=ffe_pivot[label],
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color, width=3),
                        marker=dict(size=8),
                    )
                )
        fig_ffe.update_layout(
            title="Productivity Evaluation",
            plot_bgcolor="white",
            margin=dict(l=40, r=20, t=60, b=40),
            title_x=0.05,
            xaxis=dict(title=x_axis_col, showgrid=False),
            yaxis=dict(title="Percentage %", showgrid=True, gridcolor="lightgray", ticksuffix=" %"),
        )
        if x_axis_col == "Month":
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
        fig_ffe.update_yaxes(title_text="Percentage %", showgrid=True, gridcolor="lightgray", ticksuffix=" %")
        st.plotly_chart(fig_ffe, use_container_width=True)

    with col2:
        fig_qe = go.Figure()
        for label, color in TREND_COLOR_MAP.items():
            if label in qe_pivot:
                fig_qe.add_trace(
                    go.Scatter(
                        x=qe_pivot[x_axis_col],
                        y=qe_pivot[label],
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color, width=3),
                        marker=dict(size=8),
                    )
                )
        fig_qe.update_layout(
            title="Quality Evaluation",
            plot_bgcolor="white",
            margin=dict(l=40, r=20, t=60, b=40),
            title_x=0.05,
            xaxis=dict(title=x_axis_col, showgrid=False),
            yaxis=dict(title="Percentage %", showgrid=True, gridcolor="lightgray", ticksuffix=" %"),
        )
        if x_axis_col == "Month":
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
        fig_qe.update_yaxes(title_text="Percentage %", showgrid=True, gridcolor="lightgray", ticksuffix=" %")
        st.plotly_chart(fig_qe, use_container_width=True)
