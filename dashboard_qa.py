from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from auth import show_login
from streamlit_elements import elements, mui, nivo
 
from dashboard_components.si import load_component as load_si_component
from dashboard_components.sm import load_component as load_sm_component
from dashboard_components.weekly import load_component as load_weekly_component
from dashboards.shared import MONTH_ABBREVIATIONS, render_dataframe

st.set_page_config(page_title="IDB Quality Dashboard", layout="wide")

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

NUMERIC_COLUMNS = ["Total Workload", "Completed Workload", "Defect", "Fixed defect"]

NIVO_COLOR_MAP: Dict[str, str] = {
    "✅ Standard": "#4CAF50",
    "⚠️ Issue": "#E0BC00",
    "❌ Risk": "#F44336",
    "❓ No Tasks": "#808080",
}

LINE_COLOR_MAP: Dict[str, str] = {
    "✅ Standard": "#4CAF50",
    "⚠️ Issue": "#E0BC00",
    "❌ Risk": "#F44336",
}

QUALITY_SCORE_MAP: Dict[str, int] = {
    "✅ Standard": 3,
    "⚠️ Issue": 2,
    "❌ Risk": 1,
    "❓ No Tasks": 0,
}

PADDING_STYLE = "8px 12px"
HR_STYLE = "<hr style='border: 1px solid #808080;'>"


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
        df_current["Total Workload"].sum() if "Total Workload" in df_current.columns else 0
    )
    completed_workload = (
        df_current["Completed Workload"].sum()
        if "Completed Workload" in df_current.columns
        else 0
    )
    avg_productivity = (
        df_current["Productivity Rate"].mean()
        if "Productivity Rate" in df_current.columns
        else float("nan")
    )
    avg_correction = (
        df_current["Defect Correction Rate"].mean()
        if "Defect Correction Rate" in df_current.columns
        else float("nan")
    )

    quality_counts = (
        df_current["Quality Evaluation"].value_counts().to_dict()
        if "Quality Evaluation" in df_current.columns
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
            df_current.get("Quality Evaluation", pd.Series(dtype=str)).str.contains("Risk", na=False)
            | df_current.get("Productivity Evaluation", pd.Series(dtype=str)).str.contains("Risk", na=False)
        ]
        if risk_rows.empty:
            return f"Tidak ada proyek dengan status risiko pada {context_text}."
        projects = ", ".join(sorted(risk_rows["Project Name"].astype(str).unique().tolist()))
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
        return "❓ No Tasks"
    if rate <= 70:
        return "❌ Risk"
    if 70 < rate < 80:
        return "⚠️ Issue"
    return "✅ Standard"


def evaluate_quality(density: Any, correction_rate: Any, defect_count: Any) -> str:
    if pd.isna(density) or pd.isna(correction_rate):
        return "❓ No Tasks"
    if defect_count in (0, None) or pd.isna(defect_count):
        return "✅ Standard"
    if correction_rate in (0, None) or pd.isna(correction_rate):
        return "❌ Risk"
    if density >= 3 or correction_rate <= 70:
        return "❌ Risk"
    if 2 <= density < 3 or (70 < correction_rate <= 80):
        return "⚠️ Issue"
    return "✅ Standard"


def evaluate_quality_with_check(row: pd.Series) -> str:
    productivity_eval = row.get("Productivity Evaluation")
    if productivity_eval == "❓ No Tasks":
        return "❓ No Tasks"
    return evaluate_quality(
        row.get("Defect Density"),
        row.get("Defect Correction Rate"),
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
        "Productivity Rate",
        "Defect Correction Rate",
        "Defect Density",
        "Productivity Evaluation",
        "Quality Evaluation",
    ]

    if df.empty:
        result = df.copy()
        for col in calculated_columns:
            if col not in result.columns:
                result[col] = []
        return result

    result = df.copy()
    result["Productivity Rate"] = result.apply(
        lambda r: calculate_ffr(r.get("Completed Workload"), r.get("Total Workload")), axis=1
    )
    result["Defect Correction Rate"] = result.apply(
        lambda r: calculate_dcr(r.get("Fixed defect"), r.get("Defect")), axis=1
    )
    result["Defect Density"] = result.apply(
        lambda r: (r.get("Defect") / r.get("Total Workload"))
        if r.get("Total Workload") not in (None, 0) and not pd.isna(r.get("Total Workload"))
        else 0,
        axis=1,
    )
    result["Productivity Evaluation"] = result.apply(
        lambda r: evaluate_productivity(
            r.get("Productivity Rate"), r.get("Total Workload"), r.get("Completed Workload")
        ),
        axis=1,
    )
    result["Quality Evaluation"] = result.apply(evaluate_quality_with_check, axis=1)
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


def get_valid_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"Project Name", "Total Workload"}.issubset(df.columns):
        return df.iloc[0:0].copy()
    return df[df["Project Name"].notna() & df["Total Workload"].notna()].copy()


def attach_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "Quality Evaluation" not in result.columns:
        result["Quality Evaluation Numeric"] = pd.NA
    else:
        result["Quality Evaluation Numeric"] = result["Quality Evaluation"].map(QUALITY_SCORE_MAP)
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

    merged_df = pd.merge(count_df, total_df, on=x_col)
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

    pivot_df = pivot_df.applymap(lambda value: 0.01 if value == 0 else value).reset_index()

    if color_map:
        return pivot_df[[x_col, *color_map.keys()]]

    remaining_cols = [col for col in pivot_df.columns if col != x_col]
    return pivot_df[[x_col, *remaining_cols]]


def main() -> None:
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        if not show_login():
            st.stop()
        return

    st.sidebar.image("lgsm_logo.png", width=200)
    st.sidebar.header("Navigation")

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

    file_path = r"data/evidence/data_procesor/qualityReportIDBTest_November.xlsx"
    weekly_file_path = r"data/evidence/data_procesor/Weekly/All Project - Weekly Report.xlsx"

    xls: Optional[pd.ExcelFile] = None
    sheet_names: List[str] = []

    if selected_view in {"SM", "SI"}:
        try:
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
        except Exception as exc:
            st.error(f"Unable to open main quality report file: {exc}")
            return
    else:
        try:
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
        except Exception:
            xls = None
            sheet_names = []

    context = {
        "xls": xls,
        "sheet_names": sheet_names,
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
            and "Project Name" in df_all.columns
        ):
            df_all = df_all[df_all["Project Name"] == selected_project]

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
    df_valid = attach_quality_score(df_valid)

    total_workload = (
        df_valid["Total Workload"].sum() if "Total Workload" in df_valid.columns else 0
    )
    completed_workload = (
        df_valid["Completed Workload"].sum()
        if "Completed Workload" in df_valid.columns
        else 0
    )
    ffr = calculate_ffr(completed_workload, total_workload)

    avg_quality_score = (
        df_valid["Quality Evaluation Numeric"].mean()
        if "Quality Evaluation Numeric" in df_valid.columns
        else float("nan")
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📌 Total Workload", f"{int(total_workload)}")
    col2.metric("✅ Completed Workload", f"{int(completed_workload)}")
    col3.metric("🎯 Productivity Rate", f"{ffr:.1f}%" if total_workload > 0 else "N/A")
    col4.metric("🧪 Avg. Quality Evaluation", score_to_label(avg_quality_score))

    st.markdown(HR_STYLE, unsafe_allow_html=True)

    summary_cols = [
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

    if project_type_label == "SI" and "Sprint Schedule" in df_valid.columns:
        summary_cols.insert(4, "Sprint Schedule")

    df_summary = df_valid.copy()
    if "Productivity Rate" in df_summary.columns:
        df_summary["Productivity Rate"] = df_summary["Productivity Rate"].apply(
            lambda x: f"{int(round(x))}%" if pd.notna(x) else ""
        )
    if "Defect Correction Rate" in df_summary.columns:
        df_summary["Defect Correction Rate"] = df_summary["Defect Correction Rate"].apply(
            lambda x: f"{int(round(x))}%" if pd.notna(x) else ""
        )
    if "Defect Density" in df_summary.columns:
        df_summary["Defect Density"] = df_summary["Defect Density"].round(2)

    available_cols = [col for col in summary_cols if col in df_summary.columns]
    if project_type_label == "SI" and "Month" in available_cols:
        available_cols.remove("Month")

    df_summary = df_summary[available_cols]
    df_summary.index = range(1, len(df_summary) + 1)

    col_pie_1, col_pie_2 = st.columns(2)
    with col_pie_1:
        render_nivo_pie("nivo_function_productivity", "🎯 Productivity", df_valid, "Productivity Evaluation")
    with col_pie_2:
        render_nivo_pie("nivo_quality_evaluation", "🧪 Quality", df_valid, "Quality Evaluation")

    st.markdown(HR_STYLE, unsafe_allow_html=True)
    st.subheader("📄 Project Summary Data")
    summary_df_kwargs = {}
    if project_type_label == "Weekly":
        summary_df_kwargs["max_height"] = None
    render_dataframe(df_summary, **summary_df_kwargs)

    st.markdown(HR_STYLE, unsafe_allow_html=True)
    st.subheader("🧠 Automatic Evaluation")

    eval_cols = [
        "Project Name",
        "Month",
        "Sprint",
        "Total Workload",
        "Productivity Evaluation",
        "Quality Evaluation",
        "Comment",
        "Action Plan",
    ]

    if project_type_label == "SI" and "Sprint Schedule" in df_valid.columns:
        eval_cols.insert(3, "Sprint Schedule")

    if project_type_label == "SI" and "Month" in eval_cols:
        eval_cols.remove("Month")

    df_eval = df_valid.copy()
    df_eval = df_eval[[col for col in eval_cols if col in df_eval.columns]]
    df_eval.index = range(1, len(df_eval) + 1)
    eval_df_kwargs = {}
    if project_type_label == "Weekly":
        eval_df_kwargs["max_height"] = None
    render_dataframe(df_eval, **eval_df_kwargs)

    if project_type_label != "Weekly":
        st.markdown(HR_STYLE, unsafe_allow_html=True)
        trend_heading = "### 📈 Monthly Trend Comparison"
        st.markdown(trend_heading)

        order_for_trend = bulan_order if x_axis_col == "Month" else []
        ffe_pivot = prepare_percentage_data(
            df_all, "Productivity Evaluation", order_for_trend, x_axis_col, LINE_COLOR_MAP
        )
        qe_pivot = prepare_percentage_data(
            df_all, "Quality Evaluation", order_for_trend, x_axis_col, LINE_COLOR_MAP
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
                        line=dict(color=color, width=3),
                        marker=dict(size=8),
                    )
                )
            fig_ffe.update_layout(
                title="🎯 Productivity Evaluation",
                plot_bgcolor="white",
                margin=dict(l=40, r=20, t=60, b=40),
                title_x=0.05,
                xaxis=dict(title=x_axis_col, showgrid=False),
                yaxis=dict(
                    title="Percentage %",
                    showgrid=True,
                    gridcolor="lightgray",
                    ticksuffix=" %",
                ),
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
                title_text="Percentage %",
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
                        line=dict(color=color, width=3),
                        marker=dict(size=8),
                    )
                )
            fig_qe.update_layout(
                title="🧪 Quality Evaluation per Month",
                plot_bgcolor="white",
                margin=dict(l=40, r=20, t=60, b=40),
                title_x=0.05,
                xaxis=dict(title=x_axis_col, showgrid=False),
                yaxis=dict(
                    title="Percentage %",
                    showgrid=True,
                    gridcolor="lightgray",
                    ticksuffix=" %",
                ),
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
                title_text="Percentage %",
                showgrid=True,
                gridcolor="lightgray",
                ticksuffix=" %",
            )
            st.plotly_chart(fig_qe, use_container_width=True)


if __name__ == "__main__":
    main()
