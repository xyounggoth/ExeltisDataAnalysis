from __future__ import annotations

# Streamlit App - Exeltis Assessment Dashboard
# This app loads the raw Excel assessment file, cleans and transforms the data,
# builds outlet master data, and presents an interactive dashboard with a layout
# inspired by the provided reference.

import re
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# =========================
# Configuration / Constants
# =========================

st.markdown("""
    <h1 style='text-align: center;'>EXELTIS SALES DASHBOARD</h1>
    <p style='text-align: center; color: grey;'>
        Interactive dashboard from raw distributor data to cleaned sales insight
    </p>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Exeltis Sales Dashboard",
    page_icon="📊",
    layout="wide",
)

CITY_MAPPING = {
    "bandung": "Bandung",
    "bekasi": "Bekasi",
    "bogor": "Bogor",
    "bgr": "Bogor",
    "cilegon": "Cilegon",
    "surabaya": "Surabaya",
    "kediri": "Kediri",
}

OUTLET_RULES = {
    "ibu anak": "Rumah Sakit Ibu Anak",
    "kencana": "Apotek Kencana",
    "sehat": "Apotek Sehat",
    "farma": "Apotek Farma",
    "buana": "Apotek Buana",
    "citra": "dr. Citra, Sp.Og",
}

CITY_MASTER = pd.DataFrame(
    {
        "City_Code": ["01", "02", "03", "04", "05", "06"],
        "City": ["Bandung", "Cilegon", "Bekasi", "Bogor", "Surabaya", "Kediri"],
    }
)

SEGMENT_MASTER = pd.DataFrame(
    {
        "Segment_Code": ["01", "02", "03", "99"],
        "Segment": ["Hospital", "Drugstore", "HCP", "Other"],
    }
)

PLOT_BG = "#06255C"
PAPER_BG = "#06255C"
CARD_BG = "#082B6B"
GRID_COLOR = "rgba(255,255,255,0.14)"
TEXT_COLOR = "#FFFFFF"
BAR_COLOR = "#5E8FEA"
LINE_COLOR = "#FF5C4D"
DONUT_COLORS = ["#F28C35", "#5E8FEA", "#9FB7E9", "#D4D9E8", "#FDBB74"]


# =========================
# Styling
# =========================

DASHBOARD_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #031735 0%, #051F4D 100%);
        color: #FFFFFF;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .dashboard-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: #FFFFFF;
        margin-bottom: 0.25rem;
    }
    .dashboard-subtitle {
        text-align: center;
        color: #D7E3FF;
        margin-bottom: 1.2rem;
    }
    .card-wrap {
        background: #082B6B;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 0.8rem 1rem 0.4rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }
    .card-title {
        color: #FFFFFF;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    [data-testid="stMetric"] {
        background: #082B6B;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] {
        color: #CFE0FF;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF;
    }
    .insight-box {
        background: rgba(255,255,255,0.05);
        border-left: 4px solid #F28C35;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        color: #FFFFFF;
    }
</style>
"""


# =========================
# Data Loading
# =========================

def normalize_sheet_name(sheet_name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(sheet_name).lower())


EXPECTED_SHEETS = {
    "Distributor A": {"distributora"},
    "Distributor B": {"distributorb"},
    "SKU Master": {"skumaster"},
}


def match_required_sheet(excel_file: pd.ExcelFile, logical_name: str) -> str:
    normalized_map = {normalize_sheet_name(name): name for name in excel_file.sheet_names}
    for expected in EXPECTED_SHEETS[logical_name]:
        if expected in normalized_map:
            return normalized_map[expected]
    available = ", ".join(excel_file.sheet_names)
    raise ValueError(
        f"Sheet untuk **{logical_name}** tidak ditemukan. "
        f"Pastikan upload file raw assessment. Available sheets: {available}"
    )


def load_data(file_source) -> tuple[pd.DataFrame, pd.DataFrame]:
    excel_file = pd.ExcelFile(file_source)

    dist_a_sheet = match_required_sheet(excel_file, "Distributor A")
    dist_b_sheet = match_required_sheet(excel_file, "Distributor B")
    sku_sheet = match_required_sheet(excel_file, "SKU Master")

    dist_a = pd.read_excel(excel_file, sheet_name=dist_a_sheet)
    dist_b = pd.read_excel(excel_file, sheet_name=dist_b_sheet)
    sku_master = pd.read_excel(excel_file, sheet_name=sku_sheet)

    dist_a["Distributor"] = "Distributor A"
    dist_b["Distributor"] = "Distributor B"
    sales_data = pd.concat([dist_a, dist_b], ignore_index=True)
    return sales_data, sku_master


# =========================
# Data Cleansing
# =========================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.astype(str).str.strip().str.replace(" ", "_", regex=False)
    )
    return cleaned


REQUIRED_SALES_COLUMNS = ["SKU_ID", "SKU_Name", "Qty", "City"]
REQUIRED_SALES_OPTIONAL = ["Outlet_Name", "Customer"]
REQUIRED_MASTER_COLUMNS = ["SKU_ID", "SKU_Name", "Price"]


def validate_columns(sales_df: pd.DataFrame, master_df: pd.DataFrame) -> None:
    missing_sales = [col for col in REQUIRED_SALES_COLUMNS if col not in sales_df.columns]
    if not any(col in sales_df.columns for col in REQUIRED_SALES_OPTIONAL):
        missing_sales.append("Outlet_Name/Customer")

    missing_master = [col for col in REQUIRED_MASTER_COLUMNS if col not in master_df.columns]

    messages = []
    if missing_sales:
        messages.append(f"Sales data missing columns: {', '.join(missing_sales)}")
    if missing_master:
        messages.append(f"SKU master missing columns: {', '.join(missing_master)}")
    if messages:
        raise ValueError(" | ".join(messages))


def standardize_sku(sales_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    df = sales_df.copy()
    master = master_df.copy()

    master["SKU_Name"] = master["SKU_Name"].astype(str).str.strip()
    df["SKU_Name"] = df["SKU_Name"].astype(str).str.strip()

    sku_id_lookup = master.set_index("SKU_Name")["SKU_ID"]
    sku_name_lookup = master.set_index("SKU_ID")["SKU_Name"]

    df["SKU_ID"] = df["SKU_ID"].fillna(df["SKU_Name"].map(sku_id_lookup))
    df["SKU_ID"] = pd.to_numeric(df["SKU_ID"], errors="coerce")
    df["SKU_Name"] = df["SKU_ID"].map(sku_name_lookup).fillna(df["SKU_Name"])
    return df


def clean_quantity(series: pd.Series) -> pd.Series:
    qty = series.astype(str).str.replace(",", ".", regex=False).str.strip()
    return pd.to_numeric(qty, errors="coerce").fillna(0)


def normalize_text(value: str) -> str:
    if pd.isna(value):
        return ""
    value = str(value).lower().strip()
    value = re.sub(r"[.,]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def standardize_outlet_name(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    source_col = "Outlet_Name" if "Outlet_Name" in cleaned.columns else "Customer"
    cleaned["Outlet_Name"] = cleaned[source_col].astype(str)
    cleaned["_outlet_normalized"] = cleaned["Outlet_Name"].apply(normalize_text)

    def apply_mapping(name: str) -> str:
        for keyword, standard_name in OUTLET_RULES.items():
            if keyword in name:
                return standard_name
        return name.title() if name else np.nan

    cleaned["Outlet_Name"] = cleaned["_outlet_normalized"].apply(apply_mapping)
    return cleaned.drop(columns=["_outlet_normalized"])


def clean_city(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["City"] = (
        cleaned["City"]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({"nan": np.nan, "": np.nan})
    )
    cleaned["City"] = cleaned["City"].map(CITY_MAPPING)

    city_by_outlet = (
        cleaned.dropna(subset=["City"])
        .groupby("Outlet_Name")["City"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )
    cleaned["City"] = cleaned["City"].fillna(cleaned["Outlet_Name"].map(city_by_outlet))
    return cleaned


# =========================
# Data Transformation
# =========================

def add_price_and_sales(df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()
    transformed["Price"] = transformed["SKU_ID"].map(master_df.set_index("SKU_ID")["Price"])
    transformed["Total_Sales"] = transformed["Qty"] * transformed["Price"]
    return transformed


def add_segment(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()
    outlet_series = transformed["Outlet_Name"].astype(str).str.lower()

    transformed["Segment"] = np.select(
        [
            outlet_series.str.contains(r"rumah sakit|^rs", na=False),
            outlet_series.str.contains(r"apotek|apt", na=False),
            outlet_series.str.contains(r"\bdr\b|sp\.og|spog|sp\.a|spa|sp\.k|spk", na=False),
        ],
        ["Hospital", "Drugstore", "HCP"],
        default="Other",
    )
    return transformed


def build_outlet_master(df: pd.DataFrame) -> pd.DataFrame:
    outlet_master = df[["Outlet_Name", "Segment", "City"]].drop_duplicates().reset_index(drop=True)
    outlet_master = outlet_master.merge(CITY_MASTER, on="City", how="left")
    outlet_master = outlet_master.merge(SEGMENT_MASTER, on="Segment", how="left")
    outlet_master = outlet_master.sort_values(
        ["City_Code", "Segment_Code", "Outlet_Name"],
        na_position="last",
    )
    outlet_master["Outlet_Seq"] = outlet_master.groupby(["City_Code", "Segment_Code"]).cumcount().add(1)
    outlet_master["Outlet_Seq"] = outlet_master["Outlet_Seq"].astype(str).str.zfill(3)
    outlet_master["Outlet_ID"] = (
        outlet_master["City_Code"].fillna("00")
        + outlet_master["Segment_Code"].fillna("00")
        + outlet_master["Outlet_Seq"]
    )
    return outlet_master[["Outlet_ID", "Outlet_Name", "Segment", "City"]]


def finalize_sales_data(df: pd.DataFrame, outlet_df: pd.DataFrame) -> pd.DataFrame:
    final_df = df.merge(outlet_df, on=["Outlet_Name", "Segment", "City"], how="left")
    final_df = final_df[
        [
            "City",
            "Distributor",
            "Outlet_ID",
            "Outlet_Name",
            "Segment",
            "SKU_ID",
            "SKU_Name",
            "Price",
            "Qty",
            "Total_Sales",
        ]
    ].copy()
    final_df["Outlet_ID"] = final_df["Outlet_ID"].astype(str)
    final_df["SKU_ID"] = pd.to_numeric(final_df["SKU_ID"], errors="coerce").astype("Int64")
    return final_df


# =========================
# Data Summary
# =========================

def build_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    kpi_summary = pd.DataFrame(
        {
            "Metric": ["Total Sales", "Total Qty", "Total Outlet", "Total SKU"],
            "Value": [
                df["Total_Sales"].sum(),
                df["Qty"].sum(),
                df["Outlet_ID"].nunique(),
                df["SKU_ID"].nunique(),
            ],
        }
    )

    sales_by_segment = (
        df.groupby("Segment", as_index=False)["Total_Sales"]
        .sum()
        .sort_values("Total_Sales", ascending=False)
    )
    sales_by_segment["Contribution_%"] = (
        sales_by_segment["Total_Sales"] / sales_by_segment["Total_Sales"].sum() * 100
    ).round(2)

    sales_by_sku = (
        df.groupby(["SKU_ID", "SKU_Name"], as_index=False)
        .agg(Qty=("Qty", "sum"), Total_Sales=("Total_Sales", "sum"))
        .sort_values("Total_Sales", ascending=False)
    )
    sales_by_sku["Contribution_%"] = (
        sales_by_sku["Total_Sales"] / sales_by_sku["Total_Sales"].sum() * 100
    ).round(2)

    sales_by_outlet = (
        df.groupby(["Outlet_ID", "Outlet_Name"], as_index=False)["Total_Sales"]
        .sum()
        .sort_values("Total_Sales", ascending=False)
    )

    sales_by_city = (
        df.groupby("City", as_index=False)["Total_Sales"]
        .sum()
        .sort_values("Total_Sales", ascending=False)
    )

    qty_vs_sales = (
        df.groupby("SKU_Name", as_index=False)
        .agg(Qty=("Qty", "sum"), Total_Sales=("Total_Sales", "sum"))
        .sort_values("Total_Sales", ascending=False)
    )

    return {
        "kpi_summary": kpi_summary,
        "sales_by_segment": sales_by_segment,
        "sales_by_sku": sales_by_sku,
        "sales_by_outlet": sales_by_outlet,
        "sales_by_city": sales_by_city,
        "qty_vs_sales": qty_vs_sales,
    }


def build_brief_analysis(summary: dict[str, pd.DataFrame]) -> list[str]:
    insights = []

    if not summary["sales_by_segment"].empty:
        row = summary["sales_by_segment"].iloc[0]
        insights.append(
            f"Top segment adalah {row['Segment']} dengan kontribusi sales {row['Contribution_%']:.2f}%."
        )

    if not summary["sales_by_city"].empty:
        row = summary["sales_by_city"].iloc[0]
        insights.append(
            f"Kota dengan penjualan tertinggi adalah {row['City']} dengan total sales {row['Total_Sales']:,.0f}."
        )

    if not summary["sales_by_sku"].empty:
        row = summary["sales_by_sku"].iloc[0]
        insights.append(
            f"SKU dengan sales tertinggi adalah {row['SKU_Name']} dengan total sales {row['Total_Sales']:,.0f}."
        )

    if not summary["sales_by_outlet"].empty:
        row = summary["sales_by_outlet"].iloc[0]
        insights.append(
            f"Outlet top contributor adalah {row['Outlet_Name']}."
        )

    return insights


# =========================
# Pipeline
# =========================

@st.cache_data(show_spinner=False)
def process_file(file_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    buffer = BytesIO(file_bytes)
    sales_data, sku_master = load_data(buffer)

    sales_data = standardize_columns(sales_data)
    sku_master = standardize_columns(sku_master)
    validate_columns(sales_data, sku_master)
    sales_data = standardize_sku(sales_data, sku_master)
    sales_data["Qty"] = clean_quantity(sales_data["Qty"])
    sales_data = standardize_outlet_name(sales_data)
    sales_data = clean_city(sales_data)
    sales_data = add_price_and_sales(sales_data, sku_master)
    sales_data = add_segment(sales_data)
    outlet_master = build_outlet_master(sales_data)
    sales_data = finalize_sales_data(sales_data, outlet_master)
    summary = build_summary_tables(sales_data)

    return sales_data, outlet_master, summary


def to_excel_bytes(sales_df: pd.DataFrame, outlet_df: pd.DataFrame, summary: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sales_df.to_excel(writer, sheet_name="Sales_Data", index=False)
        outlet_df.to_excel(writer, sheet_name="Outlet_Master", index=False)
        for sheet_name, data in summary.items():
            safe_name = sheet_name[:31]
            data.to_excel(writer, sheet_name=safe_name, index=False)
    output.seek(0)
    return output.getvalue()


# =========================
# UI Helpers
# =========================

def format_currency(value: float) -> str:
    return f"Rp {value:,.0f}"


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filter Dashboard")
    st.sidebar.caption("Gunakan filter untuk melihat insight per area, segment, dan SKU.")

    city_options = sorted([x for x in df["City"].dropna().unique()])
    distributor_options = sorted([x for x in df["Distributor"].dropna().unique()])
    segment_options = sorted([x for x in df["Segment"].dropna().unique()])
    sku_options = sorted([x for x in df["SKU_Name"].dropna().unique()])

    selected_city = st.sidebar.multiselect("City", city_options)
    selected_distributor = st.sidebar.multiselect("Distributor", distributor_options)
    selected_segment = st.sidebar.multiselect("Segment", segment_options)
    selected_sku = st.sidebar.multiselect("SKU Name", sku_options)

    filtered = df.copy()
    if selected_city:
        filtered = filtered[filtered["City"].isin(selected_city)]
    if selected_distributor:
        filtered = filtered[filtered["Distributor"].isin(selected_distributor)]
    if selected_segment:
        filtered = filtered[filtered["Segment"].isin(selected_segment)]
    if selected_sku:
        filtered = filtered[filtered["SKU_Name"].isin(selected_sku)]

    return filtered


# =========================
# Chart Builders
# =========================

def base_layout(title: str, height: int) -> dict:
    return {
        "title": {"text": title, "font": {"size": 24, "color": TEXT_COLOR}, "x": 0.5},
        "paper_bgcolor": PAPER_BG,
        "plot_bgcolor": PLOT_BG,
        "font": {"color": TEXT_COLOR},
        "margin": {"l": 40, "r": 40, "t": 60, "b": 40},
        "height": height,
        "legend": {"orientation": "v", "x": 1.02, "y": 0.5},
    }


def create_combo_chart(data: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=data["SKU_Name"],
            y=data["Qty"],
            name="Sum of Qty",
            marker_color=BAR_COLOR,
            opacity=0.9,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data["SKU_Name"],
            y=data["Total_Sales"],
            name="Sum of Total Sales",
            mode="lines+markers",
            line={"color": LINE_COLOR, "width": 3},
            marker={"size": 8, "color": LINE_COLOR},
        ),
        secondary_y=True,
    )
    fig.update_layout(**base_layout("Qty vs Total Sales", 380))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title_text="Qty", gridcolor=GRID_COLOR, zeroline=False, secondary_y=False)
    fig.update_yaxes(title_text="Total Sales", gridcolor=GRID_COLOR, zeroline=False, tickformat=",.0f", secondary_y=True)
    return fig


def create_donut_chart(data: pd.DataFrame, names_col: str, values_col: str, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=data[names_col],
                values=data[values_col],
                hole=0.55,
                textinfo="percent",
                textfont={"color": TEXT_COLOR, "size": 14},
                marker={"colors": DONUT_COLORS[: len(data)]},
                sort=False,
            )
        ]
    )
    fig.update_layout(**base_layout(title, 300))
    fig.update_layout(margin={"l": 20, "r": 20, "t": 60, "b": 20}, legend={"orientation": "v", "x": 1.0, "y": 0.5})
    return fig


def create_outlet_bar_chart(data: pd.DataFrame) -> go.Figure:
    chart_data = data.head(5).sort_values("Total_Sales", ascending=True)
    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_data["Total_Sales"],
                y=chart_data["Outlet_Name"],
                orientation="h",
                marker_color=BAR_COLOR,
                text=chart_data["Total_Sales"].map(lambda x: f"{x:,.0f}"),
                textposition="outside",
                name="Sum of Total Sales",
            )
        ]
    )
    fig.update_layout(**base_layout("Sales by Outlet", 300))
    fig.update_xaxes(gridcolor=GRID_COLOR, tickformat=",.0f")
    fig.update_yaxes(showgrid=False)
    return fig


def create_city_bar_chart(data: pd.DataFrame) -> go.Figure:
    chart_data = data.sort_values("Total_Sales", ascending=False)
    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_data["City"],
                y=chart_data["Total_Sales"],
                marker_color=BAR_COLOR,
                text=chart_data["Total_Sales"].map(lambda x: f"{x:,.0f}"),
                textposition="outside",
                name="Sum of Total Sales",
            )
        ]
    )
    fig.update_layout(**base_layout("Sales by City", 300))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, tickformat=",.0f")
    return fig


# =========================
# Streamlit Layout
# =========================

def main() -> None:
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    st.markdown('<div class="dashboard-title">EXELTIS SALES DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">Interactive dashboard from raw distributor data to cleaned sales insight.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload raw Excel assessment",
        type=["xlsx"],
        help="Gunakan file raw yang berisi sheet Distributor A, Distributor B, dan SKU Master.",
    )

    sample_path = Path("Dataset.xlsx")
    if uploaded_file is None and sample_path.exists():
        st.info("File upload belum dipilih. App akan memakai **Dataset.xlsx** lokal sebagai sample.")
        file_bytes = sample_path.read_bytes()
    elif uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
    else:
        st.warning("Upload file raw Excel dulu untuk menjalankan dashboard.")
        st.stop()

    try:
        sales_data, outlet_master, _summary = process_file(file_bytes)
    except Exception as err:
        st.error(str(err))
        st.stop()

    filtered_df = apply_filters(sales_data)
    filtered_summary = build_summary_tables(filtered_df)
    insights = build_brief_analysis(filtered_summary)

    total_sales = filtered_df["Total_Sales"].sum()
    total_qty = filtered_df["Qty"].sum()
    total_outlet = filtered_df["Outlet_ID"].nunique()
    total_sku = filtered_df["SKU_ID"].nunique()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Sales", format_currency(total_sales))
    metric_cols[1].metric("Total Quantity", f"{total_qty:,.0f}")
    metric_cols[2].metric("Total Outlet", f"{total_outlet:,}")
    metric_cols[3].metric("Total SKU", f"{total_sku:,}")

    with st.expander("Brief Analysis", expanded=True):
        if insights:
            for point in insights:
                st.markdown(f'<div class="insight-box">{point}</div>', unsafe_allow_html=True)
        else:
            st.info("Belum ada insight karena hasil filter kosong.")


    row_2_col_1, row_2_col_2 = st.columns([1.2, 1])
    with row_2_col_1:
        if not filtered_summary["sales_by_outlet"].empty:
            st.plotly_chart(create_outlet_bar_chart(filtered_summary["sales_by_outlet"]), use_container_width=True)
        else:
            st.info("Tidak ada data untuk chart outlet.")

    with row_2_col_2:
        donut_sku_data = filtered_summary["sales_by_sku"].head(5)
        if not donut_sku_data.empty:
            st.plotly_chart(create_donut_chart(donut_sku_data, "SKU_Name", "Total_Sales", "Sales by SKU"), use_container_width=True)
        else:
            st.info("Tidak ada data untuk chart SKU.")

    row_3_col_1, row_3_col_2 = st.columns([1, 1.2])
    with row_3_col_1:
        if not filtered_summary["sales_by_segment"].empty:
            st.plotly_chart(create_donut_chart(filtered_summary["sales_by_segment"], "Segment", "Total_Sales", "Sales by Segment"), use_container_width=True)
        else:
            st.info("Tidak ada data untuk chart segment.")

    with row_3_col_2:
        if not filtered_summary["sales_by_city"].empty:
            st.plotly_chart(create_city_bar_chart(filtered_summary["sales_by_city"]), use_container_width=True)
        else:
            st.info("Tidak ada data untuk chart city.")

    top_combo_data = filtered_summary["qty_vs_sales"].head(5)
    if not top_combo_data.empty:
        st.plotly_chart(create_combo_chart(top_combo_data), use_container_width=True)
    else:
        st.info("Tidak ada data untuk chart Qty vs Total Sales.")

    tab1, tab2, tab3 = st.tabs(["Sales Data", "Outlet Master", "Summary Tables"])

    with tab1:
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    with tab2:
        st.dataframe(outlet_master, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### KPI Summary")
        st.dataframe(filtered_summary["kpi_summary"], use_container_width=True, hide_index=True)
        st.markdown("#### Sales by SKU")
        st.dataframe(filtered_summary["sales_by_sku"], use_container_width=True, hide_index=True)
        st.markdown("#### Sales by Segment")
        st.dataframe(filtered_summary["sales_by_segment"], use_container_width=True, hide_index=True)
        st.markdown("#### Sales by City")
        st.dataframe(filtered_summary["sales_by_city"], use_container_width=True, hide_index=True)
        st.markdown("#### Sales by Outlet")
        st.dataframe(filtered_summary["sales_by_outlet"], use_container_width=True, hide_index=True)

    export_bytes = to_excel_bytes(filtered_df, outlet_master, filtered_summary)
    st.download_button(
        label="⬇️ Download Processed Excel",
        data=export_bytes,
        file_name="Exeltis_Dashboard_Output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
