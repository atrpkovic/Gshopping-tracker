import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Google Shopping Brand Tracker", layout="wide")

# ---- Load data ----
@st.cache_data
def load_data():
    df = pd.read_csv("google_shopping_brand_hits.csv", encoding="utf-8-sig")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")
    return df.dropna(subset=["Keyword", "Location", "Brand Domain"])

df = load_data()

st.title("🛍️ Google Shopping Brand Tracker")
st.caption("Interactive dashboard to explore keyword performance, brand rankings, and positions by location.")

# ---- Sidebar filters ----
with st.sidebar:
    st.header("🔍 Filters")
    locations = sorted(df["Location"].dropna().unique())
    brands = sorted(df["Brand Domain"].dropna().unique())

    selected_locations = st.multiselect("Select Location(s)", locations, default=locations[:3])
    selected_brands = st.multiselect("Select Brand(s)", brands, default=brands[:5])
    keyword = st.text_input("Search Keyword (optional)", "")

# ---- Filter data ----
filtered = df[
    df["Location"].isin(selected_locations)
    & df["Brand Domain"].isin(selected_brands)
]

if keyword:
    filtered = filtered[filtered["Keyword"].str.contains(keyword, case=False, na=False)]

# ---- KPIs ----
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(filtered))
col2.metric("Unique Keywords", filtered["Keyword"].nunique())
col3.metric("Unique Brands", filtered["Brand Domain"].nunique())

st.divider()

# ---- Average Position by Brand and Location ----
st.subheader("📈 Average Position by Brand and Location")
avg_pos = (
    filtered.groupby(["Brand Domain", "Location"], as_index=False)["Position"]
    .mean()
    .sort_values("Position")
)

fig1 = px.bar(
    avg_pos,
    x="Brand Domain",
    y="Position",
    color="Location",
    barmode="group",
    title="Average Product Position by Brand and Location",
    labels={"Position": "Avg Position (lower is better)"}
)
st.plotly_chart(fig1, use_container_width=True)

# ---- Position Over Time ----
st.subheader("📆 Position Trend Over Time")
trend = (
    filtered.groupby(["Timestamp", "Brand Domain"], as_index=False)["Position"]
    .mean()
    .sort_values("Timestamp")
)

fig2 = px.line(
    trend,
    x="Timestamp",
    y="Position",
    color="Brand Domain",
    title="Position Trends Over Time",
    markers=True
)
st.plotly_chart(fig2, use_container_width=True)

# ---- Top 10 Results Table ----
st.subheader("🏆 Top 10 Best Positions")
top10 = filtered.sort_values("Position").head(10)
st.dataframe(
    top10[["Timestamp", "Keyword", "Location", "Brand Domain", "Product Title", "Position"]],
    hide_index=True
)

# ---- Raw Data Toggle ----
with st.expander("See raw data"):
    st.dataframe(filtered, hide_index=True)
