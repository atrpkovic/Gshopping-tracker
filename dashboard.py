import streamlit as st
import pandas as pd
import plotly.express as px
import io
import json

st.set_page_config(page_title="Google Shopping Brand Tracker", layout="wide")

# ---------------------------------------
# 1) CONFIG: locations are code-driven
# ---------------------------------------
MULTI_LOCATIONS = [
    {"name": "Coppell, Texas, United States", "code": 9061237},
    {"name": "Dallas, Texas, United States", "code": 1023191},
    {"name": "Miami, Florida, United States", "code": 1021999},
    {"name": "Los Angeles, California, United States", "code": 1014221},
    {"name": "New York, New York, United States", "code": 1023768},
]
ACTIVE_LOCATION_NAMES = [l["name"] for l in MULTI_LOCATIONS]

# ---------------------------------------
# 2) HELPERS
# ---------------------------------------
@st.cache_data
def load_results(path: str = "google_shopping_brand_hits.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # normalize types
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for col in ("Position", "Rating", "Reviews"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # enforce using code-defined locations only
    df = df[df["Location"].isin(ACTIVE_LOCATION_NAMES)]
    return df

def parse_brands_upload(file) -> set[str]:
    """
    Accepts JSON (domain -> [aliases]) or CSV.
    CSV accepted shapes:
      - column 'domain' (domains to include)
      - column 'Brand Domain' (same as output CSV)
      - single-column CSV; first column treated as domain
    Returns a set of brand domains (lowercased).
    """
    if file is None:
        return set()
    name = (file.name or "").lower()
    try:
        if name.endswith(".json"):
            parsed = json.load(file)
            # keys are domains
            return {str(k).lower() for k in parsed.keys()}
        else:
            # assume CSV
            file.seek(0)
            dfb = pd.read_csv(file)
            cols = [c.strip().lower() for c in dfb.columns]
            dfb.columns = cols
            if "domain" in cols:
                vals = dfb["domain"].dropna().astype(str)
            elif "brand domain" in cols:
                vals = dfb["brand domain"].dropna().astype(str)
            else:
                # fall back to first column
                first = dfb.columns[0]
                vals = dfb[first].dropna().astype(str)
            return {v.strip().lower() for v in vals if v.strip()}
    except Exception:
        # fallback: try plain text list, one per line
        try:
            file.seek(0)
            text = file.read()
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="ignore")
            return {line.strip().lower() for line in text.splitlines() if line.strip()}
        except Exception:
            return set()

def parse_keywords_upload(file) -> set[str]:
    """
    Accepts CSV or TXT with a column/lines of keywords.
    CSV accepted shapes:
      - column 'keyword' (preferred)
      - column 'keywords'
      - single-column CSV; first column treated as keyword
    Returns a set of keywords (case-insensitive match later).
    """
    if file is None:
        return set()
    name = (file.name or "").lower()
    try:
        if name.endswith(".csv"):
            file.seek(0)
            dfk = pd.read_csv(file)
            cols = [c.strip().lower() for c in dfk.columns]
            dfk.columns = cols
            if "keyword" in cols:
                vals = dfk["keyword"].dropna().astype(str)
            elif "keywords" in cols:
                vals = dfk["keywords"].dropna().astype(str)
            else:
                first = dfk.columns[0]
                vals = dfk[first].dropna().astype(str)
            return {v.strip() for v in vals if v.strip()}
        else:
            # txt or anything else: one keyword per line
            file.seek(0)
            text = file.read()
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="ignore")
            return {line.strip() for line in text.splitlines() if line.strip()}
    except Exception:
        return set()

# ---------------------------------------
# 3) UI: uploads + info about locations
# ---------------------------------------
st.title("🛍️ Google Shopping Brand Tracker – Dashboard")

with st.expander("Active locations (from code)", expanded=True):
    st.write(
        "This dashboard filters data to the following locations defined in code "
        "(no location dropdown is shown):"
    )
    st.write(", ".join(ACTIVE_LOCATION_NAMES))

st.sidebar.header("📤 Upload Filters (optional)")
brands_file = st.sidebar.file_uploader(
    "Upload brands (JSON or CSV)", type=["json", "csv", "txt"]
)
keywords_file = st.sidebar.file_uploader(
    "Upload keywords (CSV or TXT)", type=["csv", "txt"]
)
st.sidebar.caption(
    "• Brands JSON format: `{ \"yourbrand.com\": [\"Brand Name\", \"Alias\"] }`\n"
    "• Brands CSV: a column named `domain` or `Brand Domain` (or single column)\n"
    "• Keywords CSV/TXT: a column `keyword` or one keyword per line"
)

brands_filter = parse_brands_upload(brands_file)
keywords_filter = parse_keywords_upload(keywords_file)

# ---------------------------------------
# 4) Load & filter data
# ---------------------------------------
df = load_results()

# apply brands filter if provided
if brands_filter:
    df = df[df["Brand Domain"].str.lower().isin(brands_filter)]

# apply keywords filter if provided (case-insensitive)
if keywords_filter:
    df = df[df["Keyword"].str.lower().isin({k.lower() for k in keywords_filter})]

# ---------------------------------------
# 5) KPIs
# ---------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Rows", len(df))
kpi2.metric("Unique Keywords", df["Keyword"].nunique())
kpi3.metric("Brands", df["Brand Domain"].nunique())
kpi4.metric("Locations", df["Location"].nunique())

st.divider()

# ---------------------------------------
# 6) Charts
# ---------------------------------------
if df.empty:
    st.warning("No data after applying filters. Upload different brands/keywords or re-run the tracker.")
else:
    # Average Position by Brand and Location
    st.subheader("📈 Average Position by Brand and Location")
    grp = (
        df.groupby(["Brand Domain", "Location"], as_index=False)["Position"]
        .mean()
        .sort_values("Position")
    )
    fig1 = px.bar(
        grp,
        x="Brand Domain",
        y="Position",
        color="Location",
        barmode="group",
        title="Average Product Position (lower is better)",
        labels={"Position": "Avg Position"}
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Position Trends over Time (average by brand per timestamp)
    st.subheader("📆 Position Trend Over Time")
    trend = (
        df.groupby(["Timestamp", "Brand Domain"], as_index=False)["Position"]
        .mean()
        .sort_values("Timestamp")
    )
    fig2 = px.line(
        trend, x="Timestamp", y="Position", color="Brand Domain",
        title="Position Trends Over Time", markers=True
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Top 10 best positions
    st.subheader("🏆 Top 10 Best Positions")
    top10 = df.sort_values("Position").head(10)
    st.dataframe(
        top10[["Timestamp", "Keyword", "Location", "Brand Domain", "Product Title", "Position"]],
        hide_index=True
    )

    # Download filtered results
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv_buf.getvalue(),
        file_name="filtered_google_shopping_brand_hits.csv",
        mime="text/csv"
    )

# ---------------------------------------
# 7) Raw table toggle
# ---------------------------------------
with st.expander("See raw data"):
    st.dataframe(df, hide_index=True)
