import os
import io
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# ----------------------------
# Secrets / environment
# ----------------------------
# On Streamlit Cloud, set SERPAPI_KEY in "App → Settings → Secrets"
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY") or os.getenv("SERPAPI_KEY")
if SERPAPI_KEY:
    os.environ["SERPAPI_KEY"] = SERPAPI_KEY  # ensure backend sees it

# ----------------------------
# Import your backend module
# (safe: it only runs main() when __name__ == '__main__')
# ----------------------------
import gshop_tracker as tracker  # this is your gshop-tracker.py (rename file if needed)

# File names expected by the backend:
KEYWORDS_FILE = getattr(tracker, "KEYWORDS_FILE", "keywords.csv")
BRANDS_FILE = getattr(tracker, "BRANDS_FILE", "brands.json")
OUTPUT_FILE = getattr(tracker, "OUTPUT_FILE", "google_shopping_brand_hits.csv")

# ----------------------------
# Streamlit page setup
# ----------------------------
st.set_page_config(page_title="Google Shopping Brand Tracker – Runner", layout="wide")
st.title("🛍️ Google Shopping Brand Tracker – Frontend Runner")
st.caption("Upload brands & keywords → run the tracker (SerpApi) → download & analyze fresh results")

# ----------------------------
# Locations (from code – no dropdowns)
# ----------------------------
locations = getattr(tracker, "MULTI_LOCATIONS", [])
loc_names = [l.get("name") for l in locations]

with st.expander("Active locations (from code)", expanded=True):
    if locations:
        st.write(", ".join(loc_names))
    else:
        st.warning("No MULTI_LOCATIONS found in backend; tracker will use single LOCATION/LOCATION_CODE if set there.")

# ----------------------------
# Uploaders
# ----------------------------
st.sidebar.header("📤 Upload Inputs")
brands_file = st.sidebar.file_uploader("Brands (JSON or CSV)", type=["json", "csv"])
keywords_file = st.sidebar.file_uploader("Keywords (CSV or TXT)", type=["csv", "txt"])

st.sidebar.caption(
    "• Brands JSON: `{ \"yourbrand.com\": [\"Brand\", \"Alias\"] }`\n"
    "• Brands CSV: a column `domain` or `Brand Domain` (or single column of domains)\n"
    "• Keywords CSV: a column `keyword` / `keywords` (or single column) — TXT: one per line"
)

# ----------------------------
# Helpers to persist uploads to the filenames the backend expects
# ----------------------------
def persist_brands(file) -> bool:
    """Write brands to brands.json as {domain: [aliases...]}"""
    if not file:
        return False
    name = (file.name or "").lower()
    try:
        if name.endswith(".json"):
            file.seek(0)
            data = json.load(file)
            # must be dict; minimal validation
            if not isinstance(data, dict):
                return False
            with open(BRANDS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        # CSV → map each domain to []
        file.seek(0)
        df = pd.read_csv(file)
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        if "domain" in cols:
            vals = df["domain"].dropna().astype(str)
        elif "brand domain" in cols:
            vals = df["brand domain"].dropna().astype(str)
        else:
            first = df.columns[0]
            vals = df[first].dropna().astype(str)
        out = {v.strip().lower(): [] for v in vals if str(v).strip()}
        with open(BRANDS_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

def persist_keywords(file) -> bool:
    """Write keywords to keywords.csv (one per line)"""
    if not file:
        return False
    name = (file.name or "").lower()
    try:
        if name.endswith(".csv"):
            file.seek(0)
            df = pd.read_csv(file)
            cols = [c.strip().lower() for c in df.columns]
            df.columns = cols
            if "keyword" in cols:
                vals = df["keyword"].dropna().astype(str)
            elif "keywords" in cols:
                vals = df["keywords"].dropna().astype(str)
            else:
                first = df.columns[0]
                vals = df[first].dropna().astype(str)
            kw = [v.strip() for v in vals if v and str(v).strip()]
        else:
            # TXT or anything else: one per line
            file.seek(0)
            txt = file.read()
            if isinstance(txt, bytes):
                txt = txt.decode("utf-8", errors="ignore")
            kw = [line.strip() for line in txt.splitlines() if line.strip()]
        # write CSV (one keyword per line)
        with open(KEYWORDS_FILE, "w", encoding="utf-8-sig", newline="") as f:
            for k in kw:
                f.write(f"{k}\n")
        return True
    except Exception:
        return False

# ----------------------------
# Run button
# ----------------------------
run_col, _ = st.columns([1, 3])
with run_col:
    clicked = st.button("▶️ Run Tracker", type="primary", use_container_width=True)

# ----------------------------
# Execute backend + visualize
# ----------------------------
if clicked:
    # Pre-checks
    if not SERPAPI_KEY:
        st.error("Missing SERPAPI_KEY. Add it in Streamlit Cloud → App → Settings → Secrets, or set an ENV var locally.")
        st.stop()
    if not brands_file or not keywords_file:
        st.error("Please upload both **Brands** and **Keywords** files.")
        st.stop()

    ok_b = persist_brands(brands_file)
    ok_k = persist_keywords(keywords_file)
    if not ok_b:
        st.error("Brands file could not be parsed/saved. Please upload JSON (dict) or CSV with a `domain` / `Brand Domain` column.")
        st.stop()
    if not ok_k:
        st.error("Keywords file could not be parsed/saved. Upload CSV (keyword column) or TXT with one keyword per line.")
        st.stop()

    # Run tracker (calls your main(), writes OUTPUT_FILE)
    with st.status("Running tracker… contacting SerpApi and aggregating results", expanded=True) as status:
        st.write(f"• Locations: {', '.join(loc_names) if loc_names else 'from backend config'}")
        st.write(f"• Output file: `{OUTPUT_FILE}`")
        tracker.main()  # <-- executes your backend end-to-end
        status.update(label="Done", state="complete")

    if not os.path.exists(OUTPUT_FILE):
        st.warning("Tracker finished but output file was not found.")
        st.stop()

    # Load fresh results
    df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
    # normalize types
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for col in ("Position", "Rating", "Reviews"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.empty:
        st.warning("Tracker ran, but no matching brand results were found.")
        st.stop()

    # Save a timestamped copy for archival + offer download
    stamped = f"google_shopping_brand_hits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(stamped, index=False, encoding="utf-8-sig")

    st.success(f"Results saved: `{stamped}`")
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button("⬇️ Download CSV", data=buf.getvalue(), file_name=stamped, mime="text/csv")

    # ---------------- Charts ----------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", len(df))
    k2.metric("Unique Keywords", df["Keyword"].nunique() if "Keyword" in df else 0)
    k3.metric("Brands", df["Brand Domain"].nunique() if "Brand Domain" in df else 0)
    k4.metric("Locations", df["Location"].nunique() if "Location" in df else 0)

    st.divider()

    if {"Brand Domain", "Location", "Position"}.issubset(df.columns):
        st.subheader("📈 Average Position by Brand and Location")
        avg_pos = (
            df.groupby(["Brand Domain", "Location"], as_index=False)["Position"]
              .mean()
              .sort_values("Position")
        )
        fig1 = px.bar(
            avg_pos, x="Brand Domain", y="Position", color="Location",
            barmode="group", title="Average Product Position (lower is better)",
            labels={"Position": "Avg Position"}
        )
        st.plotly_chart(fig1, use_container_width=True)

    if {"Timestamp", "Brand Domain", "Position"}.issubset(df.columns):
        st.subheader("📆 Position Trend Over Time")
        trend = (
            df.groupby(["Timestamp", "Brand Domain"], as_index=False)["Position"]
              .mean()
              .sort_values("Timestamp")
        )
        fig2 = px.line(
            trend, x="Timestamp", y="Position", color="Brand Domain", markers=True,
            title="Position Trends Over Time"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🏆 Top 10 Best Positions")
    cols_keep = [c for c in ["Timestamp","Keyword","Location","Brand Domain","Product Title","Position"] if c in df.columns]
    st.dataframe(df.sort_values("Position").head(10)[cols_keep], hide_index=True)

# ----------------------------
# Footer help
# ----------------------------
with st.expander("Notes / Troubleshooting"):
    st.markdown(
        "- This app writes your uploads to the filenames your backend expects "
        f"(`{BRANDS_FILE}`, `{KEYWORDS_FILE}`), then runs `gshop-tracker.py`'s `main()`.\n"
        "- On Streamlit Cloud, add your SerpApi key in **Secrets** as `SERPAPI_KEY`.\n"
        "- Blank results usually mean brand names/aliases didn’t match merchant/source/title fields."
    )
