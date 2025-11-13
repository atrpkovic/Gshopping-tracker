from dash import Dash, html, dcc, dash_table, Output, Input, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import os

# -------------------- Paths & Config --------------------
AUTO_REFRESH_MS = 90_000  # 90s
THEME = dbc.themes.DARKLY  # change to FLATLY for light theme

# ---- DATA PATH RESOLUTION (xlsx first, csv fallback) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_ENV = os.getenv("GSHOP_OUTPUT")  # optional override

CANDIDATES = [
    DATA_PATH_ENV,
    os.path.join(BASE_DIR, "data", "google_shopping_brand_hits.xlsx"),
    os.path.join(BASE_DIR, "..", "google_shopping_brand_hits.xlsx"),
    os.path.join(BASE_DIR, "data", "google_shopping_brand_hits.csv"),
    os.path.join(BASE_DIR, "..", "google_shopping_brand_hits.csv"),
]

def _first_existing(paths):
    for p in [p for p in paths if p]:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

def _read_any(path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(path.lower())
    if ext == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    elif ext == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported data file type: {ext}")

def load_df() -> pd.DataFrame:
    path = _first_existing(CANDIDATES)
    if not path:
        raise FileNotFoundError(
            "Cannot find google_shopping_brand_hits.[xlsx|csv]. "
            "Searched:\n  - " + "\n  - ".join([p for p in CANDIDATES if p])
        )
    print(f"[dashboard] Loading data from: {path}")
    return _read_any(path)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Strip headers and unify common names
    df.columns = [str(c).strip() for c in df.columns]

    # Map Excel → app
    rename_map = {
        "Title": "Product Title",
        "Merchant Name": "Merchant",
        "Brand Domain": "Brand",          # if it ever appears
        "Link": "Product Link",           # if it ever appears
    }
    df.rename(columns=rename_map, inplace=True)

    # Ensure core columns exist
    for col in [
        "Timestamp","Sheet","Keyword","Location","Position",
        "Product Title","Price","Price Value","Old Price","Old Price Value",
        "Rating","Reviews","Merchant","Product Link","Brand","Matched Synonym"
    ]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    # Types
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    num_cols = ["Position","Price Value","Old Price Value","Rating","Reviews"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Discount % (compute if missing)
    if "Discount %" not in df.columns:
        df["Discount %"] = np.where(
            df["Old Price Value"].gt(0) & df["Price Value"].notna(),
            (df["Old Price Value"] - df["Price Value"]) / df["Old Price Value"] * 100.0,
            np.nan,
        )
    else:
        df["Discount %"] = pd.to_numeric(df["Discount %"], errors="coerce")

    return df


df = normalize_columns(load_df())

# -------------------- App --------------------
app = Dash(__name__, external_stylesheets=[THEME], title="Google Shopping Dashboard", suppress_callback_exceptions=True)
server = app.server

def kpi_card(title, value):
    return dbc.Card(
        dbc.CardBody([html.Div(title, className="kpi-sub"), html.Div(value, className="kpi")]),
        className="mb-3",
    )

def controls(data: pd.DataFrame):
    pos_vals = pd.to_numeric(data.get("Position"), errors="coerce")
    pos_min_default = int(np.nanmin(pos_vals)) if pos_vals.notna().any() else 1
    pos_max_default = int(np.nanmax(pos_vals)) if pos_vals.notna().any() else 60

    return dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([html.Label("Keyword"),
                     dcc.Dropdown(sorted(data["Keyword"].dropna().unique()),
                                  id="kw", multi=True)], md=3),
            dbc.Col([html.Label("Brand"),
                     dcc.Dropdown(sorted(data["Brand"].dropna().unique()),
                                  id="Brand", multi=True)], md=3),
            dbc.Col([html.Label("Location"),
                     dcc.Dropdown(sorted(data["Location"].dropna().unique()),
                                  id="loc", multi=True)], md=3),
            dbc.Col([html.Label("Date range"),
                     dcc.DatePickerRange(id="dates",
                                         start_date=data["Timestamp"].min(),
                                         end_date=data["Timestamp"].max())], md=3),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.Label("Position range"),
                dcc.RangeSlider(id="pos",
                    min=pos_min_default, max=pos_max_default,
                    value=[pos_min_default, pos_max_default])
            ], md=6),
            dbc.Col([
                html.Label("Min. discount (%)"),
                dcc.Slider(id="min_disc", min=0, max=80, step=1, value=0)
            ], md=6),
        ]),
    ]), className="mb-4")


app.layout = dbc.Container([
    html.Br(),
    html.H2("Google Shopping Dashboard", id="header-title"),
    html.Div("Explore placements, prices, and discounts across brands, keywords, and locations.", className="text-muted mb-3"),
    controls(df),

    dbc.Row([
        dbc.Col(kpi_card("Observations", html.Span(id="kpi-rows")), md=3),
        dbc.Col(kpi_card("Top-3 Share", html.Span(id="kpi-top3")), md=3),
        dbc.Col(kpi_card("Avg Price", html.Span(id="kpi-avgprice")), md=3),
        dbc.Col(kpi_card("Avg Discount", html.Span(id="kpi-avgdisc")), md=3),
    ], className="gy-3"),

    html.Br(),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="fig_brand_share")])), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="fig_pos_dist")])), md=6),
    ], className="gy-3"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="fig_heatmap")])) , md=7),
        dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="fig_scatter")])) , md=5),
    ], className="gy-3"),
    dbc.Row([dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="fig_timeseries")])) , md=12)], className="gy-3"),

    # Results Table (native filter row; readable inputs)
    dbc.Card(dbc.CardBody([
        html.H5("Results Table"),
        dash_table.DataTable(
            id="table",
            columns= [
                {"name":"Time",            "id":"Timestamp"},
                {"name":"Sheet",           "id":"Sheet"},
                {"name":"Keyword",         "id":"Keyword"},
                {"name":"Location",        "id":"Location"},
                {"name":"Position",        "id":"Position",        "type":"numeric"},
                {"name":"Product Title",   "id":"Product Title"},
                {"name":"Price",           "id":"Price"},
                {"name":"Price Value",     "id":"Price Value",     "type":"numeric"},
                {"name":"Old Price",       "id":"Old Price"},
                {"name":"Old Price Value", "id":"Old Price Value", "type":"numeric"},
                {"name":"Rating",          "id":"Rating",          "type":"numeric"},
                {"name":"Reviews",         "id":"Reviews",         "type":"numeric"},
                {"name":"Merchant",        "id":"Merchant"},
                {"name":"Link",            "id":"Product Link",    "presentation":"markdown"},
                {"name":"Brand",           "id":"Brand"},
                {"name":"Matched Synonym", "id":"Matched Synonym"},
                {"name":"Discount %",      "id":"Discount %",      "type":"numeric"},
                ],
            style_as_list_view=True,
            page_size=25,
            sort_action="native",
            filter_action="native",
            filter_options={"placeholder_text": "Filter…", "case": "insensitive"},
            export_format="csv",
            export_headers="display",
            fixed_rows={"headers": True},
            fill_width=True,
            style_table={"overflowX": "auto", "height": "520px", "borderRadius": "12px", "minWidth": "100%"},
            style_header={"backgroundColor": "#0f0f10","color": "#ffffff","fontWeight": "700","border": "0"},
            style_filter={
                "backgroundColor": "#ffffff","color": "#111","border": "0",
                "fontSize": "13px","padding": "6px 8px","height": "38px",
            },
            style_cell={
                "backgroundColor": "#ffffff","color": "#111","border": "0","padding": "10px",
                "minWidth": "80px","maxWidth": "340px","whiteSpace": "nowrap",
                "textOverflow": "ellipsis","overflow": "hidden",
            },
            style_cell_conditional=[
                {"if": {"column_id": "Product Title"}, "minWidth": "220px", "maxWidth": "420px"},
                {"if": {"column_id": "Location"},      "minWidth": "180px"},
                {"if": {"column_id": "Brand"},         "minWidth": "160px"},
                {"if": {"column_id": "Price Value"},   "textAlign": "right"},
                {"if": {"column_id": "Old Price Value"},"textAlign": "right"},
                {"if": {"column_id": "Discount %"},    "textAlign": "right"},
                {"if": {"column_id": "Position"},      "textAlign": "right"},
                {"if": {"column_id": "Rating"},        "textAlign": "right"},
                {"if": {"column_id": "Reviews"},       "textAlign": "right"},
            ],
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
                {"if": {"filter_query": "{Position} <= 3"}, "backgroundColor": "#e8f5e9"},
                {"if": {"filter_query": "{Discount %} >= 20"}, "backgroundColor": "#fff3cd"},
            ],
        )
    ]), className="mb-5"),

    dcc.Interval(id="interval", interval=AUTO_REFRESH_MS, n_intervals=0),
], fluid=True)

# -------------------- Callbacks --------------------
@callback(
    Output("interval","disabled"),
    Input("refresh","value")
)
def toggle_refresh(val):
    return False if ("on" in (val or [])) else True

@callback(
    Output("kpi-rows","children"),
    Output("kpi-top3","children"),
    Output("kpi-avgprice","children"),
    Output("kpi-avgdisc","children"),
    Output("fig_brand_share","figure"),
    Output("fig_pos_dist","figure"),
    Output("fig_heatmap","figure"),
    Output("fig_scatter","figure"),
    Output("fig_timeseries","figure"),
    Output("table","data"),
    Input("interval","n_intervals"),
    Input("kw","value"),
    Input("Brand","value"),
    Input("loc","value"),
    Input("dates","start_date"),
    Input("dates","end_date"),
    Input("pos","value"),          # [min, max]
    Input("min_disc","value"),
)
def update_dashboard(n, kw_values, brand_values, loc_values, start_date, end_date, pos_range, min_disc):
    dff = df.copy()

    # Normalize inputs
    kw_values = kw_values or []
    brand_values = brand_values or []
    loc_values = loc_values or []
    pos_min, pos_max = (pos_range or [None, None])
    if isinstance(pos_min, list):  # in case a tuple slips in weirdly
        pos_min, pos_max = pos_min

    # Date filter
    if start_date or end_date:
        if "Timestamp" in dff.columns:
            dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], errors="coerce")
            if start_date:
                dff = dff[dff["Timestamp"] >= pd.to_datetime(start_date)]
            if end_date:
                dff = dff[dff["Timestamp"] <= (pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]

    # Keyword filter
    if kw_values and "Keyword" in dff.columns:
        dff = dff[dff["Keyword"].isin(kw_values)]

    # Brand filter
    brand_col = "Brand" if "Brand" in dff.columns else ("Brand Domain" if "Brand Domain" in dff.columns else None)
    if brand_values and brand_col:
        dff = dff[dff[brand_col].isin(brand_values)]

    # Location filter
    if loc_values and "Location" in dff.columns:
        dff = dff[dff["Location"].isin(locs := loc_values)]

    # Position range
    if ("Position" in dff.columns) and (pos_min is not None) and (pos_max is not None):
        pos_series = pd.to_numeric(dff["Position"], errors="coerce")
        dff = dff[(pos_series >= int(pos_min)) & (pos_series <= int(pos_max))]

    # Min discount
    if ("Discount %" in dff.columns) and (min_disc is not None):
        dff = dff[pd.to_numeric(dff["Discount %"], errors="coerce").fillna(0) >= float(min_disc)]

    # Empty-safe UI
    def empty_fig(msg):
        f = go.Figure()
        f.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                         showarrow=False, font=dict(size=14))
        f.update_layout(margin=dict(l=20, r=20, t=30, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_visible=False, yaxis_visible=False)
        return f

    if dff.empty:
        return (
            html.Div("No data for selected filters.", className="text-muted"),
            "—", "—", "—",
            empty_fig("No data"), empty_fig("No data"), empty_fig("No data"),
            empty_fig("No data"), empty_fig("No data"),
            [],
        )

    # KPIs
    top3_share = 0.0
    if "Position" in dff.columns:
        pos_num = pd.to_numeric(dff["Position"], errors="coerce")
        top3_share = (pos_num.le(3).mean() * 100.0) if pos_num.notna().any() else 0.0

    avg_price = None
    for col in ["Price Value", "Price"]:
        if col in dff.columns:
            avg_price = pd.to_numeric(dff[col], errors="coerce").mean()
            break
    avg_disc = pd.to_numeric(dff["Discount %"], errors="coerce").mean() if "Discount %" in dff.columns else None

    kpi_rows = html.Div([
        html.Div([html.Div("Top 3 Share", className="kpi-label"),
                  html.Div(f"{top3_share:0.1f}%", className="kpi-value")], className="kpi"),
        html.Div([html.Div("Avg Price", className="kpi-label"),
                  html.Div("—" if avg_price is None or np.isnan(avg_price) else f"${avg_price:,.2f}", className="kpi-value")], className="kpi"),
        html.Div([html.Div("Avg Discount", className="kpi-label"),
                  html.Div("—" if avg_disc is None or np.isnan(avg_disc) else f"{avg_disc:0.1f}%", className="kpi-value")], className="kpi"),
    ], className="kpi-row")

    kpi_top3 = f"{top3_share:0.1f}%"
    kpi_avgprice = "—" if avg_price is None or np.isnan(avg_price) else f"${avg_price:,.2f}"
    kpi_avgdisc = "—" if avg_disc is None or np.isnan(avg_disc) else f"{avg_disc:0.1f}%"

    # Brand share
    brand_share_fig = empty_fig("No data")
    if brand_col:
        counts = dff[brand_col].fillna("Unknown").value_counts()
        brand_share_fig = go.Figure(data=[go.Bar(x=counts.index.astype(str), y=counts.values)])
        brand_share_fig.update_layout(title="Listings by Brand", margin=dict(l=30, r=20, t=40, b=60),
                                      xaxis_title="Brand", yaxis_title="Count")

    # Position distribution
    pos_dist_fig = empty_fig("No data")
    if "Position" in dff.columns:
        pos_vals = pd.to_numeric(dff["Position"], errors="coerce").dropna()
        if not pos_vals.empty:
            pos_dist_fig = go.Figure(data=[go.Histogram(x=pos_vals, nbinsx=25)])
            pos_dist_fig.update_layout(title="Position Distribution", xaxis_title="Position",
                                       yaxis_title="Frequency", margin=dict(l=30, r=20, t=40, b=40))

    # Heatmap: avg discount by Location × Brand
    heatmap_fig = empty_fig("No discount data available for current filters")
    if ("Discount %" in dff.columns) and brand_col and ("Location" in dff.columns):
        try:
            piv = dff.pivot_table(index="Location", columns=brand_col, values="Discount %", aggfunc="mean")
            if piv is not None and not piv.empty:
                heatmap_fig = go.Figure(
                    data=go.Heatmap(z=piv.values, x=piv.columns.astype(str), y=piv.index.astype(str),
                                    colorscale="Blues", colorbar=dict(title="Avg Discount %"))
                )
                heatmap_fig.update_layout(title="Avg Discount by Location × Brand", margin=dict(l=40, r=20, t=40, b=40))
        except Exception:
            pass

    # Scatter: Price vs Position colored by brand
    scatter_fig = empty_fig("No data")
    if "Position" in dff.columns and (("Price Value" in dff.columns) or ("Price" in dff.columns)):
        price_col = "Price Value" if "Price Value" in dff.columns else "Price"
        tmp = dff[[price_col, "Position"] + ([brand_col] if brand_col else [])].copy()
        tmp[price_col] = pd.to_numeric(tmp[price_col], errors="coerce")
        tmp["Position"] = pd.to_numeric(tmp["Position"], errors="coerce")
        tmp = tmp.dropna(subset=[price_col, "Position"])
        if not tmp.empty:
            scatter_fig = px.scatter(tmp, x="Position", y=price_col,
                                     color=brand_col if brand_col else None,
                                     title="Price vs Position",
                                     labels={"Position": "Position", price_col: "Price"})
            scatter_fig.update_layout(margin=dict(l=30, r=20, t=40, b=40))

    # Time series: avg position over time per brand
    ts_fig = empty_fig("No data")
    if "Timestamp" in dff.columns and "Position" in dff.columns:
        ts = dff.copy()
        ts["Timestamp"] = pd.to_datetime(ts["Timestamp"], errors="coerce")
        ts["Position"] = pd.to_numeric(ts["Position"], errors="coerce")
        ts = ts.dropna(subset=["Timestamp", "Position"])
        if not ts.empty:
            if brand_col:
                grp = ts.groupby([pd.Grouper(key="Timestamp", freq="D"), brand_col])["Position"].mean().reset_index()
                ts_fig = px.line(grp, x="Timestamp", y="Position", color=brand_col, title="Avg Position over Time")
            else:
                grp = ts.groupby(pd.Grouper(key="Timestamp", freq="D"))["Position"].mean().reset_index()
                ts_fig = px.line(grp, x="Timestamp", y="Position", title="Avg Position over Time")
            ts_fig.update_layout(margin=dict(l=30, r=20, t=40, b=40), yaxis_autorange="reversed")

    # Table data
    table_cols = ["Timestamp", "Keyword", "Location", (brand_col or "Brand"), "Product Title",
                  "Price", "Price Value", "Old Price Value", "Discount %",
                  "Position", "Rating", "Reviews", "Product Link"]
    table_cols = [c for c in table_cols if c in dff.columns]
    table_data = dff[table_cols].sort_values(by=[c for c in ["Timestamp"] if c in table_cols],
                                             ascending=False, na_position="last").to_dict("records")

    return (
        kpi_rows, kpi_top3, kpi_avgprice, kpi_avgdisc,
        brand_share_fig, pos_dist_fig, heatmap_fig, scatter_fig, ts_fig,
        table_data,
    )

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
