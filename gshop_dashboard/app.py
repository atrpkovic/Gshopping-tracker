from dash import Dash, html, dcc, dash_table, Output, Input, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------- Paths & Config --------------------
DATA_PATH_MAIN = Path("data/google_shopping_brand_hits.csv")
DATA_PATH_SAMPLE = Path("data/sample_google_shopping_brand_hits.csv")
AUTO_REFRESH_MS = 90_000  # 90s
THEME = dbc.themes.DARKLY  # change to FLATLY for light theme

# -------------------- Load & prep data --------------------
def load_df() -> pd.DataFrame:
    path = DATA_PATH_MAIN if DATA_PATH_MAIN.exists() else DATA_PATH_SAMPLE
    df = pd.read_csv(path, encoding="utf-8-sig")

    expected = [
        "Timestamp","Keyword","Location","Brand Domain","Product Title",
        "Price","Price Value","Old Price Value","Discount %",
        "Product Link","Merchant Name","Position","Rating","Reviews"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    for col in ["Price Value","Old Price Value","Discount %","Position","Rating","Reviews"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def extract_state(loc):
        if pd.isna(loc): return None
        parts = [p.strip() for p in str(loc).split(",")]
        if len(parts) >= 2:
            return parts[-2] if parts[-1].lower() == "united states" else parts[-1]
        return None

    df["State"] = df["Location"].apply(extract_state)
    df["Top3"] = (df["Position"] <= 3).astype(int)
    df["Top10"] = (df["Position"] <= 10).astype(int)
    return df

df = load_df()

# -------------------- App --------------------
app = Dash(__name__, external_stylesheets=[THEME], title="Google Shopping Dashboard", suppress_callback_exceptions=True)
server = app.server

def kpi_card(title, value):
    return dbc.Card(
        dbc.CardBody([html.Div(title, className="kpi-sub"), html.Div(value, className="kpi")]),
        className="mb-3",
    )

def controls(data: pd.DataFrame):
    max_pos = int(np.nanmax(data["Position"])) if data["Position"].notna().any() else 60
    return dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Keyword"),
                dcc.Dropdown(sorted(data["Keyword"].dropna().unique()), multi=True, id="kw",
                             placeholder="Select Keyword", className="dropdown-black")
            ], md=3),
            dbc.Col([
                html.Label("Brand"),
                dcc.Dropdown(sorted(data["Brand Domain"].dropna().unique()), multi=True, id="brand",
                             placeholder="Select Brand", className="dropdown-black")
            ], md=3),
            dbc.Col([
                html.Label("Location"),
                dcc.Dropdown(sorted(data["Location"].dropna().unique()), multi=True, id="loc",
                             placeholder="Select Location", className="dropdown-black")
            ], md=3),
            dbc.Col([
                html.Label("Date range"),
                dcc.DatePickerRange(id="dates", minimum_nights=0)
            ], md=3),
        ], className="gy-2"),

        html.Hr(),

        dbc.Row([
            dbc.Col([
                html.Label("Position range"),
                dcc.RangeSlider(min=1, max=max_pos, value=[1, min(10, max_pos)], id="pos",
                                tooltip={"placement": "bottom", "always_visible": False})
            ], md=6),
            dbc.Col([
                html.Label("Min discount (%)"),
                dcc.Slider(min=0, max=60, step=1, value=0, id="min_disc",
                           marks={i: f"{i}%" for i in range(0, 61, 10)},
                           tooltip={"placement": "bottom", "always_visible": True},
                           updatemode="drag", className="slider-clean")
            ], md=3),
            dbc.Col([
                html.Label("Auto-refresh"),
                dcc.Checklist(options=[{"label": " every 90s", "value": "on"}], value=["on"], id="refresh", inline=True)
            ], md=3),
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
    # Results Table (native filter row; readable inputs)
    dbc.Card(dbc.CardBody([
        html.H5("Results Table"),
        dash_table.DataTable(
            id="table",
            columns=[
                {"name":"Time",           "id":"Timestamp"},
                {"name":"Keyword",        "id":"Keyword"},
                {"name":"Location",       "id":"Location"},
                {"name":"Brand",          "id":"Brand Domain"},
                {"name":"Product Title",  "id":"Product Title"},
                {"name":"Price",          "id":"Price"},
                {"name":"Price Value",    "id":"Price Value",    "type":"numeric"},
                {"name":"Old Price",      "id":"Old Price Value","type":"numeric"},
                {"name":"Discount %",     "id":"Discount %",     "type":"numeric"},
                {"name":"Position",       "id":"Position",       "type":"numeric"},
                {"name":"Rating",         "id":"Rating",         "type":"numeric"},
                {"name":"Reviews",        "id":"Reviews",        "type":"numeric"},
                {"name":"Link",           "id":"Product Link",   "presentation":"markdown"},
                {"name":"Merchant",       "id":"Merchant Name"},
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
        # Make filter inputs readable (pairs with CSS)
            style_filter={
                "backgroundColor": "#ffffff",
                "color": "#111",
                "border": "0",
                "fontSize": "13px",
                "padding": "6px 8px",
                "height": "38px",
            },
            style_cell={
                "backgroundColor": "#ffffff",
                "color": "#111",
                "border": "0",
                "padding": "10px",
                "minWidth": "80px", "maxWidth": "340px",
                "whiteSpace": "nowrap", "textOverflow": "ellipsis", "overflow": "hidden",
            },
            style_cell_conditional=[
                {"if": {"column_id": "Product Title"}, "minWidth": "220px", "maxWidth": "420px"},
                {"if": {"column_id": "Location"},      "minWidth": "180px"},
                {"if": {"column_id": "Brand"},         "minWidth": "160px"},
                {"if": {"column_id": "Price Value"},   "textAlign": "right"},
                {"if": {"column_id": "Old Price"},     "textAlign": "right"},
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

# -------------------- Helpers --------------------
def apply_filters(df: pd.DataFrame, kws, brands, locs, date_range, pos_range, min_disc):
    dff = df.copy()
    if kws: dff = dff[dff["Keyword"].isin(kws)]
    if brands: dff = dff[dff["Brand Domain"].isin(brands)]
    if locs: dff = dff[dff["Location"].isin(locs)]
    if date_range and date_range[0] and date_range[1]:
        start = pd.to_datetime(date_range[0])
        end   = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        dff = dff[(dff["Timestamp"] >= start) & (dff["Timestamp"] < end)]
    if pos_range and len(pos_range) == 2:
        dff = dff[(dff["Position"] >= pos_range[0]) & (dff["Position"] <= pos_range[1])]
    if min_disc is not None and min_disc > 0:
        dff = dff[(dff["Discount %"].fillna(0) >= float(min_disc))]
    return dff

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
    Input("kw","value"),
    Input("brand","value"),
    Input("loc","value"),
    Input("dates","start_date"),
    Input("dates","end_date"),
    Input("pos","value"),
    Input("min_disc","value"),
    Input("interval","n_intervals"),
)
def update_dashboard(kws, brands, locs, start, end, pos_range, min_disc, _tick):
    global df
    if DATA_PATH_MAIN.exists():
        try:
            new_df = pd.read_csv(DATA_PATH_MAIN, encoding="utf-8-sig")
            if len(new_df) != len(df):
                new_df["Timestamp"] = pd.to_datetime(new_df["Timestamp"], errors="coerce", utc=True)
                for col in ["Price Value","Old Price Value","Discount %","Position","Rating","Reviews"]:
                    new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
                def extract_state(loc):
                    if pd.isna(loc): return None
                    parts = [p.strip() for p in str(loc).split(",")]
                    if len(parts) >= 2:
                        return parts[-2] if parts[-1].lower()=="united states" else parts[-1]
                    return None
                new_df["State"] = new_df["Location"].apply(extract_state)
                new_df["Top3"] = (new_df["Position"] <= 3).astype(int)
                new_df["Top10"] = (new_df["Position"] <= 10).astype(int)
                df = new_df
        except Exception:
            pass

    dff = apply_filters(df, kws, brands, locs, (start, end), pos_range, min_disc)

    n = len(dff)
    top3_share = (dff["Top3"].sum()/n*100) if n else 0
    avg_price = dff["Price Value"].mean() if n else np.nan
    avg_disc = dff["Discount %"].mean() if n else np.nan

    kpi_rows = f"{n:,}"
    kpi_top3 = f"{top3_share:.1f}%"
    kpi_avgp = f"${avg_price:,.2f}" if pd.notna(avg_price) else "—"
    kpi_avgd = f"{avg_disc:.1f}%" if pd.notna(avg_disc) else "—"

    d_top = dff[dff["Position"] <= 10]
    if len(d_top):
        s = d_top.groupby(["Brand Domain"])["Position"].count().sort_values(ascending=False).head(15)
        fig1 = px.bar(s, orientation="v", title="Brand count in Top-10")
        fig1.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    else:
        fig1 = go.Figure().update_layout(title="Brand count in Top-10 (no data)")

    if len(dff):
        fig2 = px.histogram(dff, x="Position", nbins=40, title="Position Distribution")
        fig2.update_xaxes(autorange="reversed")
        fig2.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    else:
        fig2 = go.Figure().update_layout(title="Position Distribution (no data)")

    piv = dff.pivot_table(index="Location", columns="Brand Domain", values="Discount %", aggfunc="mean")
    if piv.size:
        fig3 = px.imshow(piv, aspect="auto", title="Avg Discount % (Brand × Location)", labels=dict(color="Discount %"))
        fig3.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    else:
        fig3 = go.Figure().update_layout(title="Avg Discount % (no data)")

    if len(dff):
        fig4 = px.scatter(
            dff, x="Position", y="Price Value", color="Brand Domain",
            hover_data=["Keyword","Location","Product Title"], size="Rating",
            title="Price vs Position"
        )
        fig4.update_xaxes(autorange="reversed")
        fig4.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    else:
        fig4 = go.Figure().update_layout(title="Price vs Position (no data)")

    if len(dff):
        dff_day = dff.dropna(subset=["Timestamp"]).copy()
        dff_day["day"] = dff_day["Timestamp"].dt.tz_convert(None).dt.date
        s = dff_day.groupby("day")["Price Value"].mean().reset_index()
        fig5 = px.line(s, x="day", y="Price Value", markers=True, title="Daily Avg Price")
        fig5.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    else:
        fig5 = go.Figure().update_layout(title="Daily Avg Price (no data)")

    table_df = dff.copy()
    def mk_link(url):
        try:
            return f"[Open]({url})" if pd.notna(url) and str(url).startswith("http") else ""
        except Exception:
            return ""
    table_df["Product Link"] = table_df["Product Link"].apply(mk_link)
    cols = ["Timestamp","Keyword","Location","Brand Domain","Product Title","Price","Price Value",
            "Old Price Value","Discount %","Position","Rating","Reviews","Product Link","Merchant Name"]
    table_df = table_df[cols].sort_values(["Timestamp","Position"], ascending=[False, True]).head(500)

    return kpi_rows, kpi_top3, kpi_avgp, kpi_avgd, fig1, fig2, fig3, fig4, fig5, table_df.to_dict("records")

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
