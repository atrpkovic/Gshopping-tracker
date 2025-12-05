from dash import Dash, html, dcc, dash_table, Output, Input, State, callback, ctx, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from functools import lru_cache

# -------------------- Paths & Config --------------------
AUTO_REFRESH_MS = 90_000  # 90s
THEME = dbc.themes.DARKLY

# ---- PRESETS ----
PRESETS = {
    "marketplace": ["priority tire", "walmart", "ebay"],
    "specialists": ["priority tire", "tire rack", "simpletire", "giga tires", "tires easy"],
}

# The pinned seller (will appear first in all multiselects, selected by default)
PINNED_SELLER = "priority tire"

# ---- DATA PATH RESOLUTION ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH_ENV = os.getenv("GSHOP_OUTPUT")

CANDIDATES = [
    r"C:\Users\MDC21\vsfiles\.vscode\Gshopping-tracker\google_shopping_brand_hits.xlsx",  # Local dev
    os.path.join(BASE_DIR, "google_shopping_brand_hits_cache.parquet"),  # Parquet for Render
    os.path.join(BASE_DIR, "google_shopping_brand_hits.xlsx"),
    DATA_PATH_ENV,
    os.path.join(BASE_DIR, "data", "google_shopping_brand_hits.xlsx"),
    os.path.join(BASE_DIR, "..", "google_shopping_brand_hits.xlsx"),
]

def _first_existing(paths):
    for p in [p for p in paths if p]:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

def _read_any(path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(path.lower())
    if ext == ".xlsx":
        sheets_dict = pd.read_excel(path, engine="openpyxl", sheet_name=None)
        all_frames = []
        for sheet_name, frame in sheets_dict.items():
            frame["Sheet"] = sheet_name
            all_frames.append(frame)
        if all_frames:
            return pd.concat(all_frames, ignore_index=True)
        else:
            return pd.DataFrame()
    elif ext == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    elif ext == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported data file type: {ext}")

def load_df() -> pd.DataFrame:
    path = _first_existing(CANDIDATES)
    if not path:
        print("WARNING: Data file not found.")
        return pd.DataFrame(columns=["Timestamp", "Keyword", "Brand", "Location", "Position", "Matched Synonym"])
    
    # Parquet cache path (same location, different extension)
    cache_path = path.rsplit('.', 1)[0] + '_cache.parquet'
    
    # Check if cache exists and is newer than source
    if os.path.exists(cache_path):
        if os.path.getmtime(cache_path) > os.path.getmtime(path):
            print(f"[dashboard] Loading from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        else:
            print(f"[dashboard] Source file updated, rebuilding cache...")
    
    # Load from source
    print(f"[dashboard] Loading from source: {path}")
    df = _read_any(path)
    
    # Save cache for next time
    try:
        df.to_parquet(cache_path, index=False)
        print(f"[dashboard] Cache saved: {cache_path}")
    except Exception as e:
        print(f"[dashboard] Warning: Could not save cache: {e}")
    
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        "Title": "Product Title",
        "Merchant Name": "Merchant",
        "Brand Domain": "Brand",
        "Link": "Product Link"
    }
    df.rename(columns=rename_map, inplace=True)
    
    for col in ["Timestamp","Sheet","Keyword","Location","Position","Product Title","Price","Price Value","Old Price Value","Rating","Reviews","Merchant","Product Link","Brand","Matched Synonym"]:
        if col not in df.columns: df[col] = pd.Series(dtype="object")
        
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for c in ["Position","Price Value","Old Price Value","Rating","Reviews"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    if "Discount %" not in df.columns:
        df["Discount %"] = np.where(
            df["Old Price Value"].gt(0) & df["Price Value"].notna() & df["Price Value"].le(df["Old Price Value"]),
            (df["Old Price Value"] - df["Price Value"]) / df["Old Price Value"] * 100.0,
            np.nan
        )
    else:
        df["Discount %"] = pd.to_numeric(df["Discount %"], errors="coerce")
    
    # Normalize Matched Synonym to lowercase for consistency
    if "Matched Synonym" in df.columns:
        df["Matched Synonym"] = df["Matched Synonym"].astype(str).str.lower().str.strip()
        df["Matched Synonym"] = df["Matched Synonym"].replace({"nan": None, "": None})

    # Convert low-cardinality string columns to categorical for memory & speed
    categorical_cols = ["Keyword", "Location", "Brand", "Matched Synonym", "Sheet", "Merchant"]
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].astype("category")

    return df

df = normalize_columns(load_df())

# -------------------- Helper Functions --------------------

def get_seller_column(consolidate: bool) -> str:
    """Return the column to use for seller grouping based on consolidation toggle."""
    return "Matched Synonym" if consolidate else "Brand"

def get_sorted_sellers(data: pd.DataFrame, seller_col: str, pinned: str = PINNED_SELLER) -> list:
    """Get sorted list of sellers with pinned seller at top."""
    unique_sellers = data[seller_col].dropna().unique()
    sellers = sorted(s for s in unique_sellers if s and str(s).lower() != "nan")

    # Find pinned seller using generator (stops at first match)
    pinned_lower = pinned.lower()
    pinned_match = next((s for s in sellers if str(s).lower() == pinned_lower), None)

    if pinned_match:
        return [pinned_match] + [s for s in sellers if s != pinned_match]
    return sellers

def get_top_n_sellers(data: pd.DataFrame, seller_col: str, metric: str, n: int, pinned: str = PINNED_SELLER) -> list:
    """Get top N sellers by specified metric, always including pinned seller."""
    if data.empty:
        return [pinned] if pinned else []

    if metric == "listings":
        ranked = data[seller_col].value_counts().head(n).index.tolist()
    elif metric == "sov":
        top3_data = data[data["Position"] <= 3]
        if top3_data.empty:
            ranked = data[seller_col].value_counts().head(n).index.tolist()
        else:
            ranked = top3_data[seller_col].value_counts().head(n).index.tolist()
    elif metric == "avg_position":
        avg_pos = data.groupby(seller_col, observed=True)["Position"].mean().sort_values()
        ranked = avg_pos.head(n).index.tolist()
    else:
        ranked = data[seller_col].value_counts().head(n).index.tolist()

    # Ensure pinned seller is included (use set for O(1) lookup)
    pinned_lower = pinned.lower()
    ranked_set = set(ranked)
    all_sellers = data[seller_col].dropna().unique()
    pinned_match = next((s for s in all_sellers if str(s).lower() == pinned_lower), None)

    if pinned_match and pinned_match not in ranked_set:
        ranked.insert(0, pinned_match)

    return ranked

def map_preset_to_sellers(data: pd.DataFrame, seller_col: str, preset_names: list) -> list:
    """Map normalized preset names to actual values in the seller column."""
    all_sellers = data[seller_col].dropna().unique()
    # Build lookup dict once: lowercase -> original value
    seller_lookup = {str(s).lower(): s for s in all_sellers}
    matched = []
    matched_set = set()

    for preset_name in preset_names:
        preset_lower = preset_name.lower()
        # Exact match first
        if preset_lower in seller_lookup:
            seller = seller_lookup[preset_lower]
            if seller not in matched_set:
                matched.append(seller)
                matched_set.add(seller)
        else:
            # Partial match fallback
            for seller_lower, seller in seller_lookup.items():
                if preset_lower in seller_lower and seller not in matched_set:
                    matched.append(seller)
                    matched_set.add(seller)

    return matched

@lru_cache(maxsize=16)
def empty_fig(msg: str) -> go.Figure:
    """Create an empty figure with a message. Cached since same messages repeat."""
    f = go.Figure()
    f.add_annotation(text=msg, showarrow=False, font=dict(size=14))
    f.update_layout(xaxis_visible=False, yaxis_visible=False, margin=dict(t=20, b=20, l=20, r=20))
    return f


def apply_global_filters(data: pd.DataFrame, kw_values, loc_values, start_date, end_date, pos_range) -> pd.DataFrame:
    """Apply global filters to dataframe. Returns a filtered view (no copy unless necessary)."""
    mask = pd.Series(True, index=data.index)

    if start_date:
        mask &= data["Timestamp"] >= pd.to_datetime(start_date)
    if end_date:
        mask &= data["Timestamp"] <= (pd.to_datetime(end_date) + pd.Timedelta(days=1))
    if kw_values:
        mask &= data["Keyword"].isin(kw_values)
    if loc_values:
        mask &= data["Location"].isin(loc_values)
    if pos_range and len(pos_range) >= 2:
        pos_min, pos_max = pos_range[0], pos_range[1]
        mask &= (data["Position"] >= pos_min) & (data["Position"] <= pos_max)

    return data.loc[mask]


# Columns needed for charts and table (reduces store payload size)
STORE_COLUMNS = [
    "Timestamp", "Sheet", "Keyword", "Location", "Position", "Product Title",
    "Price", "Price Value", "Old Price Value", "Discount %", "Rating", "Reviews",
    "Merchant", "Product Link", "Brand", "Matched Synonym"
]

# -------------------- App Layout --------------------
app = Dash(__name__, external_stylesheets=[
    THEME, 
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css"
], title="Google Shopping Dashboard", suppress_callback_exceptions=True)
server = app.server

HOVER_STYLE = {"textDecoration": "underline dotted", "cursor": "help"}

def kpi_card(title, value_component, id_suffix, tooltip_text):
    target_id = f"kpi-title-{id_suffix}"
    tooltip_id = f"kpi-tooltip-{id_suffix}"
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, id=target_id, className="kpi-sub", 
                     style={"fontSize": "0.85rem", "opacity": "0.8", **HOVER_STYLE}),
            dbc.Tooltip(tooltip_text, target=target_id, id=tooltip_id, placement="top"),
            html.Div(value_component, className="kpi", style={"fontSize": "1.5rem", "fontWeight": "bold"})
        ]),
        className="mb-3"
    )

def create_chart_filter_section(chart_id: str, chart_title: str):
    """Create a filter section for a chart with metric selector, presets, and multiselect."""
    return dbc.AccordionItem([
        # Metric selector with tooltip
        html.Div([
            html.Label("Top N by:", className="mt-2 mb-1", style={"fontSize": "0.85rem"}),
            html.I(className="bi bi-info-circle ms-2", id=f"metric-info-{chart_id}", 
                   style={"cursor": "help", "opacity": "0.7", "fontSize": "0.75rem"}),
            dbc.Tooltip(
                "Choose how sellers are ranked when using 'Top 10' or 'Top 20' presets. "
                "Listings Count = total appearances; Share of Voice = top 3 positions only; "
                "Avg Position = best average ranking (lower is better).",
                target=f"metric-info-{chart_id}",
                placement="right"
            ),
        ], className="d-flex align-items-center"),
        dcc.Dropdown(
            id={"type": "metric-dropdown", "chart": chart_id},
            options=[
                {"label": "Listings Count", "value": "listings"},
                {"label": "Share of Voice (Top 3)", "value": "sov"},
                {"label": "Avg Position (Best)", "value": "avg_position"},
            ],
            value="listings",
            clearable=False,
            style={"marginBottom": "10px"}
        ),
        
        # Preset buttons with tooltip
        html.Div([
            html.Span("Quick select:", style={"fontSize": "0.8rem", "marginRight": "8px", "opacity": "0.8"}),
            html.I(className="bi bi-info-circle", id=f"preset-info-{chart_id}", 
                   style={"cursor": "help", "opacity": "0.7", "fontSize": "0.75rem"}),
            dbc.Tooltip(
                "Top 10/20: Select top performers by the metric above. "
                "Marketplace: Priority Tire vs Walmart & eBay. "
                "Specialists: Priority Tire vs TireRack, SimpleTire, GigaTires, TiresEasy.",
                target=f"preset-info-{chart_id}",
                placement="right"
            ),
        ], className="d-flex align-items-center mb-1"),
        html.Div([
            dbc.ButtonGroup([
                dbc.Button("Top 10", id={"type": "preset-btn", "chart": chart_id, "preset": "top10"}, 
                          size="sm", outline=True, color="light", className="me-1"),
                dbc.Button("Top 20", id={"type": "preset-btn", "chart": chart_id, "preset": "top20"}, 
                          size="sm", outline=True, color="light"),
            ], className="me-2"),
            dbc.ButtonGroup([
                dbc.Button("Marketplace", id={"type": "preset-btn", "chart": chart_id, "preset": "marketplace"}, 
                          size="sm", outline=True, color="info", className="me-1"),
                dbc.Button("Specialists", id={"type": "preset-btn", "chart": chart_id, "preset": "specialists"}, 
                          size="sm", outline=True, color="success"),
            ]),
        ], className="mb-2 d-flex flex-wrap gap-1"),
        
        # Multiselect with tooltip
        html.Div([
            html.Label("Sellers:", style={"fontSize": "0.85rem"}),
            html.I(className="bi bi-info-circle ms-2", id=f"multiselect-info-{chart_id}", 
                   style={"cursor": "help", "opacity": "0.7", "fontSize": "0.75rem"}),
            dbc.Tooltip(
                "Select which sellers appear in this chart. Use presets above for quick selection, "
                "or manually add/remove sellers. Priority Tire is pinned at the top for easy access.",
                target=f"multiselect-info-{chart_id}",
                placement="right"
            ),
        ], className="d-flex align-items-center"),
        dcc.Dropdown(
            id={"type": "seller-multiselect", "chart": chart_id},
            options=[],
            value=[],
            multi=True,
            placeholder="Select sellers...",
            style={"marginTop": "5px"}
        ),
    ], title=chart_title)

def controls(data: pd.DataFrame):
    pos_vals = pd.to_numeric(data.get("Position"), errors="coerce")
    pos_min_default = int(np.nanmin(pos_vals)) if pos_vals.notna().any() else 1
    pos_max_default = int(np.nanmax(pos_vals)) if pos_vals.notna().any() else 60
    if pos_min_default == pos_max_default: pos_max_default += 1

    min_date = data["Timestamp"].min() if not data.empty else pd.Timestamp.now()
    max_date = data["Timestamp"].max() if not data.empty else pd.Timestamp.now()

    return dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label("Keyword", className="me-2"),
                    html.I(className="bi bi-info-circle", id="kw-info", 
                           style={"cursor": "help", "opacity": "0.7", "fontSize": "0.8rem"}),
                    dbc.Tooltip("Filter data by specific search keywords/queries that were tracked.", 
                               target="kw-info", placement="top"),
                ], className="d-flex align-items-center mb-1"),
                dcc.Dropdown(sorted(data["Keyword"].dropna().unique()) if not data.empty else [],
                             id="kw", multi=True)
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Label("Location", className="me-2"),
                    html.I(className="bi bi-info-circle", id="loc-info", 
                           style={"cursor": "help", "opacity": "0.7", "fontSize": "0.8rem"}),
                    dbc.Tooltip("Filter by geographic location where the search was performed.", 
                               target="loc-info", placement="top"),
                ], className="d-flex align-items-center mb-1"),
                dcc.Dropdown(sorted(data["Location"].dropna().unique()) if not data.empty else [],
                             id="loc", multi=True)
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Label("Date range", className="me-2"),
                    html.I(className="bi bi-info-circle", id="date-info", 
                           style={"cursor": "help", "opacity": "0.7", "fontSize": "0.8rem"}),
                    dbc.Tooltip("Filter data to a specific time period. Useful for comparing performance across different date ranges.", 
                               target="date-info", placement="top"),
                ], className="d-flex align-items-center mb-1"),
                dcc.DatePickerRange(id="dates", start_date=min_date, end_date=max_date)
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Label("Position range", className="me-2"),
                    html.I(className="bi bi-info-circle", id="pos-info", 
                           style={"cursor": "help", "opacity": "0.7", "fontSize": "0.8rem"}),
                    dbc.Tooltip("Filter to only show listings within a specific position range (e.g., 1-10 for top results only).", 
                               target="pos-info", placement="top"),
                ], className="d-flex align-items-center mb-1"),
                dcc.RangeSlider(id="pos", min=pos_min_default, max=pos_max_default,
                               value=[pos_min_default, pos_max_default])
            ], md=3),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                dbc.Button("Run / Update Dashboard", id="run-button", color="primary", n_clicks=0, className="w-100")
            ], md=12)
        ], className="mt-2")
    ]), className="mb-4")

def create_sidebar():
    """Create the sidebar with consolidation toggle and per-chart filters."""
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H5("Seller Filters", className="mb-0 me-2"),
                html.I(className="bi bi-info-circle", id="sidebar-info", 
                       style={"cursor": "help", "opacity": "0.7"}),
                dbc.Tooltip(
                    "Control which sellers appear in each chart independently. "
                    "Use presets for quick selection or manually pick sellers from the dropdowns. "
                    "Each chart has its own filter section.",
                    target="sidebar-info",
                    placement="right"
                ),
            ], className="d-flex align-items-center")
        ]),
        dbc.CardBody([
            # Consolidation toggle
            html.Div([
                html.Div([
                    dbc.Switch(
                        id="consolidate-toggle",
                        label="Consolidate seller variants",
                        value=True,
                        className="mb-1"
                    ),
                    html.I(className="bi bi-info-circle ms-2", id="consolidate-info", 
                           style={"cursor": "help", "opacity": "0.7"}),
                    dbc.Tooltip(
                        "When ON: Groups seller variants together (e.g., 'Priority Tire' and 'prioritytire.com' become 'priority tire'). "
                        "When OFF: Shows each seller variant separately, useful for comparing your organic listings vs marketplace storefronts.",
                        target="consolidate-info",
                        placement="right"
                    ),
                ], className="d-flex align-items-center"),
                html.Small(
                    "Groups variants like 'Priority Tire' and 'prioritytire.com' together",
                    className="text-muted d-block mb-3"
                ),
            ]),
            
            html.Hr(),
            
            # Per-chart filter sections
            dbc.Accordion([
                create_chart_filter_section("bar", "Listings by Seller"),
                create_chart_filter_section("pie", "Share of Voice"),
                create_chart_filter_section("scatter", "Price vs Position"),
                create_chart_filter_section("timeseries", "Position over Time"),
            ], id="filter-accordion", start_collapsed=False, always_open=True),
        ], style={"maxHeight": "calc(100vh - 200px)", "overflowY": "auto"})
    ])

# Main layout
app.layout = dbc.Container([
    html.Br(),
    html.H2("Google Shopping Dashboard", id="header-title"),
    html.Div("Explore placements, prices, and discounts across brands, keywords, and locations.", className="text-muted mb-3"),
    controls(df),

    # KPI Row
    dbc.Row([
        dbc.Col(kpi_card("Observations", html.Span(id="kpi-rows"), "obs", 
                        "Total number of individual product listings collected that match the current filters."), md=3),
        dbc.Col(kpi_card("Top-3 Share", html.Span(id="kpi-top3"), "top3", 
                        "Percentage of all matching listings that appear in positions 1, 2, or 3."), md=3),
        dbc.Col(kpi_card("Avg Price", html.Span(id="kpi-avgprice"), "avgp", 
                        "The average selling price across all listings matching the current filters."), md=3),
        dbc.Col(kpi_card("Avg Discount", html.Span(id="kpi-avgdisc"), "avgd", 
                        "The average percentage discount across matching listings."), md=3),
    ], className="gy-3"),

    html.Br(),
    
    # Sidebar toggle button (floating)
    html.Div([
        dbc.Button(
            html.I(className="bi bi-layout-sidebar-inset-reverse", style={"fontSize": "1.8rem"}),
            id="sidebar-toggle",
            size="sm",
            title="Toggle seller filters sidebar",
            style={
                "backgroundColor": "#F2A61D", 
                "border": "none", 
                "color": "white",
                "position": "fixed",
                "top": "385px",
                "left": "20px",
                "zIndex": "1000",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.3)",
                "borderRadius": "8px"
            }
        ),
    ]),
    
    # Main content with sidebar
    dbc.Row([
        # Sidebar (collapsible - hidden by default)
        dbc.Col(
            create_sidebar(), 
            id="sidebar-col",
            md=0,
            style={"display": "none"}
        ),
        
        # Charts (full width by default)
        dbc.Col([
            # Row 1: Bar chart and Pie chart
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Listings by Seller (All Positions)", id="tt-brand-share", style=HOVER_STYLE),
                    dbc.Tooltip(
                        "Total count of listings appearing on Google Shopping for each seller, regardless of rank position. "
                        "Higher counts indicate more product visibility. Use the sidebar filters to focus on specific competitors.",
                        target="tt-brand-share"
                    ),
                    dcc.Graph(id="fig_brand_share", style={"height": "380px"})
                ])), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Share of Voice (Top 3 Positions)", id="tt-sov", style=HOVER_STYLE),
                    dbc.Tooltip(
                        "Percentage share of only the top 3 premium slots (positions 1-3) held by each seller. "
                        "This represents the 'digital shelf' - the most valuable real estate that shoppers see first.",
                        target="tt-sov"
                    ),
                    dcc.Graph(id="fig_share_voice", style={"height": "380px"})
                ])), md=6),
            ], className="gy-3 mb-3"),
            
            # Row 2: Discount Over Time and Scatter
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Avg Discount over Time", id="tt-heat", style=HOVER_STYLE),
                    dbc.Tooltip(
                        "Trend line showing the average discount percentage for each seller over time. "
                        "Useful for spotting promotional periods and pricing strategy shifts.",
                        target="tt-heat"
                    ),
                    dcc.Graph(id="fig_heatmap", style={"height": "350px"})
                ])), md=7),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Price vs Position", id="tt-scatter", style=HOVER_STYLE),
                    dbc.Tooltip(
                        "Individual product observations plotted by their Price (Y-axis) versus their ranking Position (X-axis). "
                        "Helps identify if there's a correlation between pricing and visibility. Each dot is one product listing.",
                        target="tt-scatter"
                    ),
                    dcc.Graph(id="fig_scatter", style={"height": "350px"})
                ])), md=5),
            ], className="gy-3 mb-3"),
            
            # Row 3: Time series (full width)
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Avg Position over Time", id="tt-time", style=HOVER_STYLE),
                    dbc.Tooltip(
                        "Trend line showing the average daily ranking position for each seller over the selected time period. "
                        "Lower on the Y-axis means better rank (position 1 is best). Watch for trends and sudden changes in competitive positioning.",
                        target="tt-time"
                    ),
                    dcc.Graph(id="fig_timeseries", style={"height": "350px"})
                ])), md=12),
            ], className="gy-3"),
        ], id="charts-col", md=12),
    ]),

    html.Br(),
    
    # Results Table
    dbc.Card(dbc.CardBody([
        html.H5("Results Table"),
        dash_table.DataTable(
            id="table",
            columns=[{"name":c, "id":c} for c in ["Timestamp","Sheet","Keyword","Location"]],
            style_as_list_view=True,
            page_size=25,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#1f1f1f", "color": "white", "fontWeight": "bold", "border": "1px solid #333"},
            style_data={"backgroundColor": "white", "color": "black", "border": "1px solid #eee"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
                {"if": {"filter_query": "{Position} <= 3"}, "backgroundColor": "#d4edda", "color": "#155724"},
                {"if": {"filter_query": "{Discount %} >= 20"}, "backgroundColor": "#fff3cd", "color": "#856404"},
            ],
        )
    ]), className="mb-5"),

    dcc.Interval(id="interval", interval=AUTO_REFRESH_MS, n_intervals=0),
    
    # Store for sidebar state
    dcc.Store(id="sidebar-state", data={"visible": True}),
    
    # Store for filtered data info
    dcc.Store(id="filtered-data-store"),
], fluid=True)

# -------------------- Callbacks --------------------

# Sidebar toggle callback
@callback(
    Output("sidebar-col", "style"),
    Output("sidebar-col", "md"),
    Output("charts-col", "md"),
    Output("sidebar-toggle", "children"),
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar-col", "style"),
    prevent_initial_call=True
)
def toggle_sidebar(n_clicks, current_style):
    """Toggle sidebar visibility and adjust column widths."""
    if current_style is None:
        current_style = {"display": "block"}
    
    is_visible = current_style.get("display", "block") != "none"
    
    if is_visible:
        # Hide sidebar
        return (
            {"display": "none"},
            0,
            12,
            html.I(className="bi bi-layout-sidebar-inset-reverse", style={"fontSize": "1.8rem"})
        )
    else:
        # Show sidebar
        return (
            {"display": "block"},
            3,
            9,
            html.I(className="bi bi-layout-sidebar-inset", style={"fontSize": "1.8rem"})
        )

# Callback to update multiselect options when consolidation toggle or data filters change
@callback(
    Output({"type": "seller-multiselect", "chart": "bar"}, "options"),
    Output({"type": "seller-multiselect", "chart": "bar"}, "value"),
    Output({"type": "seller-multiselect", "chart": "pie"}, "options"),
    Output({"type": "seller-multiselect", "chart": "pie"}, "value"),
    Output({"type": "seller-multiselect", "chart": "scatter"}, "options"),
    Output({"type": "seller-multiselect", "chart": "scatter"}, "value"),
    Output({"type": "seller-multiselect", "chart": "timeseries"}, "options"),
    Output({"type": "seller-multiselect", "chart": "timeseries"}, "value"),
    Input("consolidate-toggle", "value"),
    Input("run-button", "n_clicks"),
    State("kw", "value"),
    State("loc", "value"),
    State("dates", "start_date"),
    State("dates", "end_date"),
    State("pos", "value"),
)
def update_multiselect_options(consolidate, n_clicks, kw_values, loc_values, start_date, end_date, pos_range):
    """Update all multiselect options based on consolidation toggle and filters."""
    dff = apply_global_filters(df, kw_values or [], loc_values or [], start_date, end_date, pos_range)
    seller_col = get_seller_column(consolidate)
    sellers = get_sorted_sellers(dff, seller_col)
    
    options = [{"label": s, "value": s} for s in sellers]
    
    # Default: select top 10 by listings, always including pinned seller
    default_selection = get_top_n_sellers(dff, seller_col, "listings", 10)
    
    # Return same options and defaults for all 4 charts
    return (
        options, default_selection,
        options, default_selection,
        options, default_selection,
        options, default_selection,
    )

# Pattern-matching callback for preset buttons
@callback(
    Output({"type": "seller-multiselect", "chart": MATCH}, "value", allow_duplicate=True),
    Input({"type": "preset-btn", "chart": MATCH, "preset": ALL}, "n_clicks"),
    State({"type": "metric-dropdown", "chart": MATCH}, "value"),
    State("consolidate-toggle", "value"),
    State("kw", "value"),
    State("loc", "value"),
    State("dates", "start_date"),
    State("dates", "end_date"),
    State("pos", "value"),
    prevent_initial_call=True
)
def handle_preset_click(n_clicks_list, metric, consolidate, kw_values, loc_values, start_date, end_date, pos_range):
    """Handle preset button clicks for any chart."""
    if not ctx.triggered_id or all(n is None for n in n_clicks_list):
        raise PreventUpdate

    preset = ctx.triggered_id["preset"]
    dff = apply_global_filters(df, kw_values or [], loc_values or [], start_date, end_date, pos_range)
    seller_col = get_seller_column(consolidate)
    
    if preset == "top10":
        return get_top_n_sellers(dff, seller_col, metric, 10)
    elif preset == "top20":
        return get_top_n_sellers(dff, seller_col, metric, 20)
    elif preset == "marketplace":
        return map_preset_to_sellers(dff, seller_col, PRESETS["marketplace"])
    elif preset == "specialists":
        return map_preset_to_sellers(dff, seller_col, PRESETS["specialists"])
    
    return []

# Import PreventUpdate
from dash.exceptions import PreventUpdate

# Main dashboard update callback
@callback(
    Output("filtered-data-store", "data"),
    Output("kpi-rows", "children"),
    Output("kpi-top3", "children"),
    Output("kpi-avgprice", "children"),
    Output("kpi-avgdisc", "children"),
    Input("run-button", "n_clicks"),
    Input("interval", "n_intervals"),
    Input("consolidate-toggle", "value"),  # Changed to Input so toggle triggers refresh
    State("kw", "value"),
    State("loc", "value"),
    State("dates", "start_date"),
    State("dates", "end_date"),
    State("pos", "value"),
)
def update_filtered_data(n_clicks, n_intervals, consolidate, kw_values, loc_values, start_date, end_date, pos_range):
    """Apply global filters once and store result. Updates KPIs."""
    dff = apply_global_filters(df, kw_values or [], loc_values or [], start_date, end_date, pos_range)
    seller_col = get_seller_column(consolidate)

    # Empty Data Handling
    if dff.empty:
        return None, "0", "—", "—", "—"

    # KPIs
    top3 = dff["Position"].le(3).mean() * 100
    avg_p = dff["Price Value"].mean()
    avg_d = dff["Discount %"].mean()

    kpi_rows = f"{len(dff):,}"
    kpi_top3 = f"{top3:.1f}%"
    kpi_avg_p = f"${avg_p:,.2f}" if pd.notna(avg_p) else "—"
    kpi_avg_d = f"{avg_d:.1f}%" if pd.notna(avg_d) else "—"

    # Only store columns needed for charts (reduces JSON serialization overhead)
    available_cols = [c for c in STORE_COLUMNS if c in dff.columns]
    store_data = {
        "records": dff[available_cols].to_dict("records"),
        "seller_col": seller_col
    }

    return store_data, kpi_rows, kpi_top3, kpi_avg_p, kpi_avg_d


@callback(
    Output("fig_brand_share", "figure"),
    Output("fig_share_voice", "figure"),
    Output("fig_heatmap", "figure"),
    Output("fig_scatter", "figure"),
    Output("fig_timeseries", "figure"),
    Output("table", "data"),
    Output("table", "columns"),
    Input("filtered-data-store", "data"),
    Input({"type": "seller-multiselect", "chart": "bar"}, "value"),
    Input({"type": "seller-multiselect", "chart": "pie"}, "value"),
    Input({"type": "seller-multiselect", "chart": "scatter"}, "value"),
    Input({"type": "seller-multiselect", "chart": "timeseries"}, "value"),
)
def update_charts(store_data, bar_sellers, pie_sellers, scatter_sellers, ts_sellers):
    """Generate charts from pre-filtered data. Only re-runs when filters or seller selections change."""
    
    # Handle empty/None store
    if not store_data:
        return (empty_fig("No Data"), empty_fig("No Data"), empty_fig("No Data"), 
                empty_fig("No Data"), empty_fig("No Data"), [], [])
    
    # Reconstruct dataframe from store
    dff = pd.DataFrame(store_data["records"])
    seller_col = store_data["seller_col"]
    
    # Convert Timestamp back to datetime
    if "Timestamp" in dff.columns:
        dff["Timestamp"] = pd.to_datetime(dff["Timestamp"])
    
    # Normalize seller inputs
    bar_sellers = bar_sellers or []
    pie_sellers = pie_sellers or []
    scatter_sellers = scatter_sellers or []
    ts_sellers = ts_sellers or []

    # 1. Listings by Seller (Bar Chart)
    if bar_sellers:
        bar_data = dff[dff[seller_col].isin(bar_sellers)]
    else:
        bar_data = dff
    
    if not bar_data.empty:
        bar_counts = bar_data[seller_col].value_counts().reset_index()
        bar_counts.columns = [seller_col, "count"]
        brand_fig = px.bar(bar_counts, x=seller_col, y="count")
        brand_fig.update_layout(margin=dict(t=20, b=80, l=40, r=20), xaxis_tickangle=-45)
    else:
        brand_fig = empty_fig("No Data for Selected Sellers")

    # 2. Share of Voice (Pie Chart)
    if pie_sellers:
        pie_data = dff[dff[seller_col].isin(pie_sellers)]
    else:
        pie_data = dff
    
    top_shelf = pie_data[pie_data["Position"] <= 3]
    if not top_shelf.empty:
        share_counts = top_shelf[seller_col].value_counts().reset_index()
        share_counts.columns = [seller_col, "Count"]
        share_fig = px.pie(share_counts, values="Count", names=seller_col, hole=0.5)
        share_fig.update_traces(textinfo='percent+label')
        share_fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
    else:
        share_fig = empty_fig("No Top 3 Data")

    # 3. Avg Discount over Time (uses bar_sellers for filtering)
    heat_fig = empty_fig("No Data")
    if "Discount %" in dff.columns and "Timestamp" in dff.columns:
        disc_data = dff[dff[seller_col].isin(bar_sellers)] if bar_sellers else dff
        # Filter to rows that have a discount
        disc_data = disc_data[disc_data["Discount %"].notna()]
        if not disc_data.empty:
            grp = disc_data.groupby([pd.Grouper(key="Timestamp", freq="D"), seller_col], observed=True)["Discount %"].mean().reset_index()
            if not grp.empty:
                heat_fig = px.line(grp, x="Timestamp", y="Discount %", color=seller_col)
                heat_fig.update_layout(margin=dict(t=20, b=60, l=40, r=20),
                                      legend=dict(orientation="h", y=-0.25, yanchor="top"))

    # 4. Scatter (Price vs Position) - with WebGL for performance
    if scatter_sellers:
        scatter_data = dff[dff[seller_col].isin(scatter_sellers)]
    else:
        scatter_data = dff
    
    if not scatter_data.empty:
        if len(scatter_data) > 2000:
            scatter_sample = scatter_data.sample(n=2000, random_state=42)
        else:
            scatter_sample = scatter_data
        scatter_fig = px.scatter(scatter_sample, x="Position", y="Price Value", color=seller_col,
                                  render_mode='webgl')
        scatter_fig.update_layout(margin=dict(t=20, b=20, l=40, r=20), legend=dict(orientation="h", y=-0.2))
    else:
        scatter_fig = empty_fig("No Data for Selected Sellers")

    # 5. Time Series
    if ts_sellers:
        ts_data = dff[dff[seller_col].isin(ts_sellers)]
    else:
        ts_data = dff
    
    if not ts_data.empty:
        grp = ts_data.groupby([pd.Grouper(key="Timestamp", freq="D"), seller_col], observed=True)["Position"].mean().reset_index()
        ts_fig = px.line(grp, x="Timestamp", y="Position", color=seller_col)
        ts_fig.update_layout(yaxis_autorange="reversed", margin=dict(t=20, b=60, l=40, r=20),
                            legend=dict(orientation="h", y=-0.25, yanchor="top"))
    else:
        ts_fig = empty_fig("No Data for Selected Sellers")

    # Table
    cols = ["Timestamp", "Sheet", "Keyword", "Location", seller_col, "Product Title", "Price", 
            "Price Value", "Old Price Value", "Discount %", "Position", "Product Link"]
    available_cols = [c for c in cols if c in dff.columns]
    final_cols = [{"name": c, "id": c, 
                   "type": "numeric" if c in ["Price Value", "Position", "Discount %"] else "text",
                   "presentation": "markdown" if c == "Product Link" else "input"} 
                  for c in available_cols]
    
    return (
        brand_fig, share_fig, heat_fig, scatter_fig, ts_fig,
        dff[available_cols].to_dict("records"), final_cols
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)