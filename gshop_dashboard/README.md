# Google Shopping Dashboard (Dash)

A sleek Dash (Plotly) dashboard to explore your `google_shopping_brand_hits.csv` file produced by your scraper.

## Quickstart

```bash
cd gshop_dashboard
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Put your CSV here: data/google_shopping_brand_hits.csv
# (Or keep the included sample to test it.)

python app.py
# open http://127.0.0.1:8050
```
