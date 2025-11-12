#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Google Shopping brand-hit tracker (multi-sheet in/out, multi-location).
- Loads SERPAPI_KEY from env (or .env if python-dotenv is installed)
- Reads keywords from Excel/CSV. If Excel with multiple sheets: each sheet is processed separately.
- Queries SerpAPI "google_shopping" for each keyword × each location
- Matches results by brand synonyms against a brand map:
    * Built-in DEFAULT_BRANDS_MAP (dict in code; previous behavior restored)
    * Optional external JSON override (same shape), if BRANDS_JSON_PATH points to a file
- Writes one output XLSX: one sheet per input sheet, keeping sheet names.
- If output XLSX is locked (WinError 32), falls back to CSV files (one per sheet).

Notes
- You’ll need a SerpAPI key (env var SERPAPI_KEY).
- Locations are Google Shopping uule/cid engine parameters handled via SerpAPI’s "location" parameter.
- This script is intentionally verbose & defensive, to avoid silent failures.

Author: (Your name)
"""

from __future__ import annotations

import os
import re
import io
import sys
import json
import time
import math
import copy
import csv
import uuid
import atexit
import random
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --- Optional .env loader (safe if not installed) ---
try:
    from pathlib import Path
    from dotenv import load_dotenv  # type: ignore
    # 1) load .env next to this file; 2) also load from CWD as fallback
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
    load_dotenv()
except Exception:
    pass

import requests
import pandas as pd

# ----------------------------
# Configuration (edit as needed)
# ----------------------------

# Path to your keywords file:
# - If Excel: multi-sheet supported. Every sheet must have a column named "keyword".
# - If CSV: single sheet; the sheet name will be "Sheet1" for output.
KEYWORDS_FILE = os.environ.get("KEYWORDS_FILE", "keywords_test.xlsx")  # you can set to "keywords_test.xlsx" while testing

# Optional explicit list of sheets to use (comma-separated), e.g. "sizes,brands"
# If not set, process all sheets (Excel) or the single CSV.
SHEETS = [s.strip() for s in os.environ.get("KEYWORDS_SHEETS", "").split(",") if s.strip()]

# Output file name (XLSX)
OUTPUT_XLSX = os.environ.get("OUTPUT_XLSX", "google_shopping_brand_hits.xlsx")

# Optional CSV fallback directory (used if XLSX is locked)
CSV_FALLBACK_DIR = os.environ.get("CSV_FALLBACK_DIR", "brand_hits_csv")

# SerpAPI key must be present
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Engine config
SERP_ENGINE = "google_shopping"
GOOGLE_DOMAIN = os.environ.get("GOOGLE_DOMAIN", "google.com")
GL = os.environ.get("GL", "us")      # country
HL = os.environ.get("HL", "en")      # language

# Delay between calls to be polite (seconds)
SLEEP_BETWEEN_CALLS = float(os.environ.get("SLEEP_BETWEEN_CALLS", "1.0"))

# Per-keyword maximum shopping results pulled (SerpAPI supports "num" up to ~100)
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "50"))

# Optional external brands JSON override path. If provided & valid, will override the default dict below.
BRANDS_JSON_PATH = os.environ.get("BRANDS_JSON_PATH", "").strip() or None

# Restore in-code dict loading: minimal, you can expand it. External JSON still can override this.
DEFAULT_BRANDS_MAP: Dict[str, List[str]] = {
    "prioritytire.com": ["priority tire", "prioritytire"],
    "simpletire.com":   ["simple tire", "simpletire"],
    # Feel free to add more here if desired (fallback only)
    # "tirerack.com": ["tire rack", "tirerack"],
    # "discounttire.com": ["discount tire", "discounttire"],
}

# Locations to query (SerpAPI "location" parameter strings).
# You can add more. Each is a user-friendly label and a SerpAPI "location".
LOCATIONS: List[Tuple[str, str]] = [
    ("Florida, United States", "Florida, United States"),
    ("Los Angeles, California, United States", "Los Angeles, California, United States"),
]

# Logging level
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# If you want to throttle failures
MAX_API_RETRIES = int(os.environ.get("MAX_API_RETRIES", "2"))
RETRY_BACKOFF_BASE = float(os.environ.get("RETRY_BACKOFF_BASE", "1.6"))  # exp backoff base

# Column names used for writing results
COLS = [
    "Timestamp",
    "Sheet",
    "Keyword",
    "Location",
    "Position",
    "Title",
    "Price",
    "Price Value",
    "Old Price",
    "Old Price Value",
    "Rating",
    "Reviews",
    "Merchant Name",
    "Product Link",
    "Brand Match Domain",
    "Matched Synonym",
]

# ------------- Logging setup -------------
logger = logging.getLogger("tracker")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
logger.addHandler(_handler)

# ------------ Utilities ------------

def _now() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def _normalize_price_str(p: Optional[str]) -> Tuple[str, Optional[float]]:
    """
    Try to retain the price string and extract a numeric float if possible.
    Examples:
        "$123.45" -> ("$123.45", 123.45)
        "US$1,199.00" -> ("US$1,199.00", 1199.00)
        None -> ("", None)
    """
    if not p:
        return "", None
    s = str(p)
    # keep original
    raw = s
    # strip non-digits except dot and comma
    s2 = re.sub(r"[^0-9.,]", "", s)
    # If there are commas, assume US formatting: remove commas then parse float
    try:
        if s2.count(",") and s2.count("."):
            # "1,299.99"
            s3 = s2.replace(",", "")
        else:
            # Could be "1.234,56" (EU) – handle basic swap if comma as decimal
            if s2.count(",") == 1 and s2.count(".") == 0:
                s3 = s2.replace(",", ".")
            else:
                s3 = s2.replace(",", "")
        val = float(s3)
        return raw, val
    except Exception:
        return raw, None

def _ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def _is_excel(fname: str) -> bool:
    low = fname.lower()
    return low.endswith(".xlsx") or low.endswith(".xlsm") or low.endswith(".xls")

def _is_csv(fname: str) -> bool:
    low = fname.lower()
    return low.endswith(".csv")

def _read_keywords_multi_sheet(path: str, only_sheets: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Returns {sheet_name: [keyword1, keyword2, ...]}
    If CSV, will use pseudo-sheet "Sheet1".
    Requires a 'keyword' column in each sheet/CSV.
    """
    store: Dict[str, List[str]] = {}
    if _is_excel(path):
        xl = pd.ExcelFile(path)
        candidate_sheets = only_sheets or xl.sheet_names
        for sh in candidate_sheets:
            df = pd.read_excel(path, sheet_name=sh)
            cols = [c.strip().lower() for c in df.columns]
            if "keyword" not in cols:
                logger.warning(f"Sheet '{sh}' has no 'keyword' column; skipping.")
                continue
            kw_col = df.columns[cols.index("keyword")]
            kws = [str(x).strip() for x in df[kw_col].dropna().astype(str).tolist() if str(x).strip()]
            if not kws:
                logger.info(f"Sheet '{sh}' contains 0 keywords; skipping.")
                continue
            store[sh] = kws
    elif _is_csv(path):
        df = pd.read_csv(path)
        cols = [c.strip().lower() for c in df.columns]
        if "keyword" not in cols:
            raise ValueError("CSV has no 'keyword' column.")
        kw_col = df.columns[cols.index("keyword")]
        kws = [str(x).strip() for x in df[kw_col].dropna().astype(str).tolist() if str(x).strip()]
        store["Sheet1"] = kws
    else:
        raise FileNotFoundError(f"Unsupported keywords file: {path}")
    total = sum(len(v) for v in store.values())
    logger.info(f"Loaded {total} keywords across {len(store)} sheet(s)")
    return store

# ------------ Brands map ------------

def load_brands_map(brands_json_path: Optional[str]) -> Dict[str, List[str]]:
    """
    Priority:
      A) If brands_json_path is provided & valid JSON mapping {domain: [synonyms...]}, use it
      B) Else use DEFAULT_BRANDS_MAP (dict in code; restored behavior)
    """
    if brands_json_path:
        p = pathlib.Path(brands_json_path)
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
                    return data
                else:
                    logger.warning(f"Invalid brands JSON at '{brands_json_path}'; using default in-code dict.")
            except Exception as e:
                logger.warning(f"Failed reading brands JSON '{brands_json_path}': {e}; using default in-code dict.")
    return DEFAULT_BRANDS_MAP

# Build a compiled synonym matcher
def compile_brand_matchers(brands_map: Dict[str, List[str]]) -> List[Tuple[str, re.Pattern]]:
    """
    Returns list of (domain, compiled_regex) where the regex matches any synonym word boundary-insensitive.
    """
    compiled: List[Tuple[str, re.Pattern]] = []
    for domain, synonyms in brands_map.items():
        # turn synonyms into "word-ish" regex (space-insensitive-ish)
        escaped = [re.escape(s.strip()) for s in synonyms if s.strip()]
        if not escaped:
            continue
        pattern = r"(?i)\b(" + "|".join(escaped) + r")\b"
        compiled.append((domain, re.compile(pattern)))
    return compiled

# ------------ SerpAPI client ------------

class SerpApiError(Exception):
    pass

def serp_shopping_search(
    query: str,
    location: str,
    api_key: str,
    num: int = 50,
    gl: str = "us",
    hl: str = "en",
    google_domain: str = "google.com",
    retries: int = 2,
    backoff_base: float = 1.6,
) -> Dict[str, Any]:
    """
    Calls SerpAPI "google_shopping" engine and returns raw JSON.
    Retries on 429/5xx with exponential backoff.
    """
    url = "https://serpapi.com/search.json"
    params = {
        "engine": SERP_ENGINE,
        "q": query,
        "location": location,
        "google_domain": google_domain,
        "gl": gl,
        "hl": hl,
        "num": max(10, min(int(num), 100)),
        "api_key": api_key,
    }

    for attempt in range(1, max(1, retries) + 1):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                delay = (backoff_base ** (attempt - 1)) + random.uniform(0, 1.2)
                logger.warning(f"SerpAPI HTTP {r.status_code} (attempt {attempt}/{retries}); sleeping {delay:.1f}s …")
                time.sleep(delay)
                continue
            raise SerpApiError(f"SerpAPI HTTP {r.status_code}: {r.text[:240]}")
        except requests.RequestException as e:
            delay = (backoff_base ** (attempt - 1)) + random.uniform(0, 1.2)
            logger.warning(f"SerpAPI error {e!r} (attempt {attempt}/{retries}); sleeping {delay:.1f}s …")
            time.sleep(delay)
    raise SerpApiError("SerpAPI request failed after retries.")

# ------------ Result parsing & matching ------------

def extract_shopping_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract relevant fields from SerpAPI shopping JSON.
    """
    results: List[Dict[str, Any]] = []
    if not payload:
        return results

    # SerpAPI returns "shopping_results" list
    items = payload.get("shopping_results") or []
    for idx, it in enumerate(items, 1):
        title = it.get("title") or ""
        link = it.get("link") or it.get("product_link") or ""
        merchant = it.get("source") or it.get("store") or it.get("seller") or ""
        price_str = it.get("price") or it.get("extracted_price") or ""
        extracted_price = it.get("extracted_price")
        if extracted_price is None:
            _, price_val = _normalize_price_str(price_str if isinstance(price_str, str) else str(price_str))
        else:
            price_val = float(extracted_price)

        old_price_str = it.get("original_price") or it.get("old_price") or ""
        old_price_val = None
        if old_price_str:
            _, old_price_val = _normalize_price_str(str(old_price_str))

        rating = it.get("rating")
        reviews = it.get("reviews")

        results.append({
            "Position": idx,
            "Title": title,
            "Product Link": link,
            "Merchant Name": merchant,
            "Price": price_str if isinstance(price_str, str) else (f"${price_val:.2f}" if price_val else ""),
            "Price Value": price_val,
            "Old Price": old_price_str if isinstance(old_price_str, str) else (f"${old_price_val:.2f}" if old_price_val else ""),
            "Old Price Value": old_price_val,
            "Rating": rating if isinstance(rating, (int, float)) else None,
            "Reviews": reviews if isinstance(reviews, (int, float)) else None,
        })
    return results

def match_brand(row: Dict[str, Any], brand_patterns: List[Tuple[str, re.Pattern]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Try matching any brand synonym in the Title or Merchant fields, returns (domain, matched_synonym) or (None, None).
    """
    hay = " ".join([
        str(row.get("Title", "") or ""),
        str(row.get("Merchant Name", "") or "")
    ]).lower()

    for domain, creg in brand_patterns:
        m = creg.search(hay)
        if m:
            return domain, m.group(1)
    return None, None

# ------------ Writer helpers ------------

def write_xlsx_per_sheet(out_path: str, sheet2df: Dict[str, pd.DataFrame]) -> None:
    """
    Write one DataFrame per sheet into a single XLSX.
    """
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
        for sheet, df in sheet2df.items():
            # Trim Excel sheet name safely
            safe_sheet = sheet[:31] if sheet else "Sheet1"
            # Avoid empty sheet names duplication
            if not safe_sheet:
                safe_sheet = "Sheet1"
            df.to_excel(xw, sheet_name=safe_sheet, index=False)

def write_csv_fallback(base_dir: str, sheet2df: Dict[str, pd.DataFrame]) -> None:
    _ensure_dir(base_dir)
    for sheet, df in sheet2df.items():
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sheet)[:64] or "Sheet1"
        p = pathlib.Path(base_dir) / f"{safe}.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")

# ------------ Main orchestration ------------

def run() -> None:
    # Sanity check for key
    if not SERPAPI_KEY:
        logger.error("Missing SERPAPI_KEY. Set it in your environment or .env file.")
        return

    # Load brands map (JSON override optional)
    brands_map = load_brands_map(BRANDS_JSON_PATH)
    brand_patterns = compile_brand_matchers(brands_map)

    # Read keywords grouped by sheet
    if not pathlib.Path(KEYWORDS_FILE).is_file():
        logger.error(f"Keywords file not found: {KEYWORDS_FILE}")
        return

    # Multi-sheet
    sheets_kws = _read_keywords_multi_sheet(KEYWORDS_FILE, SHEETS if SHEETS else None)

    # Multi-location logging
    logger.info(f"📍 Multi-location mode: {len(LOCATIONS)} locations")
    for label, code in LOCATIONS:
        # SerpAPI logs a resolved "location" string; we echo our human label
        logger.info(f"   - {label} (code: {hashlib_stub(code)})")

    # Process each sheet independently; accumulate per sheet
    from collections import OrderedDict
    out_frames: "OrderedDict[str, pd.DataFrame]" = OrderedDict()

    total_keywords = sum(len(v) for v in sheets_kws.values())
    logger.info(f"Loaded {total_keywords} keywords across {len(sheets_kws)} sheet(s)")

    for sheet_name, keywords in sheets_kws.items():
        logger.info(f"=== Processing sheet: {sheet_name} ({len(keywords)} keywords) ===")
        rows: List[Dict[str, Any]] = []

        for idx, kw in enumerate(keywords, 1):
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"[{idx}/{len(keywords)}] {kw}")
            logger.info("=" * 60)

            for loc_label, loc_string in LOCATIONS:
                logger.info(f"📍 Location: {loc_label}")

                # Call SerpAPI
                try:
                    payload = serp_shopping_search(
                        query=kw,
                        location=loc_string,
                        api_key=SERPAPI_KEY,
                        num=MAX_RESULTS,
                        gl=GL,
                        hl=HL,
                        google_domain=GOOGLE_DOMAIN,
                        retries=MAX_API_RETRIES,
                        backoff_base=RETRY_BACKOFF_BASE,
                    )
                except SerpApiError as e:
                    logger.error(str(e))
                    logger.error("API error; skipping.")
                    time.sleep(SLEEP_BETWEEN_CALLS)
                    continue

                # Parse results
                items = extract_shopping_results(payload)
                if not items:
                    logger.info("⚠️  No shopping results found in this location.")
                    time.sleep(SLEEP_BETWEEN_CALLS)
                    continue

                # Filter by brand synonyms
                matched_count = 0
                for it in items:
                    domain, syn = match_brand(it, brand_patterns)
                    if not domain:
                        # Not from a targeted brand; skip row
                        continue
                    matched_count += 1

                    out = {
                        "Timestamp": _now(),
                        "Sheet": sheet_name,
                        "Keyword": kw,
                        "Location": loc_label,
                        "Position": it.get("Position"),
                        "Title": it.get("Title"),
                        "Price": it.get("Price"),
                        "Price Value": it.get("Price Value"),
                        "Old Price": it.get("Old Price"),
                        "Old Price Value": it.get("Old Price Value"),
                        "Rating": it.get("Rating"),
                        "Reviews": it.get("Reviews"),
                        "Merchant Name": it.get("Merchant Name"),
                        "Product Link": it.get("Product Link"),
                        "Brand Match Domain": domain,
                        "Matched Synonym": syn,
                    }
                    rows.append(out)

                if matched_count == 0:
                    logger.info("⚠️  No matching brand products found in this location.")

                # Polite delay between keyword×location calls
                time.sleep(SLEEP_BETWEEN_CALLS)

        # Build DataFrame for this sheet
        df_sheet = pd.DataFrame(rows, columns=COLS) if rows else pd.DataFrame(columns=COLS)
        out_frames[sheet_name] = df_sheet

    # Attempt to write XLSX (one sheet per input sheet)
    try:
        write_xlsx_per_sheet(OUTPUT_XLSX, out_frames)
        if all(df.empty for df in out_frames.values()):
            logger.info("No rows collected; nothing to write.")
        else:
            logger.info(f"Results written to: {OUTPUT_XLSX}")
    except PermissionError as e:
        logger.error(f"Failed to write XLSX; falling back to single CSV: {e}")
        write_csv_fallback(CSV_FALLBACK_DIR, out_frames)
        logger.info("CSV fallback written (one CSV per sheet).")

# small helper to avoid logging full location encoding (cosmetics)
def hashlib_stub(s: str) -> str:
    try:
        import hashlib
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:24]
    except Exception:
        return "location"

if __name__ == "__main__":
    run()
