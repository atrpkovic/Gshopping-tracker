#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Shopping Brand Position + Price/Discount Tracker
------------------------------------------------------
- Tracks brand placements for Google Shopping by keyword
- Captures price, old price, and discount %
- Runs across multiple locations (6 states + 12 metros)

Requirements:
  pip install python-dotenv requests

Environment:
  SERPAPI_KEY=<your_api_key>  # in .env (same folder) or OS env

Inputs:
  keywords.csv      -> one keyword per line
  brands.json       -> {"branddomain.com": ["Brand Name", "Brand Alias", ...], ...}

Output:
  google_shopping_brand_hits.csv

Notes:
  - Uses SerpApi's google_shopping engine
  - Prefers SerpApi location_code (resolved automatically); falls back to location name
"""

import json
import os
from dotenv import load_dotenv
import requests
import csv
from datetime import datetime, timezone
import time
import random
from urllib.parse import urlparse, parse_qs
from typing import Optional, List, Dict, Tuple
import logging
import re

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Config --------------------
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

KEYWORDS_FILE = "keywords.csv"          # one keyword per line
BRANDS_FILE   = "brands.json"           # {"domain.com": ["Brand", "Alias1", "Alias2"]}
OUTPUT_FILE   = "google_shopping_brand_hits.csv"

SAVE_JSON = False   # set True to save raw SerpApi responses for debugging

# Rate limiting between keyword queries
MIN_DELAY = 2.0
MAX_DELAY = 5.0

# Optional single-location fallback (used only if MULTI_LOCATIONS is empty)
LOCATION      = None        # e.g., "Miami, Florida, United States"
LOCATION_CODE = None        # e.g., 1021999

COUNTRY  = "us"   # gl
CURRENCY = "USD"

# -------------------- Multi-location coverage --------------------
# 6 states + 12 metros
MULTI_LOCATIONS = [
    # States
    {"name": "Florida, United States"},
    #{"name": "Illinois, United States"},
    #{"name": "North Carolina, United States"},
    #{"name": "Texas, United States"},
    #{"name": "California, United States"},
    #{"name": "Pennsylvania, United States"},

    # Metros
    {"name": "Miami, Florida, United States"},
    {"name": "Charlotte, North Carolina, United States"},
    #{"name": "Philadelphia, Pennsylvania, United States"},
    #{"name": "Pittsburgh, Pennsylvania, United States"},
    #{"name": "New York, New York, United States"},
    #{"name": "Chicago, Illinois, United States"},
    #{"name": "Houston, Texas, United States"},
    #{"name": "Austin, Texas, United States"},
    #{"name": "Dallas, Texas, United States"},
    {"name": "Los Angeles, California, United States"},
    #{"name": "San Francisco, California, United States"}
]

# -------------------- Helpers: timestamps, filenames --------------------
def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _safe_name(s: str, limit=40) -> str:
    return "".join(c for c in s.replace(" ", "_")[:limit] if c.isalnum() or c in ("_", "-"))

def save_json(data, label, keyword):
    if not SAVE_JSON:
        return None
    fname = f"shopping_{label}_{_safe_name(keyword)}_{_stamp()}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON: {fname}")
    return fname

# -------------------- Helpers: price parsing & discount detection --------------------
_price_num_re = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_percent_re   = re.compile(r"(\d{1,3})\s*%")

def parse_price_value(value) -> Optional[float]:
    """
    Extract a float from numeric or string price value (e.g., "$1,299.99").
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    s = str(value)
    m = _price_num_re.search(s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def get_price_fields(item: dict) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Return (display_price_str, price_val, old_price_val).
    Uses SerpApi fields: price, extracted_price, sale_price, extracted_sale_price,
    old_price, extracted_old_price
    """
    price_str = item.get("price")
    price_val = (
        parse_price_value(item.get("extracted_price")) or
        parse_price_value(price_str)
    )
    old_price_val = (
        parse_price_value(item.get("extracted_old_price")) or
        parse_price_value(item.get("old_price")) or
        None
    )
    # Prefer sale price if present and lower
    sale_price_val = (
        parse_price_value(item.get("extracted_sale_price")) or
        parse_price_value(item.get("sale_price")) or
        None
    )
    if sale_price_val and (not price_val or sale_price_val < price_val):
        price_val = sale_price_val
    return price_str, price_val, old_price_val

def detect_discount_percent(item: dict, price_val: Optional[float], old_price_val: Optional[float]) -> Optional[float]:
    """
    Calculate discount% primarily from old vs current price.
    Fallback to parsing extensions texts like 'Save 20%'.
    """
    # from old vs price
    if old_price_val and price_val and old_price_val > 0 and price_val < old_price_val:
        pct = round((old_price_val - price_val) / old_price_val * 100.0, 2)
        if pct > 0:
            return pct

    # from extensions text
    exts = item.get("extensions") or []
    best_pct = None
    for ext in exts:
        if not isinstance(ext, str):
            continue
        for m in _percent_re.finditer(ext):
            try:
                val = float(m.group(1))
                # sanity filter
                if 0 < val <= 95:
                    best_pct = max(best_pct, val) if best_pct is not None else val
            except Exception:
                pass
    return best_pct

# -------------------- Helpers: merchant extraction --------------------
def normalize_host(u: str) -> str:
    """
    Return normalized host (strip www.)
    """
    try:
        host = urlparse(u).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""

def extract_merchant_from_url(url: str) -> Optional[str]:
    """
    Derive merchant domain from product link.
    - If direct merchant URL, return its host
    - If google redirect (/url?url=...), extract the 'url' target host
    """
    try:
        parsed = urlparse(url)
        # direct merchant
        if parsed.netloc and not parsed.netloc.endswith("google.com"):
            return normalize_host(url)
        # google redirect
        qs = parse_qs(parsed.query)
        if "url" in qs and qs["url"]:
            return normalize_host(qs["url"][0])
    except Exception:
        pass
    return None

# -------------------- Brand matching + row building --------------------
def find_brands_in_shopping_results(shopping_results: List[Dict], brands_file: str, keyword: str, location_name: str = "Unknown") -> List[list]:
    """
    Match your brands in Shopping results, capture price/discount, and build CSV rows.

    Row format:
      [Timestamp, Keyword, Location, Brand Domain, Product Title,
       Price (display), Price Value, Old Price Value, Discount %,
       Product Link, Merchant Name, Position, Rating, Reviews]
    """
    with open(brands_file, "r", encoding="utf-8") as f:
        brands = json.load(f)

    # domain -> [domain, alias1, alias2, ...] all lowercased
    brand_map = {
        domain.lower(): [domain.lower()] + [a.lower() for a in aliases]
        for domain, aliases in brands.items()
    }

    ts = _now_ts()
    rows: List[List] = []

    for idx, item in enumerate(shopping_results, 1):
        product_title = item.get("title", "")
        link          = item.get("link", "")
        source        = item.get("source", "")
        rating        = item.get("rating", 0)
        reviews       = item.get("reviews", 0)

        price_str, price_val, old_price_val = get_price_fields(item)
        discount_pct = detect_discount_percent(item, price_val, old_price_val)

        merchant_domain = None

        # Method 1: merchant "source" text
        if source:
            sl = source.lower()
            for domain, terms in brand_map.items():
                if any(term in sl for term in terms):
                    merchant_domain = domain
                    break

        # Method 2: from product link
        if not merchant_domain and link:
            extracted_domain = extract_merchant_from_url(link)
            if extracted_domain:
                for domain, terms in brand_map.items():
                    core = domain[4:] if domain.startswith("www.") else domain
                    if extracted_domain.endswith(core):
                        merchant_domain = domain
                        break

        # Method 3: alias in product title
        if not merchant_domain:
            tl = product_title.lower()
            for domain, terms in brand_map.items():
                # skip index 0 (domain), check aliases only
                if any(term in tl for term in terms[1:]):
                    merchant_domain = domain
                    break

        if merchant_domain:
            rows.append([
                ts,
                keyword,
                location_name,
                merchant_domain,
                product_title[:200],
                price_str if price_str is not None else "",
                f"{price_val:.2f}" if isinstance(price_val, (int, float)) else "",
                f"{old_price_val:.2f}" if isinstance(old_price_val, (int, float)) else "",
                f"{discount_pct:.2f}" if isinstance(discount_pct, (int, float)) else "",
                link,
                source,
                idx,
                rating,
                reviews
            ])
            logger.debug(f"Pos {idx} | {merchant_domain} | ${price_val} old=${old_price_val} disc={discount_pct}% | {product_title[:60]}")

    return rows

# -------------------- Location utilities --------------------
def get_location_code(location_name: str) -> Optional[int]:
    """
    Query SerpApi Locations API for best-matching location code by name.
    """
    try:
        resp = requests.get(
            "https://serpapi.com/locations.json",
            params={"q": location_name, "limit": 5},
            timeout=12
        )
        data = resp.json()
        if data:
            return data[0].get("id")
    except Exception as e:
        logger.warning(f"Location code fetch failed for '{location_name}': {e}")
    return None

def resolve_location_codes(locations: List[Dict]) -> None:
    """
    Fill 'code' for any location lacking it. Fallback to name targeting if not found.
    """
    for loc in locations:
        if not loc.get("code"):
            code = get_location_code(loc["name"])
            if code:
                loc["code"] = code
                logger.info(f"📍 Resolved '{loc['name']}' → code {code}")
            else:
                logger.info(f"📍 Using name fallback for '{loc['name']}' (no code found)")

# -------------------- SerpApi fetch --------------------
def fetch_google_shopping_serpapi(keyword: str, location_override: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
    """
    Fetch Shopping results for a given keyword and location (name or code).
    Attaches _location_name/_location_code to the returned dict.
    """
    if not SERPAPI_KEY:
        logger.error("Missing SERPAPI_KEY. Set it in your environment or .env file.")
        return None

    # Determine location
    if location_override:
        loc_name = location_override.get("name")
        loc_code = location_override.get("code")
    elif LOCATION_CODE:
        loc_name = f"Code {LOCATION_CODE}"
        loc_code = LOCATION_CODE
    else:
        loc_name = LOCATION or "United States"
        loc_code = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"SerpApi Shopping [{loc_name}] {attempt}/{max_retries}: {keyword}")
            params = {
                "engine": "google_shopping",
                "q": keyword,
                "api_key": SERPAPI_KEY,
                "hl": "en",
                "gl": COUNTRY,
                "currency": CURRENCY,
                "num": 60,
                "no_cache": "true",
            }
            if loc_code:
                params["location_code"] = loc_code
            else:
                params["location"] = loc_name

            resp = requests.get("https://serpapi.com/search", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                logger.error(f"SerpApi error: {data['error']}")
                if attempt < max_retries:
                    time.sleep(4)
                    continue
                return None

            data["_location_name"] = loc_name if loc_code else (loc_name or "United States")
            data["_location_code"] = loc_code
            return data

        except requests.exceptions.Timeout:
            logger.warning("Timeout; retrying...")
            if attempt < max_retries:
                time.sleep(4)
                continue
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            if attempt < max_retries:
                time.sleep(4)
                continue
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break

    return None

# -------------------- Orchestration per keyword --------------------
def process_keyword(keyword: str, brands_file: str, location_override: Optional[Dict] = None) -> Tuple[List[list], str]:
    """
    Return (rows, status) for a single keyword + location.
    """
    data = fetch_google_shopping_serpapi(keyword, location_override)
    if not data:
        return [], "API_ERROR"

    location_name = data.get("_location_name", "Unknown")

    if SAVE_JSON:
        save_json(data, "full_response", keyword)

    shopping_results = data.get("shopping_results", [])
    if not shopping_results:
        logger.info(f"No shopping results for: {keyword}")
        return [], "NO_SHOPPING"

    rows = find_brands_in_shopping_results(shopping_results, brands_file, keyword, location_name)
    return rows, "SUCCESS"

# -------------------- Account check --------------------
def check_serpapi_account():
    try:
        resp = requests.get("https://serpapi.com/account", params={"api_key": SERPAPI_KEY}, timeout=10)
        data = resp.json()
        if "error" in data:
            logger.error(f"Account check failed: {data['error']}")
            return None
        logger.info(f"📊 SerpApi Plan: {data.get('plan_name','?')} | Searches left: {data.get('total_searches_left','?')}")
        return data
    except Exception as e:
        logger.warning(f"Could not check account: {e}")
        return None

# -------------------- Runner --------------------
def run():
    logger.info("🛍️ Google Shopping Brand + Price/Discount Tracker")

    # Validate mandatory files
    if not os.path.exists(KEYWORDS_FILE):
        logger.error(f"Missing keywords file: {KEYWORDS_FILE}")
        return
    if not os.path.exists(BRANDS_FILE):
        logger.error(f"Missing brands file: {BRANDS_FILE}")
        return

    check_serpapi_account()

    # Resolve location codes (optional but recommended)
    if MULTI_LOCATIONS:
        logger.info(f"📍 Multi-location mode: {len(MULTI_LOCATIONS)} locations")
        resolve_location_codes(MULTI_LOCATIONS)
        for loc in MULTI_LOCATIONS:
            logger.info(f"   - {loc['name']} (code: {loc.get('code','N/A')})")
    else:
        if LOCATION_CODE:
            logger.info(f"📍 Single location code: {LOCATION_CODE}")
        else:
            logger.info(f"📍 Single location: {LOCATION or 'United States'}")

    # Load keywords
    with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
        keywords = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(keywords)} keywords")

    # Initialize CSV (fresh file)
    header = [
        "Timestamp",
        "Keyword",
        "Location",
        "Brand Domain",
        "Product Title",
        "Price",             # display string
        "Price Value",       # numeric
        "Old Price Value",   # numeric
        "Discount %",        # numeric
        "Product Link",
        "Merchant Name",
        "Position",
        "Rating",
        "Reviews"
    ]
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    with open(OUTPUT_FILE, mode="w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerow(header)

    stats = {"total": len(keywords), "success": 0, "no_shopping": 0, "api_error": 0, "products_found": 0}

    # Choose locations to iterate
    locations_to_check = MULTI_LOCATIONS if MULTI_LOCATIONS else [None]

    for i, keyword in enumerate(keywords, 1):
        logger.info(f"\n{'='*60}\n[{i}/{len(keywords)}] {keyword}\n{'='*60}")
        try:
            for idx_loc, location in enumerate(locations_to_check):
                if location:
                    logger.info(f"📍 Location: {location['name']}")

                rows, status = process_keyword(keyword, BRANDS_FILE, location)

                # Update keyword-level status (only once per keyword)
                if idx_loc == 0:
                    if status == "SUCCESS":
                        stats["success"] += 1
                    elif status == "NO_SHOPPING":
                        stats["no_shopping"] += 1
                    elif status == "API_ERROR":
                        stats["api_error"] += 1

                # Save rows
                if rows:
                    with open(OUTPUT_FILE, mode="a", encoding="utf-8-sig", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(rows)
                    stats["products_found"] += len(rows)

                    # 'Position' column index = 11 after adding price/discount columns
                    top_positions = sorted([row[11] for row in rows])[:3]
                    logger.info(f"✅ {len(rows)} brand product(s) found | Top positions: {top_positions}")
                else:
                    logger.info(f"ℹ️ No matching brand products (Status: {status})")

                # Small pause between locations
                if location and idx_loc != len(locations_to_check) - 1:
                    time.sleep(1.0)

            # Rate-limit between keywords
            if i < len(keywords):
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                logger.info(f"⏳ Waiting {delay:.1f}s...")
                time.sleep(delay)

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            break
        except Exception as e:
            logger.error(f"Error processing '{keyword}': {e}")
            stats["api_error"] += 1

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("✅ PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Total keywords: {stats['total']}")
    pct = (stats['success']/stats['total']*100.0) if stats['total'] else 0.0
    logger.info(f"📊 With shopping results: {stats['success']} ({pct:.1f}%)")
    pct_ns = (stats['no_shopping']/stats['total']*100.0) if stats['total'] else 0.0
    logger.info(f"📊 No shopping results: {stats['no_shopping']} ({pct_ns:.1f}%)")
    logger.info(f"📊 API Errors: {stats['api_error']}")
    logger.info(f"📊 Total brand products found: {stats['products_found']}")
    logger.info(f"📁 CSV saved: {OUTPUT_FILE}")

    check_serpapi_account()

# Allow both styles: your file may call run() directly, or via CLI
if __name__ == "__main__":
    run()
