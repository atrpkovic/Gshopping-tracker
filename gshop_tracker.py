import json
import os
from dotenv import load_dotenv
import requests
import csv
from datetime import datetime, timezone
import time
import os
import random
from urllib.parse import urlparse, parse_qs
from typing import Optional, List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====== CONFIG ======
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

KEYWORDS_FILE = "keywords.csv"
BRANDS_FILE = "brands.json"  # {"yourbrand.com": ["Brand Name", "Brand Alias"]}
OUTPUT_FILE = "google_shopping_brand_hits.csv"

SAVE_JSON = True  # Save raw API responses for debugging

# Rate limiting
MIN_DELAY = 2
MAX_DELAY = 5

# Shopping-specific settings - LOCATION TARGETING
LOCATION = None  # If you want a single location string, set it here
LOCATION_CODE = None  # If you want a single location code, set it here

COUNTRY = "us"  # Country code for gl
CURRENCY = "USD"

# Multi-location tracking
MULTI_LOCATIONS = [
    #{"name": "Coppell, Texas, United States", "code": 9061237},
    #{"name": "Dallas, Texas, United States", "code": 1023191},
    {"name": "Miami, Florida, United States", "code": 1021999},
    {"name": "Los Angeles, California, United States", "code": 1014221},
    {"name": "New York, New York, United States", "code": 1023768}
]

def _now_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _safe_name(s: str, limit=40):
    return "".join(c for c in s.replace(" ", "_")[:limit] if c.isalnum() or c in ("_", "-"))

def save_json(data, label, keyword):
    if not SAVE_JSON:
        return None
    fname = f"shopping_{label}_{_safe_name(keyword)}_{_stamp()}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON: {fname}")
    return fname

def normalize_host(u: str) -> str:
    """Extract clean domain from URL"""
    try:
        host = urlparse(u).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""

def extract_merchant_from_url(url: str) -> Optional[str]:
    """
    Extract merchant domain from Google Shopping product URL
    """
    try:
        parsed = urlparse(url)

        # Direct merchant URL
        if parsed.netloc and not parsed.netloc.endswith("google.com"):
            return normalize_host(url)

        # Extract from Google Shopping redirect URL
        # Format: https://www.google.com/url?url=https://merchant.com/...
        qs = parse_qs(parsed.query)
        if "url" in qs and qs["url"]:
            return normalize_host(qs["url"][0])
    except Exception:
        pass
    return None

def find_brands_in_shopping_results(shopping_results: List[Dict], brands_file: str, keyword: str, location_name: str = "Unknown") -> List[list]:
    """
    Match brands in Google Shopping results
    Returns: List of [timestamp, keyword, location, brand_domain, product_title, price, product_link, merchant, position, rating, reviews]
    """
    with open(brands_file, "r", encoding="utf-8") as f:
        brands = json.load(f)

    # Create brand lookup: domain -> [domain, alias1, alias2, ...]
    brand_map = {
        domain.lower(): [domain.lower()] + [a.lower() for a in aliases]
        for domain, aliases in brands.items()
    }

    ts = _now_ts()
    results = []

    for idx, item in enumerate(shopping_results, 1):
        product_title = item.get("title", "")
        price = item.get("price", "")
        link = item.get("link", "")
        source = item.get("source", "")
        rating = item.get("rating", 0)
        reviews = item.get("reviews", 0)

        merchant_domain = None

        # Method 1: Check the 'source' field (merchant name)
        if source:
            source_lower = source.lower()
            for domain, terms in brand_map.items():
                if any(term in source_lower for term in terms):
                    merchant_domain = domain
                    break

        # Method 2: Check product link
        if not merchant_domain and link:
            extracted_domain = extract_merchant_from_url(link)
            if extracted_domain:
                for domain, terms in brand_map.items():
                    domain_core = domain[4:] if domain.startswith("www.") else domain
                    if extracted_domain.endswith(domain_core):
                        merchant_domain = domain
                        break

        # Method 3: Check product title for brand mentions
        if not merchant_domain:
            title_lower = product_title.lower()
            for domain, terms in brand_map.items():
                if any(term in title_lower for term in terms[1:]):  # Skip domain, check aliases
                    merchant_domain = domain
                    break

        if merchant_domain:
            results.append([
                ts,
                keyword,
                location_name,
                merchant_domain,
                product_title[:200],
                price,
                link,
                source,
                idx,  # position within this result set
                rating,
                reviews
            ])
            logger.debug(f"Position {idx}: Found brand '{merchant_domain}' - {product_title[:50]}...")

    return results

def get_location_code(location_name: str) -> Optional[int]:
    """Find SerpApi location code for a city/region via Locations API"""
    try:
        response = requests.get(
            "https://serpapi.com/locations.json",
            params={"q": location_name, "limit": 5},
            timeout=10
        )
        data = response.json()
        if data:
            logger.info(f"Found {len(data)} locations matching '{location_name}':")
            for loc in data[:5]:
                logger.info(f"  - {loc['name']} (code: {loc['id']})")
            return data[0]['id']
    except Exception as e:
        logger.warning(f"Could not fetch location code: {e}")
    return None

def fetch_google_shopping_serpapi(keyword: str, location_override=None, max_retries: int = 3) -> Optional[Dict]:
    """
    Fetch Google Shopping results via SerpApi
    """
    # Determine location to use
    if location_override:
        loc_name = location_override.get('name')
        loc_code = location_override.get('code')
    elif LOCATION_CODE:
        loc_name = f"Code {LOCATION_CODE}"
        loc_code = LOCATION_CODE
    else:
        loc_name = LOCATION
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
                "no_cache": "true",  # correct param name
            }

            # Add location - code takes precedence over name
            if loc_code:
                params["location_code"] = loc_code
            else:
                params["location"] = loc_name or "United States"

            response = requests.get("https://serpapi.com/search", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if "error" in data:
                logger.error(f"SerpApi error: {data['error']}")
                if "Invalid API key" in str(data['error']):
                    logger.error("❌ Invalid API key")
                    return None
                if attempt < max_retries:
                    time.sleep(5)
                    continue
                return None

            # Attach location info for downstream tracking
            data['_location_name'] = loc_name if loc_code else (loc_name or "United States")
            data['_location_code'] = loc_code

            return data

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt}")
            if attempt < max_retries:
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            if attempt < max_retries:
                time.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break

    return None

def process_keyword(keyword: str, brands_file: str, location_override=None) -> tuple:
    """Process a single keyword for Google Shopping results"""
    data = fetch_google_shopping_serpapi(keyword, location_override)

    if not data:
        return [], "API_ERROR"

    location_name = data.get('_location_name', 'Unknown')

    if SAVE_JSON:
        save_json(data, "full_response", keyword)

    shopping_results = data.get("shopping_results", [])

    if not shopping_results:
        logger.info(f"No shopping results found for: {keyword}")
        return [], "NO_SHOPPING"

    logger.info(f"✓ Found {len(shopping_results)} shopping results for: {keyword}")

    rows = find_brands_in_shopping_results(shopping_results, brands_file, keyword, location_name)
    return rows, "SUCCESS"

def check_serpapi_account():
    """Check SerpApi account info"""
    try:
        response = requests.get("https://serpapi.com/account", params={"api_key": SERPAPI_KEY}, timeout=10)
        data = response.json()

        if "error" in data:
            logger.error(f"Account check failed: {data['error']}")
            return None

        searches_left = data.get("total_searches_left", "unknown")
        plan = data.get("plan_name", "unknown")

        logger.info(f"📊 SerpApi Account: {plan}")
        logger.info(f"📊 Searches remaining: {searches_left}")

        return data
    except Exception as e:
        logger.warning(f"Could not check account: {e}")
        return None

def main():
    logger.info("🛍️ Starting Google Shopping Brand Tracker\n")

    # Validate files
    if not os.path.exists(KEYWORDS_FILE):
        logger.error(f"Keywords file not found: {KEYWORDS_FILE}")
        return
    if not os.path.exists(BRANDS_FILE):
        logger.error(f"Brands file not found: {BRANDS_FILE}")
        return

    # Check account
    logger.info("Checking SerpApi account...")
    account = check_serpapi_account()
    if not account:
        logger.warning("⚠️ Could not verify account, but continuing anyway...")

    # Show location info
    if LOCATION_CODE:
        logger.info(f"📍 Targeting location code: {LOCATION_CODE}")
    else:
        logger.info(f"📍 Targeting location: {LOCATION}")

    if MULTI_LOCATIONS:
        logger.info(f"📍 Multi-location mode: {len(MULTI_LOCATIONS)} locations")
        for loc in MULTI_LOCATIONS:
            code = loc.get("code", "N/A")
            logger.info(f"   - {loc['name']} (code: {code})")

    with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
        keywords = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(keywords)} keywords\n")

    # ---- Initialize output CSV (fresh file each run; Windows-safe, no blank lines) ----
    header = [
        "Timestamp",
        "Keyword",
        "Location",
        "Brand Domain",
        "Product Title",
        "Price",
        "Product Link",
        "Merchant Name",
        "Position",
        "Rating",
        "Reviews"
    ]

    # Start fresh each run to avoid appending to old data during testing
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with open(OUTPUT_FILE, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Process keywords
    stats = {
        "total": len(keywords),
        "success": 0,
        "no_shopping": 0,
        "api_error": 0,
        "products_found": 0
    }

    # Determine which locations to check
    locations_to_check = MULTI_LOCATIONS if MULTI_LOCATIONS else [None]

    for i, keyword in enumerate(keywords, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(keywords)}] Processing: {keyword}")
        logger.info(f"{'='*60}")

        try:
            # Process for each location
            for idx_loc, location in enumerate(locations_to_check):
                if location:
                    logger.info(f"📍 Querying results for location: {location['name']}")

                rows, status = process_keyword(keyword, BRANDS_FILE, location)

                # Update stats (only once per keyword, not per location)
                if idx_loc == 0:
                    if status == "SUCCESS":
                        stats["success"] += 1
                    elif status == "NO_SHOPPING":
                        stats["no_shopping"] += 1
                    elif status == "API_ERROR":
                        stats["api_error"] += 1

                # Save results
                if rows:
                    with open(OUTPUT_FILE, mode="a", encoding="utf-8-sig", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(rows)
                    stats["products_found"] += len(rows)

                    # Show top positions
                    top_positions = sorted([row[8] for row in rows])[:3]  # Position is index 8
                    logger.info(f"✅ Found {len(rows)} product(s) from your brands")
                    logger.info(f"📍 Top positions: {top_positions}")
                else:
                    logger.info(f"ℹ️ No brand products found (Status: {status})")

                # Small delay between locations
                if location and idx_loc != len(locations_to_check) - 1:
                    time.sleep(1)

            # Rate limiting between keywords
            if i < len(keywords):
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                logger.info(f"⏳ Waiting {delay:.1f}s before next keyword...")
                time.sleep(delay)

        except KeyboardInterrupt:
            logger.info("\n⚠️ Interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Error processing '{keyword}': {e}")
            stats["api_error"] += 1
            continue

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Total keywords: {stats['total']}")
    logger.info(f"📊 With shopping results: {stats['success']} ({(stats['success']/stats['total']*100 if stats['total'] else 0):.1f}%)")
    logger.info(f"📊 No shopping results: {stats['no_shopping']} ({(stats['no_shopping']/stats['total']*100 if stats['total'] else 0):.1f}%)")
    logger.info(f"📊 API Errors: {stats['api_error']}")
    logger.info(f"📊 Total brand products found: {stats['products_found']}")
    logger.info(f"📁 Results saved to: {OUTPUT_FILE}")

    # Check final account status
    check_serpapi_account()

if __name__ == "__main__":
    main()
