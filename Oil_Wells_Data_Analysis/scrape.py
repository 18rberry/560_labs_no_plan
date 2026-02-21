"""
DrillingEdge.com well data scraper.
Enriches well_info.csv with data from DrillingEdge.com for each well.
"""

import csv
import os
import re
import time
import random
from datetime import datetime, timezone
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV  = "/Users/benlu/560_labs_no_plan/Oil_Wells_Data_Analysis/well_info.csv"
OUTPUT_CSV = "/Users/benlu/560_labs_no_plan/Oil_Wells_Data_Analysis/well_info_enriched.csv"

SEARCH_URL = "https://www.drillingedge.com/search"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# New columns to add (in order)
DE_COLUMNS = [
    "de_url",
    "de_found",
    "de_status",
    "de_well_type",
    "de_direction",
    "de_operator",
    "de_closest_city",
    "de_latitude",
    "de_longitude",
    "de_oil_bbls",
    "de_gas_mcf",
    "de_production_start",
    "de_production_end",
]

SLEEP_MIN = 1.5
SLEEP_MAX = 2.2

# Regex to match North-Dakota well URLs in search results
ND_WELL_URL_RE = re.compile(
    r'href="(https://www\.drillingedge\.com/north-dakota[^"]+/wells/[^"]+)"'
)

# OCR-corrected well names keyed by well_file_number.
# Used as the search term when the name in the CSV is garbled.
NAME_CORRECTIONS = {
    "21796": "DAHL FEDERAL 2-15H",
    "22221": "Innoko 5301 43-12T",
    "22249": "Magnum 2-36-25H",
    "23359": "Atlanta 14-6H",
    "23361": "Atlanta 12-6H",
    "23372": "Atlanta 1-6H",
    "25160": "Columbus Federal 3-16H",
    "28601": "Chalmers Wade Federal 5301 44-24 12TXR",
    "28636": "Chalmers 5300 21-19 8T",
    "28637": "Chalmers 5300 21-19 10T",
    "28755": "Kline Federal 5300 31-18 7T",
    "29316": "Gramma Federal 5300 41-31 12B",
    "30188": "Lewis Federal 5300 11-31",
    "30789": "KLINE FEDERAL 5300 31-18 15T",
    "28600": "Chalmers 5301 44-24 4T2",
    "90258": "Atlanta #1 SWD",
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def _get(url: str, retries: int = 3) -> str | None:
    """GET a URL and return HTML text, with retry + exponential back-off."""
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=20)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (429, 503):
                wait = (2 ** attempt) * 3
                print(f"    [rate-limit {resp.status_code}] sleeping {wait}s …")
                time.sleep(wait)
            else:
                print(f"    [HTTP {resp.status_code}] {url}")
                return None
        except requests.RequestException as exc:
            wait = (2 ** attempt) * 2
            print(f"    [error attempt {attempt+1}] {exc}  retry in {wait}s")
            time.sleep(wait)
    return None


def _sleep():
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------
def _search_url(well_name: str, api: str = "") -> str:
    params = (
        f"type=wells"
        f"&operator_name="
        f"&well_name={quote_plus(well_name)}"
        f"&api_no={quote_plus(api)}"
        f"&lease_key=&state=&county=&section=&township=&range="
        f"&min_boe=&max_boe=&min_depth=&max_depth=&field_formation="
    )
    return f"{SEARCH_URL}?{params}"


def _find_well_url(well_name: str, api: str = "") -> tuple[str | None, str]:
    """
    Try to find a DrillingEdge well URL.
    Returns (url_or_None, confidence)  where confidence is HIGH / MEDIUM / NOT_FOUND.
    """
    # --- attempt 1: name + api (if api exists) ---
    if api:
        html = _get(_search_url(well_name, api))
        _sleep()
        if html:
            matches = ND_WELL_URL_RE.findall(html)
            if matches:
                return matches[0], "HIGH"

    # --- attempt 2: name only ---
    html = _get(_search_url(well_name))
    _sleep()
    if html:
        matches = ND_WELL_URL_RE.findall(html)
        if matches:
            conf = "HIGH" if api == "" else "MEDIUM"
            return matches[0], conf

    return None, "NOT_FOUND"


# ---------------------------------------------------------------------------
# Detail page parser
# ---------------------------------------------------------------------------
def _parse_detail(html: str) -> dict:
    """Parse a DrillingEdge well detail page and return a dict of fields."""
    soup = BeautifulSoup(html, "html.parser")

    result = {k: "N/A" for k in DE_COLUMNS}
    result["de_found"] = "1"

    # --- Production dates ---
    # Pattern: "Production Dates on File: <Month YYYY> to <Month YYYY>"
    # Match "Month YYYY" explicitly to avoid greedily consuming trailing page text.
    page_text = soup.get_text(" ", strip=True)
    m = re.search(
        r"Production Dates on File:\s*"
        r"([A-Za-z]+ \d{4})\s+to\s+([A-Za-z]+ \d{4})",
        page_text,
    )
    if m:
        result["de_production_start"] = m.group(1).strip()
        result["de_production_end"]   = m.group(2).strip()

    # --- block_stat paragraphs (oil & gas production) ---
    for p in soup.find_all("p", class_="block_stat"):
        dropcap = p.find("span", class_="dropcap")
        if not dropcap:
            continue
        val_text = dropcap.get_text(strip=True).replace(",", "")
        full_text = p.get_text(" ", strip=True)
        try:
            val = int(val_text)
        except ValueError:
            try:
                val = float(val_text)
            except ValueError:
                val = val_text
        if "Barrels of Oil" in full_text or "BBL" in full_text.upper():
            result["de_oil_bbls"] = val
        elif "MCF of Gas" in full_text or "MCF" in full_text.upper():
            result["de_gas_mcf"] = val

    # --- Detail table: iterate <th> and grab next sibling <td> ---
    th_to_field = {
        "Well Status":          "de_status",
        "Well Type":            "de_well_type",
        "Well Direction":       "de_direction",
        "Operator":             "de_operator",
        "Closest City":         "de_closest_city",
        "Latitude / Longitude": "_latlon",
    }

    for th in soup.find_all("th"):
        label = th.get_text(strip=True)
        if label not in th_to_field:
            continue
        td = th.find_next_sibling("td")
        if not td:
            continue
        value = td.get_text(" ", strip=True)

        field = th_to_field[label]
        if field == "_latlon":
            parts = [p.strip() for p in value.split(",")]
            if len(parts) >= 2:
                result["de_latitude"]  = parts[0]
                result["de_longitude"] = parts[1]
        else:
            result[field] = value if value else "N/A"

    return result


# ---------------------------------------------------------------------------
# CSV I/O helpers
# ---------------------------------------------------------------------------
def _load_existing(path: str, rescrape: bool = False) -> dict[str, dict]:
    """Return {well_file_number: row_dict} for rows already processed.

    If rescrape=True, return an empty dict so all wells are re-scraped.
    """
    if rescrape or not os.path.exists(path):
        return {}
    found = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("de_found") == "1":
                found[row["well_file_number"]] = row
    return found


def _write_rows(path: str, fieldnames: list[str], rows: list[dict]):
    """Write (or overwrite) the enriched CSV."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Load input
    with open(INPUT_CSV, newline="", encoding="utf-8") as fh:
        reader    = csv.DictReader(fh)
        orig_cols = reader.fieldnames or []
        input_rows = list(reader)

    print(f"Loaded {len(input_rows)} wells from {INPUT_CSV}")

    all_cols = orig_cols + [c for c in DE_COLUMNS if c not in orig_cols]

    # 2. Load any previously completed rows (resume support)
    # Set RESCRAPE=True to force re-scraping of all wells.
    RESCRAPE = False
    completed = _load_existing(OUTPUT_CSV, rescrape=RESCRAPE)
    print(f"  {len(completed)} already completed – will skip these.\n")

    # Build output rows list (preserve order)
    output_rows: list[dict] = []

    stats = {"found": 0, "not_found": 0, "skipped": 0, "errors": 0}
    scraped_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for idx, row in enumerate(input_rows, start=1):
        wfn       = row.get("well_file_number", "").strip()
        well_name = NAME_CORRECTIONS.get(wfn, row.get("well_name", "").strip())
        raw_api   = row.get("api", "").strip()

        # Normalise API: keep only digits and dashes, validate roughly
        api = raw_api if re.match(r"^\d{2}-\d{3}-\d{5}", raw_api) else ""

        # --- Resume: skip if already done ---
        if wfn in completed:
            output_rows.append(completed[wfn])
            stats["skipped"] += 1
            print(f"[{wfn}] {well_name} → SKIPPED (already found)")
            continue

        # Start with a blank enriched row
        enriched = dict(row)
        for col in DE_COLUMNS:
            enriched[col] = "N/A"

        # --- Search ---
        well_url, confidence = _find_well_url(well_name, api)

        if not well_url:
            enriched["de_found"] = "0"
            print(f"[{wfn}] {well_name} → NOT FOUND")
            stats["not_found"] += 1
            output_rows.append(enriched)
            # Save progress after every well
            _write_rows(OUTPUT_CSV, all_cols, output_rows)
            continue

        enriched["de_url"]   = well_url
        enriched["de_found"] = "1"
        print(f"[{wfn}] {well_name} → {well_url}")

        # --- Scrape detail page ---
        detail_html = _get(well_url)
        _sleep()

        if not detail_html:
            print(f"    [error] Could not fetch detail page.")
            stats["errors"] += 1
            enriched["de_found"] = "0"
            output_rows.append(enriched)
            _write_rows(OUTPUT_CSV, all_cols, output_rows)
            continue

        detail = _parse_detail(detail_html)
        detail["de_url"]   = well_url
        detail["de_found"] = "1"
        enriched.update(detail)

        stats["found"] += 1
        output_rows.append(enriched)
        _write_rows(OUTPUT_CSV, all_cols, output_rows)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print(f"  Total wells       : {len(input_rows)}")
    print(f"  Found (new)       : {stats['found']}")
    print(f"  Skipped (already) : {stats['skipped']}")
    print(f"  Not found         : {stats['not_found']}")
    print(f"  Errors            : {stats['errors']}")
    print(f"  Output            : {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
