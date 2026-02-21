# DrillingEdge Scraper — Persistent Agent Memory

See `patterns.md` for detailed notes.

## Key facts
- Input: `/Users/benlu/560_labs_no_plan/Oil_Wells_Data_Analysis/well_info.csv` (77 wells, ND)
- Output: `/Users/benlu/560_labs_no_plan/Oil_Wells_Data_Analysis/well_info_enriched.csv`
- Scraper: `/Users/benlu/560_labs_no_plan/Oil_Wells_Data_Analysis/scrape.py`
- Libraries confirmed working: requests 2.32.3, bs4 4.14.3

## Confirmed match rate (run 2026-02-20)
- 33 found / 44 not found out of 77 wells
- Most not-found are OCR-corrupted well names or wells not indexed on DrillingEdge

## Critical lessons
1. Production date regex: use `([A-Za-z]+ \d{4})` explicitly — NOT `.+` which greedily
   captures surrounding page text. See `patterns.md` for details.
2. `de_oil_bbls` / `de_gas_mcf` legitimately N/A for Inactive/Plugged/Abandoned wells.
3. Gas production values may appear as abbreviated strings like "1.2 k" — store as-is.

## Links
- patterns.md — HTML structure, regex patterns, not-found root causes
