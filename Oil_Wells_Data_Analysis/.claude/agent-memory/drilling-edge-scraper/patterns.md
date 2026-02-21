# DrillingEdge Scraper — Detailed Patterns

## HTML structure (verified 2026-02-20)

### Search URL
```
GET https://www.drillingedge.com/search?type=wells&well_name=<encoded>&api_no=<api>
```
Server-rendered HTML — no Selenium needed. Plain requests.Session works.

### ND well URL regex (search results page)
```python
re.findall(r'href="(https://www\.drillingedge\.com/north-dakota[^"]+/wells/[^"]+)"', html)
```
Take `matches[0]` as the best result.

### Production date (detail page)
The text "Production Dates on File: Month YYYY to Month YYYY" appears in the page body.
Use `soup.get_text(" ", strip=True)` on the whole page, then:
```python
re.search(
    r"Production Dates on File:\s*([A-Za-z]+ \d{4})\s+to\s+([A-Za-z]+ \d{4})",
    page_text
)
```
CRITICAL: Do NOT use `.+` for the end date — the page text has trailing content that
will be greedily consumed, resulting in "December" instead of "December 2025".

### block_stat paragraphs
```html
<p class="block_stat"><span class="dropcap">608</span> Barrels of Oil Produced in Dec 2025</p>
<p class="block_stat"><span class="dropcap">884</span> MCF of Gas Produced in Dec 2025</p>
```
Parse: find all `p.block_stat`, get `span.dropcap` text for the number.
Key strings to detect: "Barrels of Oil" → oil; "MCF of Gas" → gas.

### Detail table
Iterate `<th>` tags, take next sibling `<td>`:
- "Well Status"          → de_status
- "Well Type"            → de_well_type
- "Well Direction"       → de_direction (H / V / D)
- "Operator"             → de_operator
- "Closest City"         → de_closest_city
- "Latitude / Longitude" → split on ", " for de_latitude, de_longitude

## Rate limiting
- 1.5–2.2 second random sleep between requests
- No blocks observed at this rate
- Retry on 429/503 with exponential back-off (3 attempts max)

## Not-found root causes (2026-02-20 run)
- OCR artifacts in well names (e.g., "AUmta", "~!8T1'3N P'>IOrw", "Atl mta")
- Truncated/garbled names (e.g., "Lewis Federa1 5300 11‐")
- No API# available for many not-found wells (no API fallback possible)
- Some wells (SWD disposal, very old wells) not indexed on DrillingEdge ND
- Name variants: "Atlanta 14-SH" vs "Atlanta 14-6H", "Atlanta 1-GH" vs "Atlanta 1-6H"
- Spacing differences: "1 OT" (OCR) vs "10T"

## Production values
- Active wells: integer BBLs/MCF for most recent month (Dec 2025 in this run)
- Inactive/Plugged/Abandoned: no block_stat paragraphs → de_oil_bbls and de_gas_mcf = N/A
- Some values appear as "1.2 k" (abbreviated by DrillingEdge) — stored as string as-is

## Resume support
Set `RESCRAPE = False` in `main()` to skip already-found wells.
Set `RESCRAPE = True` to force full re-scrape of all wells.
