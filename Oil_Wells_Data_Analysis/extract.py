"""
extract.py
Extracts well-specific information and stimulation data from North Dakota
Oil & Gas Division PDF files in the DSCI560_Lab5 folder.

Outputs:
  - well_info.csv        : one row per well
  - stimulations.csv     : one row per stimulation event

Tool choice rationale
---------------------
These PDFs contain *embedded* (machine-readable) text — they are digital forms,
not scanned images.  Tool comparison for this type of document:

  pdfplumber  ← used here
    • Built on pdfminer.six; preserves character-level layout.
    • Best for extracting text + tables from native (non-scanned) PDFs.
    • extract_tables() can parse grid-lined tables; extract_text() gives a
      reading-order string that works well with these government forms.
    • Pure Python, no system dependencies.

  PyPDF / PyPDF2
    • Simpler API but significantly weaker layout preservation; frequently
      joins adjacent columns or drops whitespace between fields.
    • Fine for sequential prose, poor for multi-column form data.

  ocrmypdf
    • Adds a Tesseract OCR *layer* onto a PDF, producing a new searchable PDF.
    • Useful to make scanned-image PDFs searchable; not a data-extraction tool
      by itself — still needs pdfplumber/PyPDF to read the layer afterward.
    • Unnecessary here since text is already embedded.

  pytesseract
    • Runs Tesseract OCR directly on rasterised images of PDF pages.
    • Requires pdf2image / poppler as a preprocessing step.
    • Best when PDFs are true image scans with NO embedded text.
    • Lower accuracy than pdfplumber for these forms because the rasterisation
      and OCR chain introduces noise (e.g. "Da· e" for "Date", "Sa11d" for "Sand").
    • These PDFs already show OCR noise from prior scanning; adding another OCR
      pass would only make it worse.

  camelot / tabula-py
    • Specialised table extractors (lattice and stream modes).
    • Excellent for clearly-delimited HTML/Excel-style tables.
    • The ND NDIC forms use borderless columns with whitespace alignment, which
      causes both tools to mis-segment columns; pdfplumber's text stream +
      regex parsing is more reliable for these forms.

Conclusion: pdfplumber + regex post-processing is the right tool here.
"""

import csv
import os
import re
from pathlib import Path

import pdfplumber


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", text or "").strip()


def _first_match(pattern: str, text: str, group: int = 1, flags: int = 0) -> str:
    m = re.search(pattern, text, flags)
    return _clean(m.group(group)) if m else ""


def _clean_ocr_date(raw: str) -> str:
    """
    Fix common OCR substitutions in dates.
    '1/2()/2014'  -> '1/20/2014'   (O rendered as ())
    '1O/O5/2O14'  -> '10/05/2014'  (letter O for zero)
    """
    # Replace letter-O and parenthesis-pairs that stand in for digit 0
    s = raw
    s = re.sub(r"\(\)", "0", s)        # () -> 0
    s = re.sub(r"(?<=[/\d])[OoI](?=[/\d])", "0", s)  # O/I -> 0 when between separators/digits
    return s


def _clean_address(addr: str) -> str:
    """Remove pool/field/form-label suffixes that bleed in from adjacent columns."""
    # Remove trailing pool/field name and table-border artefacts
    addr = re.sub(
        r"\s+(?:[I|]\s+)?\b(Pool|Bakken|Williston|Baker|Field|Permit\s+Type)\b.*$",
        "", addr, flags=re.I,
    )
    addr = re.sub(r"\s+[I|]\s*$", "", addr)   # trailing table-border 'I'
    return addr.strip()


def _parse_form8_well_line(line: str):
    """
    Extract well name and county from a Form 8 location data row.
    Format: "[well name] [Qtr-Qtr?] [Section] [Township] [N/S] [Range] [E/W?] [County]"
    Examples:
      "Basic Game & Fish 34-3 2 153 N 101 McKenzie"
      "Corps of Engineers 31-10 NWNE 10 153 N 101 McKenzie"
      "Lewis & Clark 2-4H SENE 4 153 N 101 w McKenzie"
    Returns (well_name, county).
    """
    tokens = line.split()
    if len(tokens) < 3:
        return "", ""
    i = len(tokens) - 1
    county = ""
    # County: trailing alphabetic word (3+ chars)
    if i >= 0 and re.match(r"^[A-Za-z]{3,}$", tokens[i]):
        county = tokens[i].title()
        i -= 1
    # Range direction (E/W) – single letter
    if i >= 0 and re.match(r"^[EWew]$", tokens[i]):
        i -= 1
    # Range number (2–3 digits)
    if i >= 0 and re.match(r"^\d{2,3}$", tokens[i]):
        i -= 1
    # N/S direction
    if i >= 0 and re.match(r"^[NSns]$", tokens[i]):
        i -= 1
    # Township (3 digits)
    if i >= 0 and re.match(r"^\d{3}$", tokens[i]):
        i -= 1
    # Section (1–2 digits)
    if i >= 0 and re.match(r"^\d{1,2}$", tokens[i]):
        i -= 1
    # Quarter-quarter (e.g. NWNE, SESE, SWSW)
    if i >= 0 and re.match(r"^[NSEWnsew]{2,4}$", tokens[i]):
        i -= 1
    return _clean(" ".join(tokens[: i + 1])), county


def _strip_location_suffix(name: str) -> str:
    """
    Remove trailing geographic location description from a well name line.
    E.g.: "Yukon 5301 41-12T T153N R101W Sec 13 & 24"  →  "Yukon 5301 41-12T"
          "Kline Federal 5300 41-18 13T2X Sec. 17/18/19/20 T153N R100W" → "Kline Federal 5300 41-18 13T2X"
          "Tallahassee 2-16H S!!C 18 & 21 153N-101W" → "Tallahassee 2-16H"
    """
    s = name
    # "Sec." or "Sec " followed by any text
    s = re.sub(r"\s+Sec\.?\s+.*$", "", s)
    # OCR-corrupted "Sec" (e.g. "S!!C")
    s = re.sub(r"\s+S[^\w]{0,3}[Cc]\s+\d.*$", "", s)
    # Township-Range notation: "T153N R100W" or "153N-101W"
    s = re.sub(r"\s+T\d+[NS]\s+R\d+[EW].*$", "", s, flags=re.I)
    s = re.sub(r"\s+\d{2,3}[NS][-\s]\d{2,3}[EW].*$", "", s, flags=re.I)
    return s.strip()


# ---------------------------------------------------------------------------
# Well-information extraction  (SFN 2468 completion form + supporting pages)
# ---------------------------------------------------------------------------

def extract_well_info(pdf_path: str) -> dict:
    """
    Extract well-level metadata from a single PDF.

    Priority for each field:
      well_file_number : from the filename  (Wxxxxx → xxxxx)
      well_name        : completion form "Well Name and Number" field; or "Well Name:" label
      operator         : completion form "Operator" field; or "Operator:" label
      api              : first "API #: XX-XXX-XXXXX" found in the document
      county           : "County, State: <County> Co., ND" or location row of completion form
      state            : always "ND"  (all files are North Dakota NDIC records)
      address          : operator address block in completion form
      latitude         : decimal "Latitude: XX.XXXXXX" (directional survey pages)
      longitude        : decimal "Longitude: -XXX.XXXXXX"
    """
    well_number = Path(pdf_path).stem.lstrip("Ww")

    info = {
        "well_file_number": well_number,
        "well_name":        "",
        "operator":         "",
        "api":              "",
        "county":           "",
        "state":            "ND",
        "address":          "",
        "latitude":         "",
        "longitude":        "",
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Early exit once all fields are populated
            if all([
                info["well_name"], info["operator"], info["api"],
                info["county"], info["address"],
                info["latitude"], info["longitude"],
            ]):
                break

            text = page.extract_text()
            if not text:
                continue

            lines = [l.strip() for l in text.split("\n")]

            # -- API number -----------------------------------------------
            if not info["api"]:
                api = _first_match(r"API\s*[#:]\s*([\d\-]+)", text)
                # Accept only standard NN-NNN-NNNNN format
                if api and re.match(r"^\d{2}-\d{3}-\d{5}", api):
                    info["api"] = api

            # -- County (broad scan — incident/spill reports, Form 8, etc.) --
            if not info["county"]:
                # "County : MCKENZIE" or "County: McKenzie"
                c = _first_match(r"County\s*[:\-]\s*([A-Z][A-Za-z]+)", text)
                if c and len(c) >= 4:
                    info["county"] = c.title()
                # "McKenzie County" or "MCKENZIE COUNTY"
                _FORM_LABELS = {"range", "section", "township", "field", "pool",
                                "state", "city", "address", "permit", "spacing"}
                if not info["county"]:
                    c2 = _first_match(r"([A-Z][A-Za-z]{3,})\s+County", text)
                    if c2 and c2.lower() not in _FORM_LABELS:
                        info["county"] = c2.title()

            # -- Lat / Lon (decimal degrees, most reliable) ----------------
            if not info["latitude"]:
                lat = _first_match(r"Latitude[:\s]+([\d]{2}\.[\d]+)", text)
                if lat:
                    info["latitude"] = lat

            if not info["longitude"]:
                lon = _first_match(r"Longitude[:\s]+(-?[\d]{2,3}\.[\d]+)", text)
                if lon:
                    info["longitude"] = lon

            # -- "Well Information" consolidated page ----------------------
            # This is the most reliable single-page source; always override
            # values already set from other (noisier) pages.
            if "Well Information" in text and "Operator:" in text:
                op = _first_match(r"Operator:\s*(.*?)\s+API\s*#:", text)
                if op:
                    info["operator"] = op
                api2 = _first_match(r"API\s*#:\s*([\d\-]+)", text)
                if api2 and re.match(r"^\d{2}-\d{3}-\d{5}", api2):
                    info["api"] = api2
                wn = _first_match(r"Well Name:\s*(.*?)(?=\n|Footages)", text)
                if wn:
                    info["well_name"] = wn
                county_raw = _first_match(
                    r"County,\s*State:\s*([\w\s]+?)(?:\s+Co\.?,?\s*(?:ND|North Dakota)|$)",
                    text,
                )
                if county_raw:
                    info["county"] = county_raw.rstrip(",").strip()
                addr = _first_match(
                    r"Address:\s*([\w\s,\.]+?)(?=\n.*?(?:Well Name|Footages))",
                    text,
                    flags=re.DOTALL,
                )
                if addr:
                    info["address"] = addr.replace("\n", " ")

            # -- Completion Report (SFN 2468 / 2466 / Form 6) -------------
            # Detect by content rather than specific form number to handle
            # OCR noise in numbers and older form variants (2466 vs 2468).
            is_completion_form = bool(
                re.search(r"Well\s+Name.{0,10}Number", text, re.I)
                and "Operator" in text
                and re.search(r"(COMPLETION|RECOMPLETION)", text, re.I)
            )
            if is_completion_form:
                for i, line in enumerate(lines):

                    # Well Name — header uses "Well Name and Number" but OCR
                    # may corrupt "and" → match loosely
                    if not info["well_name"] and re.search(
                        r"Well\s+Name.{0,10}Number", line, re.I
                    ):
                        for j in range(i + 1, min(i + 4, len(lines))):
                            candidate = lines[j]
                            if candidate and "Spacing" not in candidate:
                                name = _strip_location_suffix(candidate)
                                if name:
                                    info["well_name"] = _clean(name)
                                break

                    # Operator — header line: "Operator [!|]Telephone ... Field"
                    if not info["operator"] and re.match(r"Operator\b", line, re.I):
                        # Try to read operator from same line
                        m = re.match(
                            r"Operator\s+(.*?)\s+(?:Telephone|\(|\d{3})", line
                        )
                        if m and m.group(1).strip():
                            info["operator"] = _clean(m.group(1))
                        elif i + 1 < len(lines):
                            nxt = lines[i + 1]
                            if nxt and not re.match(
                                r"^(Address|City|State|Zip|Pool)", nxt, re.I
                            ):
                                # Strip phone number  e.g. "(281) 404-9591 Baker"
                                op_clean = re.sub(r"\s+\(\d{3}\).*$", "", nxt)
                                op_clean = re.sub(r"\s+\d{3}[-\.\s]\d{3}.*$", "", op_clean)
                                info["operator"] = _clean(op_clean)

                    # Address block (line right after "Address" label)
                    if not info["address"] and re.match(r"^Address\b", line, re.I):
                        after = re.sub(r"^Address\s*", "", line, flags=re.I).strip()
                        # Strip form column labels that bleed in ("Pool", "Bakken", etc.)
                        after = re.sub(
                            r"\s*\b(Pool|Field|Bakken|Baker|Permit\s+Type)\b\s*$",
                            "", after, flags=re.I
                        ).strip()
                        if after and after.lower() not in ("pool", "field", "bakken"):
                            info["address"] = _clean_address(after)
                        elif i + 1 < len(lines):
                            info["address"] = _clean_address(_clean(lines[i + 1]))

                    # County from location row — look at next 3 lines for safety
                    if not info["county"] and "County" in line:
                        for j in range(i + 1, min(i + 4, len(lines))):
                            loc_data = lines[j]
                            tokens = loc_data.split()
                            if tokens:
                                possible = tokens[-1]
                                # County names: capitalised word ≥4 chars
                                if re.match(r"^[A-Z][a-z]{3,}$", possible):
                                    info["county"] = possible
                                    break

            # -- Form 8: Authorization to Purchase / Transfer (fallback) --
            is_form8 = bool(
                re.search(r"AUTHORIZATION TO PURCHASE", text, re.I)
                or re.search(r"SFN\s*5698", text, re.I)
            )
            if is_form8 and (not info["well_name"] or not info["operator"]):
                for i, line in enumerate(lines):
                    if not info["well_name"] and re.search(
                        r"Well\s+Name\s+and\s+Number", line, re.I
                    ):
                        for j in range(i + 1, min(i + 6, len(lines))):
                            cand = lines[j]
                            # Skip noise lines (single chars, too short)
                            if len(cand) < 4 or re.match(r"^[a-z]{1,2}\s*$", cand):
                                continue
                            name, county = _parse_form8_well_line(cand)
                            if name:
                                info["well_name"] = name
                            if county and not info["county"]:
                                info["county"] = county
                            break
                    if not info["operator"] and re.match(r"^Operator\b", line, re.I):
                        for j in range(i + 1, min(i + 4, len(lines))):
                            cand = lines[j]
                            if len(cand) < 3 or re.match(r"^[a-z]{1,2}\s*$", cand):
                                continue
                            # Strip trailing phone number and field name
                            op = re.sub(
                                r"\s+\(?\d{3}\)?[\s\-\.]\d{3}[\-\.]\d{4}.*$", "", cand
                            ).strip()
                            if op:
                                info["operator"] = _clean(op)
                            break
                    if not info["address"] and re.match(r"^Address\b", line, re.I):
                        for j in range(i + 1, min(i + 4, len(lines))):
                            cand = lines[j]
                            if len(cand) < 3 or re.match(r"^[a-z]{1,2}\s*$", cand):
                                continue
                            info["address"] = _clean(cand)
                            break

            # -- Well Summary page (Oasis / operator header format) --------
            if not info["well_name"] and re.search(r"^Well\s+Summary$", text, re.M):
                for i, line in enumerate(lines):
                    if re.match(r"^Well\s+Summary$", line, re.I):
                        if i > 0 and not info["operator"]:
                            info["operator"] = _clean(lines[i - 1])
                        if i + 1 < len(lines) and not info["well_name"]:
                            info["well_name"] = _clean(lines[i + 1])
                        if not info["county"]:
                            for j in range(i + 1, min(i + 6, len(lines))):
                                m = re.search(
                                    r"(\w+)\s+County,?\s*(?:ND|North Dakota)",
                                    lines[j], re.I,
                                )
                                if m:
                                    info["county"] = m.group(1)
                                    break
                        break

            # -- Spill / Incident Report (last resort) ---------------------
            if "Spill / Incident Report" in text and (
                not info["well_name"] or not info["operator"]
            ):
                if not info["operator"]:
                    op = _first_match(r"Well\s+Operator\s*:\s*(.+)", text)
                    if op:
                        info["operator"] = op.title()
                if not info["well_name"]:
                    wn = _first_match(
                        r"Well\s+or\s+Facility\s+Name\s*:\s*(.+)", text
                    )
                    if wn:
                        info["well_name"] = wn.title()
                if not info["county"]:
                    county = _first_match(r"^County\s*:\s*(\w+)", text, flags=re.M)
                    if county:
                        info["county"] = county.title()

    return info


# ---------------------------------------------------------------------------
# Stimulation data extraction  (Page 3 of SFN 2468)
# ---------------------------------------------------------------------------

def extract_stimulations(pdf_path: str) -> list:
    """
    Extract all Well Specific Stimulations entries from a PDF.
    Returns a list of dicts, one per non-empty stimulation entry.
    """
    well_number = Path(pdf_path).stem.lstrip("Ww")
    results = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            # Quickly check if this page could have stimulation data.
            # The "Stimulated Formation" column header is present on every
            # stimulation page (with or without data).
            if not re.search(r"Stimulated\s+Formation", text, re.I):
                continue
            # Also require at least one potential date-like pattern to avoid
            # pages that just reference "formation" in passing.
            if not re.search(r"\d{1,2}/\d{1,4}/\d{4}", text):
                continue

            lines = [l.strip() for l in text.split("\n")]
            stims = _parse_stimulation_lines(lines, well_number)
            # Deduplicate: skip entries already captured (same well, same date)
            seen = {(r["well_file_number"], r["date_stimulated"]) for r in results}
            for s in stims:
                key = (s["well_file_number"], s["date_stimulated"])
                if key not in seen:
                    results.append(s)
                    seen.add(key)

    return results


def _parse_stimulation_lines(lines: list, well_number: str) -> list:
    """
    State-machine parser for the Well Specific Stimulations section.

    Each stimulation slot has the structure:
      [HDR-1]   ... Stimulated Formation ... Volume Units
      [DATA-1]  MM/DD/YYYY  <Formation>  <Top>  <Bottom>  <Stages>  <Volume>  <Units>
      [HDR-2]   Type Treatment ... Maximum Treatment Rate (BBLS/Min)
      [DATA-2]  <TypeTreatment>  [Acid%]  <LbsProppant>  <MaxPSI>  <MaxRate>
      [DETAILS] Details
                <free-text lines…>

    OCR artefacts handled:
      - "Date" may appear as "Da· e", "Da :e" etc.
      - "Stimulated Formation" is robust and used as the DATA-1 trigger.
      - Date digits may contain () or O (e.g. "1/2()/2014" → "1/20/2014").
      - "Details" may appear as "De:ails", "De'.ails".
      - Formation names starting with digits (e.g. "3 Forks") are handled by
        detecting numbers via a right-to-left scan, not a left-to-right scan.
    """
    STATE_SCAN    = "scan"
    STATE_DATA1   = "data1"
    STATE_DATA2   = "data2"
    STATE_DETAILS = "details"

    # Header-1: Stimulated Formation column heading  (robust to date-noise)
    RE_HDR1 = re.compile(r"Stimulated\s+Formation", re.I)
    # Header-2: "Type Treatment" heading — OCR may corrupt "Type" to "Typ1 ~", "Typ·", etc.
    # Reliable alternative markers: "Lbs Proppant" or "Maximum Treatment Pressure"
    RE_HDR2 = re.compile(
        r"(?:Typ\S*\s*[\~\·\s]*Treatment|Lbs\s+Proppant|Maximum\s+Treatment\s+Pressure)",
        re.I,
    )
    # Data-1: line that starts with a date (allow OCR noise inside digits)
    RE_DATE = re.compile(r"^([\d()\[\]OoI]{1,2}/[\d()\[\]OoI]{1,4}/\d{4})\s+(.*)")
    # "Details" label (handle OCR: De:ails, De'.ails, De·tails …)
    RE_DETAILS = re.compile(r"^De[\w'\.\:\-]+\s*$", re.I)
    # End of section
    RE_END = re.compile(r"ADDITIONAL INFORMATION", re.I)
    # Noise lines: table-border artefacts, page headers, empty
    RE_NOISE = re.compile(r"^[I|'\-\.\s~_]*$|^Page\s+\d|^SFN?\s|^SFl|^\"|^\.\.")

    state = STATE_SCAN
    current: dict = {}
    detail_lines: list = []
    results: list = []

    def save_current():
        if current.get("date_stimulated"):
            current["details"] = "; ".join(detail_lines)
            current["well_file_number"] = well_number
            results.append(dict(current))

    for line in lines:
        if not line or RE_NOISE.match(line):
            continue
        if RE_END.search(line):
            save_current()
            break

        # ── SCAN: wait for the first Header-1 ──────────────────────────────
        if state == STATE_SCAN:
            if RE_HDR1.search(line):
                state = STATE_DATA1
            continue

        # ── DATA-1: expect date-formation-numbers row ───────────────────────
        if state == STATE_DATA1:
            if RE_HDR1.search(line):
                # Repeated HDR-1 before data → still waiting
                continue
            if RE_HDR2.search(line):
                # HDR-2 arrived before any data → empty slot, skip to DATA-2
                state = STATE_DATA2
                continue

            m = RE_DATE.match(line)
            if m:
                # Save the previous entry (if any)
                save_current()
                current = {}
                detail_lines = []

                raw_date = m.group(1)
                rest     = m.group(2).strip()

                date_str = _clean_ocr_date(raw_date)

                # Parse: rest = "<Formation words…> <Top> <Bottom> <Stages> <Volume> <Units>"
                # Strategy: work right-to-left to extract numbers, then the unit
                # word(s), leaving whatever text remains as the formation name.
                tokens = [
                    re.sub(r"[I|,]", "", t)       # strip table-border chars
                    for t in rest.split()
                    if re.sub(r"[I|,]", "", t)     # drop empty after clean
                ]

                # 1. Find the unit word (last purely-alphabetic token at tail)
                volume_units = ""
                while tokens and re.match(r"^[A-Za-z]+$", tokens[-1]):
                    volume_units = tokens.pop(-1)
                    break   # take only one unit word

                # 2. Collect contiguous trailing integers (formation may start with digit)
                numeric_tail = []
                for t in reversed(tokens):
                    if re.match(r"^\d+$", t):
                        numeric_tail.insert(0, t)
                    else:
                        break

                formation_tokens = tokens[: len(tokens) - len(numeric_tail)]
                formation = _clean(" ".join(formation_tokens))

                # 3. Assign numbers: expected order [top, bottom, stages, volume]
                n = numeric_tail
                current["date_stimulated"]     = date_str
                current["stimulated_formation"] = formation
                current["top_ft"]              = n[-4] if len(n) >= 4 else (n[0] if n else "")
                current["bottom_ft"]           = n[-3] if len(n) >= 4 else (n[1] if len(n) > 1 else "")
                current["stimulation_stages"]  = n[-2] if len(n) >= 4 else (n[2] if len(n) > 2 else "")
                current["volume"]              = n[-1] if len(n) >= 1 else ""
                current["volume_units"]        = volume_units

                state = STATE_DATA2
            else:
                # Non-date line (split column or OCR garbage) → advance
                state = STATE_DATA2
            continue

        # ── DATA-2: expect treatment-type / numbers row ─────────────────────
        if state == STATE_DATA2:
            if RE_HDR2.search(line):
                continue   # another HDR-2, still waiting for data
            if RE_HDR1.search(line):
                state = STATE_DATA1
                continue
            if RE_DETAILS.match(line) and len(line) < 15:
                state = STATE_DETAILS
                continue

            # Parse the treatment data line
            tokens = line.split()
            pure_nums:  list = []
            text_parts: list = []
            for t in tokens:
                t_c = re.sub(r"[,|I]", "", t)
                if re.match(r"^\d+\.?\d*$", t_c):
                    pure_nums.append(t_c)
                else:
                    text_parts.append(t)

            current["type_treatment"]    = " ".join(text_parts)
            # Separate integer and float numbers
            floats = [n for n in pure_nums if "." in n]
            ints   = [n for n in pure_nums if "." not in n]

            # Layout: [acid%?]  lbs_proppant  max_pressure  max_rate(float)
            # acid% is typically absent for sand fracs; identify by count
            current["acid_pct"]          = ""
            current["lbs_proppant"]      = ints[0] if len(ints) > 0 else ""
            current["max_pressure_psi"]  = ints[1] if len(ints) > 1 else ""
            current["max_rate_bbls_min"] = floats[0] if floats else (ints[2] if len(ints) > 2 else "")
            # If there are 4+ integers, the first is likely acid%
            if len(ints) >= 4:
                current["acid_pct"]          = ints[0]
                current["lbs_proppant"]      = ints[1]
                current["max_pressure_psi"]  = ints[2]
                if not floats:
                    current["max_rate_bbls_min"] = ints[3]

            state = STATE_DETAILS
            continue

        # ── DETAILS: collect free-text until next slot header ───────────────
        if state == STATE_DETAILS:
            if RE_HDR1.search(line):
                save_current()
                current = {}
                detail_lines = []
                state = STATE_DATA1
                continue
            if RE_DETAILS.match(line) and len(line) < 15:
                continue   # "Details" label of an empty slot
            detail_lines.append(line)

    save_current()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

WELL_INFO_FIELDS = [
    "well_file_number", "well_name", "operator", "api",
    "county", "state", "address", "latitude", "longitude",
]

STIM_FIELDS = [
    "well_file_number",
    "date_stimulated", "stimulated_formation",
    "top_ft", "bottom_ft", "stimulation_stages",
    "volume", "volume_units",
    "type_treatment", "acid_pct", "lbs_proppant",
    "max_pressure_psi", "max_rate_bbls_min",
    "details",
]


def main():
    pdf_dir   = Path("DSCI560_Lab5")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir.resolve()}")
        return

    out_well = "well_info.csv"

    # If well_info.csv already exists, only re-process wells missing a well_name.
    existing_rows: dict = {}   # well_file_number → row dict, preserves CSV order
    if Path(out_well).exists():
        with open(out_well, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_rows[row["well_file_number"]] = row
        pdf_files = [
            p for p in pdf_files
            if not existing_rows.get(Path(p).stem.lstrip("Ww"), {}).get("well_name")
        ]
        if not pdf_files:
            print("No wells with missing well_name found — nothing to update.")
            return
        print(f"Re-processing {len(pdf_files)} wells with missing well_name ...")

    well_rows: list = []
    stim_rows: list = []
    total = len(pdf_files)

    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"[{idx:3d}/{total}] {pdf_path.name} ...", end=" ", flush=True)
        try:
            wi = extract_well_info(str(pdf_path))
            well_rows.append(wi)
            if not existing_rows:
                st = extract_stimulations(str(pdf_path))
                stim_rows.extend(st)
                print(f"stimulations: {len(st)}")
            else:
                print(wi["well_name"] or "(still empty)")
        except Exception as exc:
            print(f"ERROR: {exc}")

    # --- Write / merge well_info.csv ---
    if existing_rows:
        # Merge: fill in only fields that were blank
        for row in well_rows:
            wfn = row["well_file_number"]
            existing = existing_rows.setdefault(wfn, row)
            for field in WELL_INFO_FIELDS:
                if not existing.get(field) and row.get(field):
                    existing[field] = row[field]
        all_rows = list(existing_rows.values())
    else:
        all_rows = well_rows

    with open(out_well, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WELL_INFO_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote/updated {len(all_rows)} well records  →  {out_well}")

    # --- Write stimulations.csv (full run only) ---
    if not existing_rows and stim_rows:
        out_stim = "stimulations.csv"
        with open(out_stim, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=STIM_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(stim_rows)
        print(f"Wrote {len(stim_rows)} stimulation records  →  {out_stim}")


if __name__ == "__main__":
    main()
