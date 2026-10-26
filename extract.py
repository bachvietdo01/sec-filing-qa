"""Deterministic extraction: rendered-PDF financial statements -> grounded facts.

This is the trust-critical spine. NO LLM here: numbers come out of the PDF by
deterministic parsing only, and every number is grounded (its printed token must
appear verbatim in the source page text) before it is allowed into the Bank.

Why line-parsing and not pdfplumber's table detection: the filings render with
zero ruling lines, so `extract_tables()` finds no tables, and its text strategy
shreds tokens ("sales" -> "s","ales"; stray "$"). But `extract_text()` returns
each statement row cleanly as `<label words> <num> <num> <num>`. So we parse the
text line-by-line: split the trailing run of numeric columns off the leading
label, and map columns to the periods named in the statement header.

At full scale this deterministic pass is the ~65-75%-accurate first stage; messy
regions would escalate to an LLM/vision fallback (still grounding-checked). Here
the clean statement tables parse deterministically, which is the point: no model
is trusted with a number it can misread.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import pdfplumber

# --------------------------------------------------------------------------- #
# The three core statements. The SEC does NOT prescribe statement TITLES (filers
# use Operations/Income/Earnings/Financial Position interchangeably) — but Reg S-X
# DOES prescribe the line-item captions inside (Rules 5-02/5-03). So each statement
# is matched by THREE gates: a title synonym (fast candidate), a period row nearby
# (real statement, not a TOC entry), and a line-item signature (the regulation-
# grounded fingerprint). A region that fails any gate is dropped, never misread.
# --------------------------------------------------------------------------- #
from dataclasses import dataclass as _dataclass


@_dataclass
class StmtSpec:
    key: str
    display: str
    title_re: "re.Pattern"
    anchors: tuple[str, ...]      # lowercased Reg S-X-style line-item captions
    min_anchors: int = 2


STATEMENTS: list[StmtSpec] = [
    StmtSpec(
        "operations", "Statement of Operations",
        re.compile(r"^(condensed\s+)?(consolidated\s+)?(statements?\s+of\s+"
                   r"(operations|income|earnings)|"
                   r"(operations|income|earnings)\s+statements?)\b", re.I),
        anchors=("total revenue", "total revenues", "net revenue", "net sales",
                 "cost of revenue", "cost of sales", "cost of goods",
                 "operating expenses", "income from operations",
                 "operating income", "net income"),
    ),
    StmtSpec(
        "balance_sheet", "Balance Sheet",
        re.compile(r"^(condensed\s+)?(consolidated\s+)?(balance\s+sheets?|"
                   r"statements?\s+of\s+financial\s+(position|condition))\b", re.I),
        anchors=("total assets", "total liabilities", "total current assets",
                 "cash and cash equivalents", "stockholders", "total equity"),
    ),
    StmtSpec(
        "cash_flows", "Statement of Cash Flows",
        re.compile(r"^(condensed\s+)?(consolidated\s+)?(statements?\s+of\s+"
                   r"cash\s+flows?|cash\s+flows?\s+statements?)\b", re.I),
        anchors=("operating activities", "investing activities",
                 "financing activities", "net cash provided", "net cash used",
                 "net cash from", "net change in cash"),
    ),
]

# A line that opens ANY statement or the Notes — used to STOP a region scan.
_BOUNDARY_RE = re.compile(
    r"^(condensed\s+)?(consolidated\s+)?("
    r"statements?\s+of\s+(operations|income|earnings|comprehensive|cash\s+flows?|"
    r"stockholders|changes\s+in|redeemable|equity|members)|"
    r"(operations|comprehensive\s+income|income|earnings|cash\s+flows?|"
    r"stockholders[’']?\s+equity|changes\s+in\s+equity)\s+statements?|"
    r"balance\s+sheets?|statements?\s+of\s+financial\s+(position|condition)|"
    r"notes\s+to\s+(the\s+)?(consolidated\s+)?financial\s+statements)",
    re.I,
)

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    # abbreviations (NVIDIA etc. use "Jan 25, 2026")
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_ALT = ("january|february|march|april|may|june|july|august|september|"
              "october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|"
              "oct|nov|dec")

# A token that belongs to the numeric columns (money, dash, or a bare "$").
_NUM_TOKEN = re.compile(r"^\(?\$?-?[\d,]+(?:\.\d+)?\)?$")
_DASH_TOKEN = re.compile(r"^[—–-]$")
_YEAR = re.compile(r"^(19|20)\d{2}$")


@dataclass
class Fact:
    # identity / provenance
    company: str
    cik: int
    ticker: str
    filing_stem: str
    form: str
    accession_no: str
    # what / when
    statement: str          # operations | balance_sheet | cash_flows
    raw_label: str          # verbatim line-item label from the filing
    period: str             # FY2025 | 2026Q2 | 2026YTD6M | ASOF2025-12-31
    period_type: str        # duration | instant  (kept for back-compat)
    period_kind: str        # annual | quarter | ytd | instant
    period_end: str         # 2025-12-31
    fiscal_year: int
    fiscal_quarter: int | None   # 1..4 for quarter/ytd, else None
    months: int | None           # period length: 12 annual, 3 Q, 6/9 YTD, None instant
    # the number, kept as printed (not pre-multiplied) so it stays grounded
    value: float
    unit: str               # USD
    scale: str              # millions | units
    is_negative: bool
    is_dash: bool
    gaap_flag: bool         # statement line items are GAAP
    # traceability
    page_no: int            # 1-based PDF page
    raw_cell_text: str      # the printed token, e.g. "94,827" or "(338)"
    extraction_method: str  # pdfplumber_text
    grounding_ok: bool
    confidence: float
    section: str = field(default="")       # nearest header above the row
    source_line: str = field(default="")   # the full text line it came from
    period_end_inferred: bool = False      # day assumed, not printed in the header


# --------------------------------------------------------------------------- #
# Number parsing
# --------------------------------------------------------------------------- #
def _parse_number(tok: str) -> tuple[float, bool, bool]:
    """Return (value, is_negative, is_dash) for one printed cell token."""
    if _DASH_TOKEN.match(tok):
        return 0.0, False, True
    neg = tok.startswith("(") and tok.endswith(")")
    cleaned = tok.strip("()").replace("$", "").replace(",", "").strip()
    val = float(cleaned)
    if neg:
        val = -val
    return val, neg, False


def _is_grounded(tok: str, page_text: str, is_dash: bool) -> bool:
    """The trust invariant's gate: a number is grounded only if its printed
    token appears verbatim on its source page (a dash '—' means nil and needs
    no lookup). Anything else is NOT grounded and must not enter an answer."""
    if is_dash:
        return True
    return tok in page_text


_SCALE_RE = re.compile(r"\bin\s+(thousands|millions|billions)\b", re.I)


def _read_scale(region: list[tuple[int, str]], default: str = "millions") -> str:
    """Read the money scale from the statement's '(in thousands|millions|
    billions …)' parenthetical. NEVER assume: a '(in thousands)' filer would
    otherwise be 1000x off. Falls back to `default` only if none is stated."""
    for _pg, ln in region[:8]:
        m = _SCALE_RE.search(ln)
        if m:
            return m.group(1).lower()
    return default


def _unit_and_scale(section: str, label: str,
                    default_scale: str = "millions") -> tuple[str, str]:
    """Derive (unit, scale) from the row's section header + label, given the
    statement's money scale (`default_scale`, read from its parenthetical).

    Two row kinds share generic labels ("Basic", "Diluted"): earnings *per
    share* (dollars, not scaled) and *weighted average shares* (a share count in
    the statement's scale). The section header disambiguates them."""
    lab, sec = label.lower(), section.lower()
    # Decide from the LABEL first — NOT a stale section header. A prior
    # "Common stock; ... 3,751 shares issued and outstanding" row would otherwise
    # leak a shares context onto every row beneath it (e.g. "Total stockholders'
    # equity" was mislabeled unit='shares').
    if re.search(r"per\s+(?:\w+\s+)?share", lab):
        return "USD", "units"            # the label itself says "per share" -> EPS
    if "weighted" in lab and re.search(r"\bshares?\b", lab):
        return "shares", default_scale   # an explicit weighted-average share count
    # Generic reused labels ("Basic"/"Diluted"/"Basic and diluted") take their
    # meaning from the nearest section header. Check the SHARE-COUNT sense before
    # per-share: a weighted-average-shares header ("...shares used in computing...
    # per share") contains "per share" too, so order matters.
    if lab.strip().strip("*0123456789() ") in ("basic", "diluted",
                                               "basic and diluted"):
        if "weighted" in sec or "shares used" in sec or "shares outstanding" in sec:
            return "shares", default_scale
        if re.search(r"per\s+(?:\w+\s+)?share", sec):
            return "USD", "units"
        return "USD", "units"            # bare generic -> EPS (the common case)
    return "USD", default_scale          # everything else is a money line


def _split_label_and_values(line: str) -> tuple[str, list[str]]:
    """Split a statement line into (label, [printed numeric tokens]).

    Works right-to-left: consume the trailing run of numeric / $ / dash tokens;
    the remaining prefix is the label. Bare "$" tokens are dropped (they are
    currency marks, not columns). A line with no trailing numbers -> no values
    (a section header like "Revenues" or "Current assets").
    """
    # Drop footnote/cross-references like "(Note 12)" that would otherwise be
    # mis-parsed as a numeric column (the "12)" token) and break the column count.
    line = re.sub(r"\(\s*notes?\s+[^)]*\)", " ", line, flags=re.I)
    toks = line.split()
    values: list[str] = []
    i = len(toks)
    while i > 0:
        t = toks[i - 1]
        if t == "$":                       # currency mark, not a column
            i -= 1
            continue
        if _NUM_TOKEN.match(t) or _DASH_TOKEN.match(t):
            values.append(t)
            i -= 1
            continue
        break
    values.reverse()
    label = " ".join(toks[:i]).strip()
    return label, values


# --------------------------------------------------------------------------- #
# Statement-region scanning
# --------------------------------------------------------------------------- #
def _page_lines(pdf) -> list[tuple[int, str]]:
    """Flatten the PDF into (page_no_1based, line_text) pairs."""
    out: list[tuple[int, str]] = []
    for idx, page in enumerate(pdf.pages):
        for line in (page.extract_text() or "").splitlines():
            out.append((idx + 1, line))
    return out


def _find_region(lines: list[tuple[int, str]], spec: StmtSpec,
                 max_lines: int = 260) -> list[tuple[int, str]]:
    """Return the (page, line) slice for one statement, or [] if not safely
    found. THREE gates, all required (else keep scanning / give up):
      1. the line matches the statement's title synonym regex;
      2. a PERIOD ROW (a line of only years, e.g. "2025 2024 2023") follows
         within a few lines — proves a real statement, not a TOC/index entry;
      3. the region carries >= min_anchors of the statement's Reg S-X line-item
         captions — the regulation-grounded fingerprint.
    A region that fails gate 3 is rejected and scanning continues; a statement
    that never passes yields [] -> no facts -> an honest refusal downstream."""
    for k, (_pg, ln) in enumerate(lines):
        if not spec.title_re.match(ln.strip()):
            continue
        # gate 2: a period row (bare-year OR inline date+year) within a few lines
        if not any(_period_row_years(l) for _p, l in lines[k + 1:k + 9]):
            continue
        # build the prospective region: title -> next statement/Notes boundary
        region = [lines[k]]
        for j in range(k + 1, min(k + max_lines, len(lines))):
            if _BOUNDARY_RE.match(lines[j][1].strip()):
                break
            region.append(lines[j])
        # gate 3: line-item signature
        body = " ".join(l for _p, l in region).lower()
        if sum(1 for a in spec.anchors if a in body) >= spec.min_anchors:
            return region
        # else: a title+year lookalike without the right rows -> keep scanning
    return []


@dataclass
class PeriodCol:
    """One value column's period: who it is in time."""
    year: int
    kind: str               # annual | quarter | ytd | instant
    period_end: str         # YYYY-MM-DD
    quarter: int | None
    months: int | None
    label: str              # FY2025 | 2026Q2 | 2026YTD6M | ASOF2025-12-31
    period_type: str        # duration | instant (back-compat)
    inferred: bool = False  # month/day NOT printed for this column (assumed)


# Duration captions -> number of months. Filers phrase the column period many
# ways and often put it on a DIFFERENT line than the dates, so we scan durations
# and dates separately and reconcile them.
_DURATION_RE = re.compile(
    r"(twelve\s+months|fiscal\s+years?|years?)\s+end|"       # -> 12
    r"(nine\s+months|three\s+fiscal\s+quarters|three\s+quarters)\s+end|"  # -> 9
    r"(six\s+months|two\s+fiscal\s+quarters|two\s+quarters|first\s+half)\s+end|"  # -> 6
    r"(three\s+months|fiscal\s+quarters?|quarters?)\s+end",  # -> 3
    re.I)
# An inline date WITH its year ("Jan 25, 2026", "December 31, 2025") — filers
# (NVIDIA, Vertiv, Hershey-Q) that put the date+year on the period row itself.
_INLINE_DATE_RE = re.compile(
    rf"({_MONTH_ALT})\.?\s+(\d{{1,2}}),?\s+((?:19|20)\d{{2}})", re.I)
# A bare "<Month> <day>" (no year) — dates in the context above a bare-year row.
_DATE_RE = re.compile(rf"({_MONTH_ALT})\.?\s+(\d{{1,2}})\b(?!\s*,?\s*(?:19|20)\d{{2}})",
                      re.I)


def _months_of(m: "re.Match") -> int:
    return 12 if m.group(1) else 9 if m.group(2) else 6 if m.group(3) else 3


def _durations_in(context: str) -> list[int]:
    return [_months_of(m) for m in _DURATION_RE.finditer(context)]


def _dates_in(context: str) -> list[tuple[int, int]]:
    return [(_MONTHS[m.group(1).lower()], int(m.group(2)))
            for m in _DATE_RE.finditer(context)]


# Duration / date words allowed on a period row (some filers put the whole
# header on one line, e.g. Hershey's "For the years ended December 31, 2025 2024 2023").
_PERIOD_WORDS_OK = {
    "for", "the", "years", "year", "ended", "ending", "month", "months",
    "three", "six", "nine", "twelve", "and", "period", "periods", "as", "of",
    "fiscal", "week", "weeks", "quarter", "quarters", "ended:",
    # scale/caption words some filers put on the SAME line as the period
    # (Procter & Gamble's interim header "Amounts in millions except per share
    # amounts 2026 2025 2026 2025" has no ';' to strip, so the caption tokens
    # must be tolerated here or the whole statement is dropped). These are
    # specific enough that a prose sentence still leaves real words behind.
    "amounts", "amount", "in", "thousands", "millions", "billions", "except",
    "per", "share", "shares", "stated", "value", "values", "data", "dollars",
}


def _period_row_years(line: str) -> list[int]:
    """If `line` is a PERIOD ROW, return its years, else []. Accepts:
      * a bare-year row  — "2025 2024 2023"
      * an inline date+year row — "December 31, 2025 December 31, 2024"
      * a combined header line — "For the years ended December 31, 2025 2024 2023"
    A prose line that merely mentions a year ("We audited … December 31, 2025")
    is NOT a period row: after removing years, months, day-numbers and known
    duration words, a real period row leaves nothing behind.

    Some filers (Procter & Gamble) put a scale/caption note and the period on ONE
    line — "Amounts in millions except per share amounts; fiscal years ended June
    30 2026 2025 2024". The note before the ';' is not part of the period row, so
    it is stripped before the residual test (the period segment after the last
    ';' — "fiscal years ended June 30 2026 2025 2024" — then leaves nothing
    behind). Lines without a ';' are unaffected."""
    if ";" in line:
        line = line.rsplit(";", 1)[1]
    toks = line.replace(",", " ").split()
    years = [int(t) for t in toks if _YEAR.match(t)]
    if not years:
        return []
    residual = [t for t in toks
                if not _YEAR.match(t) and t.lower() not in _MONTHS
                and not t.isdigit() and t.lower() not in _PERIOD_WORDS_OK]
    return years if not residual else []


def _find_period_row(region: list[tuple[int, str]]) -> tuple[int | None, list[int], str]:
    """Return (index_in_region, years, raw_line) of the statement's period row."""
    for i, (_pg, ln) in enumerate(region[1:14], start=1):
        ys = _period_row_years(ln)
        if ys:
            return i, ys, ln.strip()
    return None, [], ""


def _mk_col(year: int, months: int | None, month: int, day: int) -> PeriodCol:
    period_end = f"{year:04d}-{month:02d}-{day:02d}"
    if months is None:
        return PeriodCol(year, "instant", period_end, None, None,
                         f"ASOF{period_end}", "instant")
    if months == 12:
        return PeriodCol(year, "annual", period_end, None, 12,
                         f"FY{year}", "duration")
    q = (month - 1) // 3 + 1                    # calendar quarter it ends in
    if months == 3:
        return PeriodCol(year, "quarter", period_end, q, 3, f"{year}Q{q}", "duration")
    return PeriodCol(year, "ytd", period_end, q, months, f"{year}YTD{months}M",
                     "duration")


def _fix_periods(cols: list[PeriodCol], stmt_key: str | None,
                 annual_filing: bool) -> list[PeriodCol]:
    """Correct two period derivations that `_mk_col` can't get right in isolation:

    1. Operations / cash-flow statements are DURATION statements — never instant.
       A column that came out 'instant' there is a missed duration; for an annual
       filing it is the fiscal year (Bloom Energy's 10-K income statement printed
       dates but no duration keyword, so its annual columns were tagged instant
       and all annual questions refused).
    2. FISCAL quarter numbering relative to the company's fiscal year-end month,
       not the calendar quarter of the period-end: Disney's fiscal year ends in
       September, so its quarter ending in June is fiscal Q3, not calendar Q2; its
       nine-month YTD is 3 fiscal quarters (Q3), not Q2.
    """
    if stmt_key in ("operations", "cash_flows") and annual_filing:
        cols = [replace(c, kind="annual", months=12, quarter=None,
                        label=f"FY{c.year}", period_type="duration")
                if c.kind == "instant" else c for c in cols]
    # the fiscal year-end month: from an annual column, else inferred from a YTD
    # column's own (end month, length), else the filing's period-end month.
    fy_end = next((int(c.period_end[5:7]) for c in cols if c.kind == "annual"), None)
    if fy_end is None:
        for c in cols:
            if c.kind == "ytd" and c.months:
                m = int(c.period_end[5:7])
                fy_end = ((m + (12 - c.months) - 1) % 12) + 1
                break
    out = []
    for c in cols:
        if c.kind == "quarter" and fy_end:
            m = int(c.period_end[5:7])
            fq = ((m - fy_end - 1) % 12) // 3 + 1
            out.append(replace(c, quarter=fq, label=f"{c.year}Q{fq}"))
        elif c.kind == "ytd" and c.months:
            fq = c.months // 3          # 6 months -> 2 fiscal quarters, 9 -> 3
            out.append(replace(c, quarter=fq, label=f"{c.year}YTD{c.months}M"))
        else:
            out.append(c)
    return out


def _read_periods(region: list[tuple[int, str]],
                  default_md: tuple[int, int] = (12, 31),
                  stmt_key: str | None = None,
                  annual_filing: bool = False) -> list[PeriodCol]:
    """Read one PeriodCol per value column, for both header conventions:
    a bare-year row with the dates in a context line above (Tesla/GOOG), and an
    inline date+year row (NVIDIA/Vertiv). Also covers the interim 10-Q headers
    (3-month + YTD groups; balance sheet with two different as-of dates)."""
    year_idx, years, period_line = _find_period_row(region)
    if not years:
        return []
    n = len(years)
    context = " ".join(l.strip() for _p, l in region[1:year_idx])
    durations = _durations_in(context + " " + period_line)

    # Shape B: the period row carries inline date+year, one per column (NVIDIA,
    # Vertiv, Hershey-Q). Durations (if any) split the columns into equal groups.
    triples = [(_MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3)))
               for m in _INLINE_DATE_RE.finditer(period_line)]
    if triples and len(triples) == n:
        if not durations:
            mpc = [None] * n                       # dates only -> instants (B/S)
        elif len(durations) == n:
            mpc = durations
        elif n % len(durations) == 0:
            k = n // len(durations)
            mpc = [durations[i // k] for i in range(n)]
        else:
            mpc = [durations[0]] * n
        return _fix_periods([_mk_col(yr, mpc[i], mon, day)
                             for i, (mon, day, yr) in enumerate(triples)],
                            stmt_key, annual_filing)

    # Shape A: bare-year row; durations + dates come from the context (they may
    # be on separate lines). Columns split into equal groups per duration/date.
    # A period row that states the date ONCE and then lists prior fiscal years
    # ("June 30, 2026 2025", "Year Ended June 30, 2026 2025 2024" — Microsoft)
    # gives a single inline date for several columns: that date's month/day
    # applies to every column, and with no duration keyword the columns are
    # as-of instants (a balance sheet), not 12-month durations.
    dates = _dates_in(context)
    line_md = (triples[0][0], triples[0][1]) if triples else None
    g = len(durations) or len(dates) or 1
    if n % g != 0:
        g = 1
    per = n // g
    cols = []
    for c, yr in enumerate(years):
        gi = c // per
        if durations:
            months = durations[gi] if gi < len(durations) else durations[-1]
        elif dates or (line_md and stmt_key == "balance_sheet"):
            months = None                          # a date but no duration -> instant
        else:
            months = 12                            # bare years, no date -> annual
        if dates and len(dates) == n:
            mon, day = dates[c]                    # one printed date per column
        elif dates:
            mon, day = dates[gi] if gi < len(dates) else dates[-1]
        elif line_md:
            mon, day = line_md                     # the once-stated header date
        else:
            mon, day = default_md                  # headerless -> filing's own date
        col = _mk_col(yr, months, mon, day)
        if not dates and not line_md:
            col = replace(col, inferred=True)      # the day was assumed, not printed
        cols.append(col)
    return _fix_periods(cols, stmt_key, annual_filing)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_facts(pdf_path: str | Path, entity, filing) -> list[Fact]:
    """Parse the three core statements of a rendered 10-K/10-Q PDF into Facts.

    `entity` needs .cik/.ticker/.title; `filing` needs ['form'],['accession_no'].
    """
    pdf_path = Path(pdf_path)
    stem = pdf_path.stem
    facts: list[Fact] = []

    # the filing's own reported period-end -> the date fallback for headers that
    # print only years (e.g. Disney). Try the filing dict, else the stem date.
    rd = str(filing.get("report_date") or "")
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", rd) or re.search(r"(\d{4})-(\d{2})-(\d{2})", stem)
    default_md = (int(m.group(2)), int(m.group(3))) if m else (12, 31)

    with pdfplumber.open(pdf_path) as pdf:
        lines = _page_lines(pdf)
        # page text cache for the grounding check
        page_text = {i + 1: (pdf.pages[i].extract_text() or "")
                     for i in range(len(pdf.pages))}

        for spec in STATEMENTS:
            stmt_key = spec.key
            region = _find_region(lines, spec)
            if not region:
                continue
            cols = _read_periods(region, default_md, stmt_key=stmt_key,
                                 annual_filing=(filing.get("form") == "10-K"))
            if not cols:
                continue
            default_scale = _read_scale(region)   # from the "(in …)" parenthetical

            # Start parsing data rows only after the year header line, and track
            # the nearest section header so ambiguous sub-labels ("Basic",
            # "Diluted") can be qualified and scaled correctly.
            data_start = 1
            for k, (_pg, ln) in enumerate(region):
                if _period_row_years(ln):
                    data_start = k + 1
                    break

            section = ""
            for pg, raw_line in region[data_start:]:
                label, vals = _split_label_and_values(raw_line)
                if not label:
                    continue
                if not vals:
                    section = label        # a header row -> remember it
                    continue
                if len(vals) != len(cols):
                    continue  # column-count mismatch -> don't guess the mapping
                # skip a wrapped footnote/continuation fragment ("…respectively)",
                # a dangling ")") that isn't a real line item
                _lbl = label.strip().lower().strip("().,")
                if not any(c.isalpha() for c in label) or _lbl in ("respectively",):
                    continue

                unit, scale = _unit_and_scale(section, label, default_scale)

                for col, tok in enumerate(vals):
                    pc = cols[col]
                    value, neg, dash = _parse_number(tok)
                    col_scale = scale
                    # A share COUNT printed with 8+ integer digits (e.g. Vertiv's
                    # 381,712,181) is a FULL count, not a figure in the statement's
                    # money scale — tagging it 'millions' would give ~1e14 shares.
                    if unit == "shares" and len(re.sub(r"\D", "", tok.split(".")[0])) >= 8:
                        col_scale = "units"
                    # GROUNDING: the printed token must appear on its page.
                    grounded = _is_grounded(tok, page_text.get(pg, ""), dash)
                    facts.append(Fact(
                        company=entity.title, cik=entity.cik, ticker=entity.ticker,
                        filing_stem=stem, form=filing["form"],
                        accession_no=filing["accession_no"],
                        statement=stmt_key, raw_label=label,
                        period=pc.label, period_type=pc.period_type,
                        period_kind=pc.kind, period_end=pc.period_end,
                        # an assumed day is exact only for the filing's own period
                        period_end_inferred=bool(pc.inferred and pc.period_end != rd[:10]),
                        fiscal_year=pc.year, fiscal_quarter=pc.quarter,
                        months=pc.months,
                        value=value, unit=unit, scale=col_scale,
                        is_negative=neg, is_dash=dash, gaap_flag=True,
                        page_no=pg, raw_cell_text=tok,
                        extraction_method="pdfplumber_text",
                        grounding_ok=grounded,
                        confidence=0.98 if grounded else 0.0,
                        section=section,
                        source_line=raw_line.strip(),
                    ))
    return facts


def facts_to_dicts(facts: list[Fact]) -> list[dict]:
    return [asdict(f) for f in facts]


if __name__ == "__main__":
    import config
    import edgar

    cands, _ = edgar.resolve_entity("Tesla")
    ent = cands[0]
    filing = {"form": "10-K", "accession_no": "0001628280-25-000000"}
    pdf = config.FILINGS_DIR / "TSLA_10K_2025-12-31.pdf"

    facts = extract_facts(pdf, ent, filing)
    print(f"extracted {len(facts)} facts")
    ungrounded = [f for f in facts if not f.grounding_ok]
    print(f"grounded: {len(facts) - len(ungrounded)}/{len(facts)}  "
          f"(ungrounded: {len(ungrounded)})")

    # spot-check the metrics we care about
    wanted = [
        ("operations", "Total revenues"),
        ("operations", "Income from operations"),
        ("operations", "Net income attributable to common stockholders"),
        ("balance_sheet", "Cash and cash equivalents"),
    ]
    for stmt, lab in wanted:
        hits = [f for f in facts if f.statement == stmt and f.raw_label == lab]
        if hits:
            cells = "  ".join(f"{f.period}={f.raw_cell_text}" for f in hits)
            print(f"  [{stmt}] {lab}: {cells}  (p{hits[0].page_no}, {hits[0].scale})")
        else:
            print(f"  [{stmt}] {lab}: NOT FOUND")
