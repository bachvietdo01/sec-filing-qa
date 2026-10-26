"""Unit + integration tests for extract.py — the deterministic PDF->facts spine.

extract.py is trust-critical: every number must come out of the PDF by
deterministic parsing and be grounded (printed token present verbatim on its
page). These tests pin the pure helpers with table-style cases and then run the
full extractor against the real rendered 10-K, asserting the trust invariants
(grounding, label disambiguation, scale/period handling) rather than just the
happy path.
"""
from __future__ import annotations

import pytest

import extract


# --------------------------------------------------------------------------- #
# _parse_number
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tok, value, is_neg, is_dash",
    [
        # plain thousands-separated integer
        ("94,827", 94_827.0, False, False),
        ("0", 0.0, False, False),
        ("494", 494.0, False, False),
        # parenthesised negatives
        ("(338)", -338.0, True, False),
        ("(1,234)", -1_234.0, True, False),
        # dash family -> zero, flagged as dash
        ("—", 0.0, False, True),   # em dash
        ("–", 0.0, False, True),   # en dash
        ("-", 0.0, False, True),   # hyphen
        # decimals (per-share values)
        ("1.08", 1.08, False, False),
        ("1.18", 1.18, False, False),
        # $-prefixed
        ("$65,821", 65_821.0, False, False),
        # parenthesised decimal negative
        ("(0.50)", -0.50, True, False),
    ],
)
def test_parse_number_table(tok, value, is_neg, is_dash):
    val, neg, dash = extract._parse_number(tok)
    assert val == pytest.approx(value)
    assert neg is is_neg
    assert dash is is_dash


def test_parse_number_dash_is_zero_not_none():
    # A dash must become a real 0.0 (so downstream math is safe), flagged dash.
    val, neg, dash = extract._parse_number("—")
    assert val == 0.0
    assert dash is True
    assert neg is False


def test_parse_number_non_numeric_raises():
    # Guards the grounding contract: a non-number token is never silently 0.
    with pytest.raises(ValueError):
        extract._parse_number("abc")


# --------------------------------------------------------------------------- #
# _split_label_and_values
# --------------------------------------------------------------------------- #
def test_split_plain_three_columns():
    label, vals = extract._split_label_and_values(
        "Total revenues 94,827 97,690 96,773")
    assert label == "Total revenues"
    assert vals == ["94,827", "97,690", "96,773"]


def test_split_drops_interleaved_currency_marks():
    # Bare "$" tokens are currency marks, not columns: exactly 3 numbers remain.
    label, vals = extract._split_label_and_values(
        "Automotive sales $ 65,821 $ 72,480 $ 78,509")
    assert label == "Automotive sales"
    assert vals == ["65,821", "72,480", "78,509"]
    assert "$" not in vals
    assert len(vals) == 3


def test_split_keeps_dash_as_a_value():
    # A dash column is a real (zero) data cell, so it must be kept, not dropped.
    label, vals = extract._split_label_and_values(
        "Restructuring and other 494 684 —")
    assert label == "Restructuring and other"
    assert vals == ["494", "684", "—"]
    assert len(vals) == 3


def test_split_keeps_parenthesised_negative():
    label, vals = extract._split_label_and_values(
        "Other income expense net (338) 530 172")
    assert label == "Other income expense net"
    assert vals == ["(338)", "530", "172"]


def test_split_header_only_row_has_no_values():
    # A section header carries the label and no numeric columns.
    assert extract._split_label_and_values("Revenues") == ("Revenues", [])
    assert extract._split_label_and_values("Current assets") == (
        "Current assets", [])


def test_split_empty_line():
    assert extract._split_label_and_values("") == ("", [])


def test_split_multiword_label_preserved():
    label, vals = extract._split_label_and_values(
        "Net income attributable to common stockholders 3,794 7,091 14,974")
    assert label == "Net income attributable to common stockholders"
    assert vals == ["3,794", "7,091", "14,974"]


# --------------------------------------------------------------------------- #
# _unit_and_scale  (the reused-label disambiguation invariant)
# --------------------------------------------------------------------------- #
def test_unit_scale_per_share_section_is_dollars_units():
    unit, scale = extract._unit_and_scale(
        "Net income per share of common stock attributable to "
        "common stockholders", "Diluted")
    assert (unit, scale) == ("USD", "units")


def test_unit_scale_shares_branch_wins_over_per_share():
    # The weighted-average-shares header ALSO contains "per share"; the shares
    # branch must win so a share count is never mislabelled as dollars.
    section = ("Weighted average shares used in computing net income per "
               "share of common stock")
    assert "per share" in section.lower()        # the trap is present
    unit, scale = extract._unit_and_scale(section, "Diluted")
    assert (unit, scale) == ("shares", "millions")


def test_unit_scale_shares_used_in_computing_phrase():
    unit, scale = extract._unit_and_scale(
        "Shares used in computing earnings per share", "Basic")
    assert (unit, scale) == ("shares", "millions")


def test_unit_scale_ordinary_row_is_usd_millions():
    assert extract._unit_and_scale("Revenues", "Total revenues") == (
        "USD", "millions")
    assert extract._unit_and_scale("", "Cash and cash equivalents") == (
        "USD", "millions")


def test_unit_scale_honors_statement_scale():
    # a "(in thousands)" filer: ordinary rows AND share counts take that scale,
    # but EPS stays per-share units.
    assert extract._unit_and_scale("Revenues", "Total revenues",
                                   "thousands") == ("USD", "thousands")
    assert extract._unit_and_scale(
        "Weighted average shares", "Diluted", "thousands") == (
        "shares", "thousands")
    assert extract._unit_and_scale(
        "Net income per share", "Diluted", "thousands") == ("USD", "units")


def test_read_scale_from_parenthetical():
    mk = lambda s: [(1, "Consolidated Statements of Operations"), (1, s)]
    assert extract._read_scale(mk("(in thousands, except per share data)")) == "thousands"
    assert extract._read_scale(mk("(In millions)")) == "millions"
    assert extract._read_scale(mk("(in billions)")) == "billions"
    # none stated -> safe default
    assert extract._read_scale([(1, "Consolidated Balance Sheets"),
                                (1, "December 31,")]) == "millions"


# --------------------------------------------------------------------------- #
# _read_periods  (per-column period specs; annual 10-K + interim 10-Q)
# --------------------------------------------------------------------------- #
def _labels(cols):
    return [(c.label, c.kind, c.period_end) for c in cols]


def test_periods_annual_three_years():
    cols = extract._read_periods([
        (82, "Consolidated Statements of Operations"),
        (82, "Year Ended December 31,"),
        (82, "2025 2024 2023"),
    ])
    assert _labels(cols) == [
        ("FY2025", "annual", "2025-12-31"),
        ("FY2024", "annual", "2024-12-31"),
        ("FY2023", "annual", "2023-12-31")]
    assert all(c.months == 12 for c in cols)


def test_periods_balance_sheet_two_instants():
    cols = extract._read_periods([
        (80, "Consolidated Balance Sheets"),
        (80, "December 31, December 31,"),
        (80, "2025 2024"),
    ])
    assert _labels(cols) == [
        ("ASOF2025-12-31", "instant", "2025-12-31"),
        ("ASOF2024-12-31", "instant", "2024-12-31")]


def test_periods_10q_balance_sheet_different_dates():
    # The real 10-Q trap: two instants with DIFFERENT month/day per column.
    cols = extract._read_periods([
        (6, "Consolidated Balance Sheets"),
        (6, "June 30, December 31,"),
        (6, "2026 2025"),
    ])
    assert _labels(cols) == [
        ("ASOF2026-06-30", "instant", "2026-06-30"),
        ("ASOF2025-12-31", "instant", "2025-12-31")]


def test_periods_10q_income_dual_group():
    # Two groups (3-month + 6-month YTD) over repeated years -> 4 distinct cols.
    cols = extract._read_periods([
        (7, "Consolidated Statements of Operations"),
        (7, "Three Months Ended June 30, Six Months Ended June 30,"),
        (7, "2026 2025 2026 2025"),
    ])
    assert _labels(cols) == [
        ("2026Q2", "quarter", "2026-06-30"),
        ("2025Q2", "quarter", "2025-06-30"),
        ("2026YTD6M", "ytd", "2026-06-30"),
        ("2025YTD6M", "ytd", "2025-06-30")]
    assert [c.months for c in cols] == [3, 3, 6, 6]
    assert [c.quarter for c in cols] == [2, 2, 2, 2]


def test_periods_non_december_fiscal_end():
    cols = extract._read_periods([
        (1, "Consolidated Statements of Operations"),
        (1, "Year Ended June 30,"),
        (1, "2025 2024"),
    ])
    assert _labels(cols) == [
        ("FY2025", "annual", "2025-06-30"),
        ("FY2024", "annual", "2024-06-30")]


def test_periods_no_year_row_yields_empty():
    assert extract._read_periods([(1, "Consolidated Balance Sheets")]) == []


def test_period_row_years_accepts_shapes_rejects_prose():
    # bare-year row
    assert extract._period_row_years("2025 2024 2023") == [2025, 2024, 2023]
    # inline date+year rows (NVIDIA / Vertiv conventions)
    assert extract._period_row_years("December 31, 2025 December 31, 2024") == [2025, 2024]
    assert extract._period_row_years("Jan 25, 2026 Jan 26, 2025 Jan 28, 2024") == [2026, 2025, 2024]
    # combined header line (Hershey): duration words on the same line are allowed
    assert extract._period_row_years(
        "For the years ended December 31, 2025 2024 2023") == [2025, 2024, 2023]
    # real PROSE (content words beyond dates/durations) is NOT a period row
    assert extract._period_row_years(
        "We audited the balance sheets as of December 31, 2025 and 2024") == []
    assert extract._period_row_years("Revenue $ 215,938 130,497") == []


def test_periods_combined_header_line_hershey_style():
    # Hershey: "(in thousands…)" then the duration+date+years all on one line.
    cols = extract._read_periods([
        (70, "CONSOLIDATED STATEMENTS OF INCOME"),
        (70, "(in thousands, except per share amounts)"),
        (70, "For the years ended December 31, 2025 2024 2023"),
    ])
    assert _labels(cols) == [
        ("FY2025", "annual", "2025-12-31"),
        ("FY2024", "annual", "2024-12-31"),
        ("FY2023", "annual", "2023-12-31")]


def test_periods_inline_dates_annual_nvidia_style():
    # NVIDIA: "Year Ended" + an inline date row with abbreviated months.
    cols = extract._read_periods([
        (71, "Consolidated Statements of Income"),
        (71, "(In millions, except per share data)"),
        (71, "Year Ended"),
        (71, "Jan 25, 2026 Jan 26, 2025 Jan 28, 2024"),
    ])
    assert _labels(cols) == [
        ("FY2026", "annual", "2026-01-25"),
        ("FY2025", "annual", "2025-01-26"),
        ("FY2024", "annual", "2024-01-28")]


def test_periods_inline_dates_instant_vertiv_style():
    # Vertiv balance sheet: no duration word -> two instants, dates inline.
    cols = extract._read_periods([
        (104, "CONSOLIDATED BALANCE SHEETS"),
        (104, "(Dollars in millions)"),
        (104, "December 31, 2025 December 31, 2024"),
    ])
    assert _labels(cols) == [
        ("ASOF2025-12-31", "instant", "2025-12-31"),
        ("ASOF2024-12-31", "instant", "2024-12-31")]


# --------------------------------------------------------------------------- #
# _find_region  (must pick the real statement, not the table-of-contents row)
# --------------------------------------------------------------------------- #
def _spec(key):
    return next(s for s in extract.STATEMENTS if s.key == key)


def test_find_region_skips_index_picks_real_statement():
    title = "Consolidated Statements of Operations"
    lines = [
        (1, "Index to Financial Statements"),
        (1, title),                       # index row: no year row nearby
        (1, "Report of Independent ... 82"),
        (2, "filler"), (2, "filler"), (2, "filler"), (2, "filler"),
        (2, "filler"), (2, "filler"), (2, "filler"),
        (82, title),                      # the real statement
        (82, "Year Ended December 31,"),
        (82, "2025 2024 2023"),
        (82, "Total revenues 94,827 97,690 96,773"),   # anchor 1
        (82, "Net income 3,794 7,091 14,997"),         # anchor 2
        (83, "Consolidated Balance Sheets"),           # next title closes region
        (83, "more"),
    ]
    region = extract._find_region(lines, _spec("operations"))
    assert region, "region must not be empty"
    assert region[0] == (82, title)                    # the real page, not index
    assert all(ln != "Consolidated Balance Sheets" for _pg, ln in region)
    assert any("Total revenues" in ln for _pg, ln in region)


def test_find_region_absent_title_returns_empty():
    lines = [(1, "Something Else"), (1, "2025 2024")]
    assert extract._find_region(lines, _spec("balance_sheet")) == []


def test_find_region_requires_year_near_title():
    # A title with no period row after it is treated as not-the-statement.
    title = "Consolidated Statements of Cash Flows"
    lines = [(1, title), (1, "see page 90"), (2, "no numbers here")]
    assert extract._find_region(lines, _spec("cash_flows")) == []


def test_find_region_rejects_title_without_line_item_signature():
    # Gate 3: a title + a year row but NONE of the statement's Reg S-X line items
    # must be rejected (a lookalike, not the real statement).
    title = "Consolidated Statements of Operations"
    lines = [
        (10, title),
        (10, "Year Ended December 31,"),
        (10, "2025 2024 2023"),
        (10, "Some unrelated caption 1 2 3"),
        (11, "Notes to Consolidated Financial Statements"),
    ]
    assert extract._find_region(lines, _spec("operations")) == []


def test_find_region_matches_condensed_and_synonym_titles():
    # 10-Q "Condensed" prefix + the "Income" synonym must still match operations.
    title = "CONDENSED CONSOLIDATED STATEMENTS OF INCOME"
    lines = [
        (1, title),
        (1, "Three Months Ended June 30,"),
        (1, "2025 2024"),
        (1, "Total revenues 25,500 25,182"),      # anchor 1
        (1, "Income from operations 923 1,605"),  # anchor 2
        (2, "Notes to Condensed Consolidated Financial Statements"),
    ]
    region = extract._find_region(lines, _spec("operations"))
    assert region and region[0][1] == title


def test_find_region_ignores_stray_year_in_prose_after_index():
    # REGRESSION: the "Index to Financial Statements" row for Cash Flows is
    # followed by the auditor's report, whose prose mentions a year — but there
    # is no dedicated period-header row. That stray year must NOT make the index
    # row match; the real statement (with a "2025 2024 2023" row) must win.
    title = "Consolidated Statements of Cash Flows"
    lines = [
        (76, title + " 53"),                               # index row
        (76, "Notes to Financial Statements 54"),
        (76, "Report of Independent Registered Public Accounting Firm"),
        (76, "We audited the balance sheets as of December 31, 2025 and 2024"),
        (76, "Opinion paragraph one continues here at length"),
        (76, "Opinion paragraph two continues here at length"),
        (77, "Critical audit matter description text here"),
        (77, "more auditor prose that is not a year row"),
        (78, "still more auditor prose without a year row"),
        (87, title),                                       # the REAL statement
        (87, "Year Ended December 31,"),
        (87, "2025 2024 2023"),                            # period-header row
        (87, "Net income $ 3,855 $ 7,153 $ 14,974"),
        (87, "Net cash provided by operating activities 14,747 ..."),   # anchor 1
        (87, "Net cash used in investing activities (15,478) ..."),     # anchor 2
        (88, "Notes to Consolidated Financial Statements"),
    ]
    region = extract._find_region(lines, _spec("cash_flows"))
    assert region and region[0] == (87, title)             # not the index row
    assert any("operating activities" in ln.lower() for _pg, ln in region)


# --------------------------------------------------------------------------- #
# Integration: the real rendered 10-K  (trust invariants end-to-end)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def extracted_facts():
    # module-scoped: parse the real 10-K once for all integration tests (the
    # PDF parse is the slow step). Self-contained so it needn't depend on
    # function-scoped fixtures.
    import edgar
    from conftest import TSLA_PDF, CIK
    if not TSLA_PDF.exists():
        pytest.skip(f"rendered PDF not present: {TSLA_PDF}")
    ent = edgar.Entity(cik=CIK, ticker="TSLA", title="Tesla, Inc.")
    filing = {"form": "10-K", "accession_no": "0001628280-25-000000"}
    return extract.extract_facts(TSLA_PDF, ent, filing)


def _one(facts, statement, raw_label, period):
    hits = [f for f in facts
            if f.statement == statement and f.raw_label == raw_label
            and f.period == period]
    assert len(hits) == 1, (
        f"expected exactly one {statement}/{raw_label}/{period}, got {len(hits)}")
    return hits[0]


def test_integration_every_fact_is_grounded(extracted_facts):
    # THE core invariant: no number enters the Bank unless its printed token is
    # present verbatim on its source page.
    assert extracted_facts, "extractor produced no facts"
    ungrounded = [f for f in extracted_facts if not f.grounding_ok]
    assert ungrounded == [], (
        f"{len(ungrounded)} ungrounded facts, e.g. "
        f"{ungrounded[0].raw_label!r}={ungrounded[0].raw_cell_text!r}"
        if ungrounded else "")


def test_integration_all_three_statements_present(extracted_facts):
    stmts = {x.statement for x in extracted_facts}
    assert stmts == {"operations", "balance_sheet", "cash_flows"}, (
        f"expected all three core statements, got {stmts}")


def _grounded(facts, statement):
    sub = [x for x in facts if x.statement == statement]
    assert sub, f"no facts for {statement}"
    assert all(x.grounding_ok for x in sub), f"ungrounded facts in {statement}"


# --- Statement 1 of 3: Income Statement (Operations) — hand-verified FY2025 ---
def test_integration_income_statement(extracted_facts):
    f = extracted_facts
    _grounded(f, "operations")
    assert _one(f, "operations", "Total revenues", "FY2025").value == 94_827.0
    assert _one(f, "operations", "Gross profit", "FY2025").value == 17_094.0
    assert _one(f, "operations", "Income from operations",
                "FY2025").value == 4_355.0
    assert _one(f, "operations", "Net income attributable to common "
                "stockholders", "FY2025").value == 3_794.0


# --- Statement 2 of 3: Balance Sheet — hand-verified as of 2025-12-31 ---
def test_integration_balance_sheet(extracted_facts):
    f = extracted_facts
    _grounded(f, "balance_sheet")
    asof = "ASOF2025-12-31"
    assert _one(f, "balance_sheet", "Cash and cash equivalents", asof).value == 16_513.0
    assert _one(f, "balance_sheet", "Total current assets", asof).value == 68_642.0
    assert _one(f, "balance_sheet", "Total assets", asof).value == 137_806.0
    assert _one(f, "balance_sheet", "Total liabilities", asof).value == 54_941.0
    # every balance-sheet fact is an instant (not a duration)
    assert all(x.period_kind == "instant"
               for x in f if x.statement == "balance_sheet")


# --- Statement 3 of 3: Cash Flow Statement — hand-verified FY2025 ---
def test_integration_cash_flow_statement(extracted_facts):
    f = extracted_facts
    _grounded(f, "cash_flows")
    assert _one(f, "cash_flows", "Net cash provided by operating activities",
                "FY2025").value == 14_747.0
    assert _one(f, "cash_flows", "Net cash used in investing activities",
                "FY2025").value == -15_478.0   # parenthesized -> negative
    assert _one(f, "cash_flows", "Net cash provided by financing activities",
                "FY2025").value == 1_139.0


def test_integration_period_shapes(extracted_facts):
    rev = _one(extracted_facts, "operations", "Total revenues", "FY2025")
    assert rev.period_type == "duration"
    assert rev.period_end == "2025-12-31"
    assert rev.unit == "USD" and rev.scale == "millions"

    cash = _one(extracted_facts, "balance_sheet",
                "Cash and cash equivalents", "ASOF2025-12-31")
    assert cash.period_type == "instant"
    assert cash.period_end == "2025-12-31"


def test_integration_eps_and_share_rows_disambiguated(extracted_facts):
    # The reused "Basic"/"Diluted" labels must be scaled by their section:
    # EPS rows -> dollars/units; weighted-average share rows -> shares/millions.
    eps_rows = [f for f in extracted_facts
                if f.raw_label in ("Basic", "Diluted")
                and "per share" in f.section.lower()
                and "weighted average" not in f.section.lower()]
    share_rows = [f for f in extracted_facts
                  if f.raw_label in ("Basic", "Diluted")
                  and ("weighted average shares" in f.section.lower()
                       or "shares used in computing" in f.section.lower())]

    assert eps_rows, "no EPS Basic/Diluted rows found"
    assert share_rows, "no weighted-average share Basic/Diluted rows found"

    for f in eps_rows:
        assert (f.unit, f.scale) == ("USD", "units"), (
            f"EPS row {f.raw_label}/{f.period} mis-scaled: {f.unit}/{f.scale}")
    for f in share_rows:
        assert (f.unit, f.scale) == ("shares", "millions"), (
            f"share row {f.raw_label}/{f.period} mis-scaled: "
            f"{f.unit}/{f.scale}")


def test_integration_facts_to_dicts_returns_plain_dicts(extracted_facts):
    dicts = extract.facts_to_dicts(extracted_facts)
    assert isinstance(dicts, list)
    assert len(dicts) == len(extracted_facts)
    assert all(type(d) is dict for d in dicts)
    # round-trips the required identity/provenance + grounding fields
    required = {
        "company", "cik", "ticker", "statement", "raw_label", "period",
        "period_type", "period_end", "fiscal_year", "value", "unit", "scale",
        "is_negative", "is_dash", "page_no", "raw_cell_text", "grounding_ok",
    }
    assert required <= dicts[0].keys()


def test_periods_bare_year_header_uses_report_date_fallback():
    # Disney prints only "2025 2024 2023" (no month). The Sept fiscal year-end
    # must come from the filing's report_date (default_md), not a blind Dec 31.
    cols = extract._read_periods([
        (121, "CONSOLIDATED STATEMENTS OF INCOME"),
        (121, "(in millions, except per share data)"),
        (121, "2025 2024 2023"),
    ], default_md=(9, 27))
    assert _labels(cols) == [
        ("FY2025", "annual", "2025-09-27"),
        ("FY2024", "annual", "2024-09-27"),
        ("FY2023", "annual", "2023-09-27")]
