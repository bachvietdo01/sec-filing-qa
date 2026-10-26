"""Regression tests for the bugs found by the 5-agent stress test (hermetic)."""
from __future__ import annotations
import bank, core, edgar, extract


def _labels(cols):
    return [(c.label, c.kind, c.period_end) for c in cols]


# --- extract: interim period headers ------------------------------------- #
def test_periods_durations_and_dates_on_separate_lines():
    # ASTS/BE style: "Three/Six Months ended" on one line, dates on the next,
    # bare years on the period row. Must split into 3-month and 6-month groups.
    cols = extract._read_periods([
        (6, "Consolidated Statements of Operations"),
        (6, "For the Three Months ended For the Six Months ended"),
        (6, "June 30, June 30,"),
        (6, "2026 2025 2026 2025"),
    ])
    assert _labels(cols) == [
        ("2026Q2", "quarter", "2026-06-30"), ("2025Q2", "quarter", "2025-06-30"),
        ("2026YTD6M", "ytd", "2026-06-30"), ("2025YTD6M", "ytd", "2025-06-30")]


def test_periods_inline_three_six_months_hershey():
    cols = extract._read_periods([
        (7, "CONSOLIDATED STATEMENTS OF INCOME"),
        (7, "Three Months Ended Six Months Ended"),
        (7, "June 28, 2026 June 29, 2025 June 28, 2026 June 29, 2025"),
    ])
    assert [c.kind for c in cols] == ["quarter", "quarter", "ytd", "ytd"]
    assert [c.months for c in cols] == [3, 3, 6, 6]


def test_periods_quarter_and_nine_months_disney():
    cols = extract._read_periods([
        (5, "Consolidated Statements of Income"),
        (5, "Quarter Ended Nine Months Ended"),
        (5, "June 28, 2026 June 29, 2025 June 28, 2026 June 29, 2025"),
    ])
    assert [c.months for c in cols] == [3, 3, 9, 9]
    assert [c.kind for c in cols] == ["quarter", "quarter", "ytd", "ytd"]


def test_unit_scale_weighted_avg_shares_used_in_per_share_calc():
    # AVGO: share-count rows labeled "Basic"/"Diluted" under this section must be
    # 'shares' scale, NOT 'units' (else EPS grabs the share count).
    unit, scale = extract._unit_and_scale(
        "Weighted-average shares used in per share calculation", "Basic", "millions")
    assert (unit, scale) == ("shares", "millions")


# --- core: metric/routing fixes ------------------------------------------ #
_ENT = edgar.Entity(1, "X", "X Co")


def _bank(monkeypatch, rows):
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)


def test_eps_magnitude_guard_rejects_share_count(monkeypatch, mkfact):
    # a 'Basic' units row of 4,712 (a mis-scaled share count) must NOT be EPS.
    rows = [mkfact(cik=1, raw_label="Basic", scale="units", value=4712.0,
                   period_kind="annual", fiscal_year=2025)]
    _bank(monkeypatch, rows)
    _f, status, _ = core.find_fact(1, "eps_basic", 2025)
    assert status == "not_found"


def test_net_income_prefers_attributable(monkeypatch, mkfact):
    rows = [
        mkfact(cik=1, raw_label="Net income", value=13431.0,
               period_kind="annual", fiscal_year=2025),
        mkfact(cik=1, raw_label="Net income attributable to The Walt Disney Company",
               value=12404.0, period_kind="annual", fiscal_year=2025),
    ]
    _bank(monkeypatch, rows)
    f, status, _ = core.find_fact(1, "net_income", 2025)
    assert status == "found" and f["value"] == 12404.0


def test_operating_income_loss_qualifier(monkeypatch, mkfact):
    rows = [mkfact(cik=1, raw_label="Operating profit (loss)", value=1829.7,
                   period_kind="annual", fiscal_year=2025)]
    _bank(monkeypatch, rows)
    f, status, _ = core.find_fact(1, "operating_income", 2025)
    assert status == "found" and f["value"] == 1829.7


def test_multi_year_answer(monkeypatch, mkfact, no_llm):
    rows = [mkfact(cik=1, raw_label="Total revenues", value=v, period=f"FY{y}",
                   period_kind="annual", fiscal_year=y, raw_cell_text=str(v))
            for y, v in [(2023, 96773.0), (2024, 97690.0), (2025, 94827.0)]]
    _bank(monkeypatch, rows)
    a = core.answer("revenue for 2023, 2024 and 2025?", entity=_ENT)
    assert a.status == "answered"
    assert len(a.trace.citations) == 3
    assert "96,773" in a.text and "97,690" in a.text and "94,827" in a.text


def test_nongaap_request_is_refused(no_llm):
    a = core.answer("What was adjusted EBITDA in 2025?", entity=_ENT)
    assert a.status == "hedged" and "non-gaap" in a.text.lower()


def test_resolve_entity_collapses_share_classes(monkeypatch):
    rows = [{"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
            {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc."}]
    monkeypatch.setattr(edgar, "_load_tickers", lambda: rows)
    cands, status = edgar.resolve_entity("Alphabet")
    assert status == "exact" and len(cands) == 1 and cands[0].cik == 1652044
