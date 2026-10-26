"""Unit tests for core.py — the question->answer engine.

Focus is the trust spine:
  * deterministic query understanding (offline fallback),
  * period/scale formatting,
  * label disambiguation in find_fact (the reused-label collisions),
  * honest gating in answer(): answered / hedged / refused,
  * the LLM phrasing step never silently drifts off the grounded sentence.

All tests run offline (no_llm) and against the in-memory sample Fact Store
(banked) with deterministic entity resolution (fake_resolve). The LLM and the
network are never touched.
"""
from __future__ import annotations

import pytest

import bank
import core

CIK = 1318605          # Tesla's CIK, as in conftest.sample_facts


# --------------------------------------------------------------------------- #
# _period_matches: the annual/quarterly guard at the lookup layer
# --------------------------------------------------------------------------- #
def test_period_matches_annual_income(mkfact):
    spec = core.METRICS["revenue"]
    annual = mkfact(period_kind="annual", fiscal_year=2025)
    assert core._period_matches(annual, spec, 2025, "annual", None)
    # an annual request must NOT accept a quarterly fact
    q = mkfact(period_kind="quarter", fiscal_year=2025, fiscal_quarter=2)
    assert not core._period_matches(q, spec, 2025, "annual", None)


def test_period_matches_quarter_requires_quarter_fact(mkfact):
    spec = core.METRICS["revenue"]
    q2 = mkfact(period_kind="quarter", fiscal_year=2026, fiscal_quarter=2)
    assert core._period_matches(q2, spec, 2026, "quarter", 2)
    assert not core._period_matches(q2, spec, 2026, "quarter", 3)   # wrong qtr
    # a quarterly request must NOT accept an annual fact (the mixing guard)
    annual = mkfact(period_kind="annual", fiscal_year=2026)
    assert not core._period_matches(annual, spec, 2026, "quarter", 2)


def test_period_matches_balance_sheet_instants(mkfact):
    spec = core.METRICS["cash"]            # statement == balance_sheet
    ye = mkfact(statement="balance_sheet", period_kind="instant",
                fiscal_year=2025, period_end="2025-12-31")
    assert core._period_matches(ye, spec, 2025, "annual", None)       # year-end
    qe = mkfact(statement="balance_sheet", period_kind="instant",
                fiscal_year=2026, period_end="2026-06-30")
    assert core._period_matches(qe, spec, 2026, "quarter", 2)         # Q2-end
    assert not core._period_matches(qe, spec, 2026, "annual", None)   # not year-end


# --------------------------------------------------------------------------- #
# _fmt_money: scale-aware rendering
# --------------------------------------------------------------------------- #
def test_fmt_money_units_two_decimals(mkfact):
    f = mkfact(value=1.08, scale="units", unit="USD")
    assert core._fmt_money(f) == "$1.08"


def test_fmt_money_millions_thousands_separator(mkfact):
    f = mkfact(value=94_827.0, scale="millions")
    assert core._fmt_money(f) == "$94,827 million"


def test_fmt_money_units_rounds_to_cents(mkfact):
    assert core._fmt_money(mkfact(value=1.2, scale="units")) == "$1.20"


# --------------------------------------------------------------------------- #
# _understand_fallback: deterministic offline parser
# --------------------------------------------------------------------------- #
def test_understand_diluted_eps_longest_cue_wins_over_bare_eps():
    """'diluted eps' must map to eps_diluted, not the generic eps_basic."""
    q = core._understand_fallback("What was the diluted EPS in 2025?")
    assert q.metric == "eps_diluted"


def test_understand_bare_eps_maps_to_basic():
    q = core._understand_fallback("What was EPS in 2025?")
    assert q.metric == "eps_basic"


def test_understand_revenue_and_sales_map_to_revenue():
    assert core._understand_fallback("What was revenue?").metric == "revenue"
    assert core._understand_fallback("How much in sales?").metric == "revenue"


def test_understand_growth_cues_set_growth_calc():
    for text in ("How did revenue grow?",
                 "What was the change in revenue?",
                 "revenue year over year"):
        assert core._understand_fallback(text).calc == "growth", text


def test_understand_grow_from_year_to_year_extracts_both_years():
    q = core._understand_fallback("How did Tesla's revenue grow from 2024 to 2025?")
    assert q.calc == "growth"
    assert q.years == [2024, 2025]


def test_understand_operating_margin_is_operating_income_plus_margin():
    q = core._understand_fallback("What was Tesla's operating margin in 2025?")
    assert q.metric == "operating_income"
    assert q.calc == "margin"


def test_understand_risks_question_is_prose_with_no_metric():
    q = core._understand_fallback("What are the biggest risks Tesla faces?")
    assert q.kind == "prose"
    assert q.metric is None


def test_understand_extracts_four_digit_years():
    q = core._understand_fallback("revenue in 2023 and 2025")
    assert q.years == [2023, 2025]


def test_understand_company_strips_possessive_and_skips_stopwords():
    q = core._understand_fallback("What was Tesla's revenue in 2025?")
    # 'What' is a stopword; "Tesla's" -> "Tesla" (possessive dropped).
    assert q.company_mention == "Tesla"


def test_understand_picks_all_caps_ticker():
    q = core._understand_fallback("How much revenue did AAPL report in 2025?")
    assert q.company_mention == "AAPL"


# --------------------------------------------------------------------------- #
# find_fact: exact-label match + scale/label disambiguation
# --------------------------------------------------------------------------- #
def test_find_fact_revenue_found(banked):
    fact, status, cands = core.find_fact(CIK, "revenue", 2025)
    assert status == "found"
    assert fact["value"] == 94_827.0
    assert cands == []


def test_find_fact_net_income_picks_attributable_not_total(banked):
    """The reused-concept collision: a bare total 'Net income' look-alike must
    NOT shadow the 'attributable to common stockholders' row the metric maps to."""
    fact, status, _ = core.find_fact(CIK, "net_income", 2025)
    assert status == "found"
    assert fact["raw_label"] == "Net income attributable to common stockholders"
    assert fact["value"] == 3_794.0           # NOT 3_855.0 (total Net income)


def test_find_fact_eps_diluted_picks_per_share_not_share_count(banked):
    """The reused-LABEL collision: 'Diluted' is both the $/share EPS row and a
    weighted-share-count row (millions). The scale filter must select $/share."""
    fact, status, _ = core.find_fact(CIK, "eps_diluted", 2025)
    assert status == "found"
    assert fact["scale"] == "units"
    assert fact["value"] == 1.08              # NOT 3_528.0 (share count)


def test_find_fact_year_not_in_store_is_not_found(banked):
    fact, status, cands = core.find_fact(CIK, "revenue", 2019)
    assert status == "not_found"
    assert fact is None
    assert cands == []


def test_find_fact_duplicate_label_is_ambiguous(monkeypatch, mkfact):
    """Two rows sharing the SAME exact candidate label -> never silent-pick."""
    rows = [
        mkfact(raw_label="Total revenues", period="FY2025", value=1.0),
        mkfact(raw_label="Total revenues", period="FY2025", value=2.0),
    ]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    fact, status, cands = core.find_fact(CIK, "revenue", 2025)
    assert status == "ambiguous" and fact is None


def test_find_fact_non_synonym_lookalike_is_not_found(monkeypatch, mkfact):
    """A look-alike that is NOT one of the metric's exact synonyms must fail
    safe (not_found), never be guessed into an answer."""
    rows = [mkfact(raw_label="Automotive total revenues", period="FY2025")]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    fact, status, _ = core.find_fact(CIK, "revenue", 2025)
    assert status == "not_found" and fact is None


def test_find_fact_revenue_synonym_net_sales(monkeypatch, mkfact):
    """A filer that labels revenue 'Net sales' (Hershey) still resolves."""
    rows = [mkfact(raw_label="Net sales", period="FY2025", value=11_200.0)]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    fact, status, _ = core.find_fact(CIK, "revenue", 2025)
    assert status == "found" and fact["value"] == 11_200.0


# --------------------------------------------------------------------------- #
# answer(): end-to-end offline gating
# --------------------------------------------------------------------------- #
def test_answer_level_revenue_2025(banked, fake_resolve, no_llm):
    a = core.answer("What was Tesla's revenue in 2025?")
    assert a.status == "answered"
    assert a.value == 94_827.0
    # trace carries a real citation: page + printed cell
    assert a.trace.citations, "expected at least one citation"
    c = a.trace.citations[0]
    assert c["page"] == 82
    assert c["cell"] == "94,827"


def test_answer_growth_revenue_2024_to_2025(banked, fake_resolve, no_llm):
    a = core.answer("How did Tesla's revenue grow from 2024 to 2025?")
    assert a.status == "answered"
    assert a.value == pytest.approx(-2.93, abs=0.02)
    # growth cites both periods
    periods = {c["period"] for c in a.trace.citations}
    assert periods == {"FY2024", "FY2025"}


def test_answer_operating_margin_2025(banked, fake_resolve, no_llm):
    a = core.answer("What was Tesla's operating margin in 2025?")
    assert a.status == "answered"
    assert a.value == pytest.approx(4.59, abs=0.02)


def test_answer_net_income_surfaces_alt_note(banked, fake_resolve, no_llm):
    a = core.answer("What was Tesla's net income in 2025?")
    assert a.status == "answered"
    assert a.value == 3_794.0
    # the honest caveat about the total 'Net income' look-alike must be shown
    assert 'total "Net income"' in a.text


def test_answer_out_of_corpus_year_is_refused(banked, fake_resolve, no_llm):
    a = core.answer("What was Tesla's net income in 2019?")
    assert a.status == "refused"
    assert a.value is None


def test_answer_unknown_company_is_refused(banked, fake_resolve, no_llm):
    a = core.answer("What was Microsoft's revenue in 2025?")
    assert a.status == "refused"


def test_answer_ambiguous_company_is_hedged(banked, fake_resolve, no_llm):
    a = core.answer("What was AMBIG's revenue in 2025?")
    assert a.status == "hedged"


# --------------------------------------------------------------------------- #
# _understand_fallback: spelled-out "earnings per share" phrasing.
#
# Regression guard for a fixed bug: metric selection must pick the GLOBALLY
# longest matching cue across metrics, so the specific phrase "earnings per
# share" (eps_basic) beats the shorter substring cue "earnings" (net_income).
# Before the fix a spelled-out EPS question was misclassified as net income — a
# wrong-metric, wrong-unit ($ millions vs $/share) answer.
# --------------------------------------------------------------------------- #
def test_understand_spelled_out_diluted_eps_maps_to_eps_diluted():
    q = core._understand_fallback("What was diluted earnings per share in 2025?")
    assert q.metric == "eps_diluted"


def test_understand_spelled_out_eps_maps_to_eps_basic():
    q = core._understand_fallback("What was earnings per share in 2025?")
    assert q.metric == "eps_basic"


# --------------------------------------------------------------------------- #
# _phrase: offline it returns the grounded sentence verbatim
# --------------------------------------------------------------------------- #
def test_phrase_offline_returns_grounded_sentence_unchanged(no_llm):
    tr = core.Trace()
    grounded = "Tesla, Inc.'s Total revenue for 2025 was $94,827 million."
    assert core._phrase("anything?", grounded, tr) == grounded


# --------------------------------------------------------------------------- #
# _phrase: LLM-ON drift guard (the trust invariant — the LLM may restate but
# never drop/alter a cited number). Exercises the branch the offline suite
# never reaches.
# --------------------------------------------------------------------------- #
def test_phrase_llm_on_keeps_output_when_all_cells_present(monkeypatch, mkfact):
    import llm
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(
        llm, "complete",
        lambda sys, user, max_tokens=200: "Revenue came in at $94,827 million in 2025.")
    tr = core.Trace()
    tr.cite(mkfact(raw_cell_text="94,827"))
    out = core._phrase("What was revenue?",
                       "Tesla's revenue for 2025 was $94,827 million.", tr)
    # every number in the draft — the year included — must survive the rewrite
    assert out == "Revenue came in at $94,827 million in 2025."


def test_phrase_llm_on_falls_back_when_cited_cell_dropped(monkeypatch, mkfact):
    """If the LLM phrasing loses a cited figure, _phrase MUST discard it and
    return the code-grounded sentence (no silent numeric drift)."""
    import llm
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(
        llm, "complete",
        lambda sys, user, max_tokens=200: "Revenue was strong this year.")
    tr = core.Trace()
    tr.cite(mkfact(raw_cell_text="94,827"))
    grounded = "Tesla's revenue for 2025 was $94,827 million."
    assert core._phrase("What was revenue?", grounded, tr) == grounded


def test_phrase_llm_on_exception_falls_back_to_grounded(monkeypatch, mkfact):
    import llm

    def boom(*a, **k):
        raise RuntimeError("api down")
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "complete", boom)
    tr = core.Trace()
    tr.cite(mkfact(raw_cell_text="94,827"))
    grounded = "Tesla's revenue for 2025 was $94,827 million."
    assert core._phrase("q", grounded, tr) == grounded


# --------------------------------------------------------------------------- #
# answer()/compute: refusal & hedge gating that the happy-path suite omits
# --------------------------------------------------------------------------- #
def test_answer_growth_refuses_on_scale_mismatch(monkeypatch, fake_resolve,
                                                 no_llm, mkfact):
    """Trust invariant: never compute growth across mismatched units/scales."""
    rows = [
        mkfact(raw_label="Total revenues", period="FY2025", fiscal_year=2025,
               value=94_827.0, raw_cell_text="94,827", scale="millions"),
        mkfact(raw_label="Total revenues", period="FY2024", fiscal_year=2024,
               value=97.69, raw_cell_text="97.69", scale="units"),
    ]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    a = core.answer("How did Tesla's revenue grow from 2024 to 2025?")
    assert a.status == "refused"
    assert a.value is None


def test_answer_growth_missing_one_period_refuses(monkeypatch, fake_resolve,
                                                  no_llm, mkfact):
    rows = [
        mkfact(raw_label="Total revenues", period="FY2025", fiscal_year=2025,
               value=94_827.0, raw_cell_text="94,827"),
    ]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    a = core.answer("How did Tesla's revenue grow from 2024 to 2025?")
    assert a.status == "refused"


def test_answer_margin_hedges_for_non_margin_metric(banked, fake_resolve, no_llm):
    """revenue is margin_eligible=False -> a revenue 'margin' must be hedged,
    never a nonsense 100% computed silently."""
    a = core.answer("What was Tesla's revenue margin in 2025?")
    assert a.status == "hedged"
    assert a.value is None


def test_answer_level_ambiguous_label_is_hedged(monkeypatch, fake_resolve,
                                                no_llm, mkfact):
    """find_fact ambiguity must surface as a hedge end-to-end, not a silent pick."""
    rows = [
        mkfact(raw_label="Total revenues", period="FY2025", value=1.0,
               statement="operations"),
        mkfact(raw_label="Total revenues", period="FY2025", value=2.0,
               statement="operations"),
    ]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    a = core.answer("What was Tesla's revenue in 2025?")
    assert a.status == "hedged"
    assert a.value is None


def _sample_prose(cik):
    return [
        {"section": "risk_factors", "section_name": "Item 1A. Risk Factors",
         "page_start": 23, "page_end": 23, "filing_stem": "TSLA_10K_2025-12-31",
         "form": "10-K", "cik": cik, "ticker": "TSLA", "company": "Tesla, Inc.",
         "text": "We face intense competition in the automotive market, and "
                 "competitors may introduce electric vehicles that reduce our "
                 "sales and margins."},
        {"section": "mdna", "section_name": "Item 7. MD&A",
         "page_start": 54, "page_end": 54, "filing_stem": "TSLA_10K_2025-12-31",
         "form": "10-K", "cik": cik, "ticker": "TSLA", "company": "Tesla, Inc.",
         "text": "Automotive revenue decreased primarily due to lower average "
                 "selling prices and fewer deliveries during the period."},
    ]


def test_answer_prose_question_routes_and_grounds(monkeypatch, banked,
                                                   fake_resolve, no_llm):
    # A prose question routes to the retriever, returns an answered/grounded
    # result (offline: the top passage verbatim) with a page citation.
    import bank as _bank
    from conftest import CIK
    monkeypatch.setattr(_bank, "load_prose", lambda path=None: _sample_prose(CIK))
    a = core.answer("What are the biggest competition risks Tesla faces?")
    assert a.status == "answered"
    assert a.value is None
    assert a.passages and a.trace.citations
    assert a.trace.citations[0]["page"] in (23, 54)


def test_answer_prose_refuses_when_no_prose_banked(monkeypatch, banked,
                                                   fake_resolve, no_llm):
    import bank as _bank
    monkeypatch.setattr(_bank, "load_prose", lambda path=None: [])
    a = core.answer("What are the biggest risks Tesla faces?")
    assert a.status == "refused"


# --------------------------------------------------------------------------- #
# Interim periods end-to-end: quarterly request uses the quarterly fact and
# never substitutes an annual one (the annual/quarterly guard).
# --------------------------------------------------------------------------- #
def test_answer_quarter_uses_quarter_fact(monkeypatch, fake_resolve, no_llm,
                                          mkfact):
    import bank
    rows = [
        mkfact(raw_label="Total revenues", period="2026Q2",
               period_kind="quarter", fiscal_year=2026, fiscal_quarter=2,
               months=3, value=28_236.0, raw_cell_text="28,236"),
        mkfact(raw_label="Total revenues", period="FY2025",
               period_kind="annual", fiscal_year=2025, value=94_827.0,
               raw_cell_text="94,827"),
    ]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    a = core.answer("What was Tesla's revenue in Q2 2026?")
    assert a.status == "answered" and abs(a.value - 28_236) < 1
    assert a.trace.citations[0]["period"] == "2026Q2"
    # a quarter with no banked quarterly fact refuses (no annual substitution)
    b = core.answer("What was Tesla's revenue in Q2 2025?")
    assert b.status == "refused"


# --------------------------------------------------------------------------- #
# Generalization fixes (from the AVGO/ASTS/HSY multi-company test)
# --------------------------------------------------------------------------- #
def test_find_fact_duplicate_same_value_collapses_prefers_10k(monkeypatch, mkfact):
    """The SAME figure reported in both a 10-K and a 10-Q's prior-year column is
    consistent, not a conflict -> found (prefer the 10-K), never hedged."""
    rows = [
        mkfact(raw_label="Total assets", statement="balance_sheet",
               period_kind="instant", period="ASOF2025-11-02",
               period_end="2025-11-02", fiscal_year=2025, months=None,
               form="10-Q", value=171_092.0),
        mkfact(raw_label="Total assets", statement="balance_sheet",
               period_kind="instant", period="ASOF2025-11-02",
               period_end="2025-11-02", fiscal_year=2025, months=None,
               form="10-K", value=171_092.0),
    ]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    fact, status, _ = core.find_fact(CIK, "total_assets", 2025)
    assert status == "found" and fact["form"] == "10-K"


def test_find_fact_duplicate_differing_value_is_ambiguous(monkeypatch, mkfact):
    """Differing values for the same line+period (a restatement) stay ambiguous."""
    rows = [
        mkfact(raw_label="Total assets", statement="balance_sheet",
               period_kind="instant", period_end="2025-12-31", fiscal_year=2025,
               months=None, form="10-K", value=100.0),
        mkfact(raw_label="Total assets", statement="balance_sheet",
               period_kind="instant", period_end="2025-12-31", fiscal_year=2025,
               months=None, form="10-K/A", value=110.0),
    ]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    _, status, _ = core.find_fact(CIK, "total_assets", 2025)
    assert status == "ambiguous"


def test_find_fact_label_match_is_case_insensitive(monkeypatch, mkfact):
    """ASTS renders balance subtotals in caps ("TOTAL ASSETS")."""
    rows = [mkfact(raw_label="TOTAL ASSETS", statement="balance_sheet",
                   period_kind="instant", period_end="2025-12-31",
                   fiscal_year=2025, months=None, value=5_014_387.0)]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    fact, status, _ = core.find_fact(CIK, "total_assets", 2025)
    assert status == "found" and fact["value"] == 5_014_387.0


def test_find_fact_operating_cash_flow_metric(monkeypatch, mkfact):
    rows = [mkfact(raw_label="Net cash provided by operating activities",
                   statement="cash_flows", period_kind="annual",
                   fiscal_year=2025, value=27_537.0)]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    fact, status, _ = core.find_fact(CIK, "operating_cash_flow", 2025)
    assert status == "found" and fact["value"] == 27_537.0


def test_find_fact_balance_annual_matches_non_december_year_end(monkeypatch, mkfact):
    """Broadcom's fiscal year ends in November: an 'annual' cash request must
    match the Nov year-end instant, not require Dec 31."""
    rows = [mkfact(raw_label="Cash and cash equivalents",
                   statement="balance_sheet", period_kind="instant",
                   period_end="2025-11-02", fiscal_year=2025, months=None,
                   value=16_178.0)]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    fact, status, _ = core.find_fact(CIK, "cash", 2025)
    assert status == "found" and fact["value"] == 16_178.0


# --------------------------------------------------------------------------- #
# Label-variant fixes from the AAL / DIS / BE round
# --------------------------------------------------------------------------- #
def test_find_fact_airline_revenue_synonym(monkeypatch, mkfact):
    rows = [mkfact(raw_label="Total operating revenues", value=54_633.0)]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    fact, status, _ = core.find_fact(CIK, "revenue", 2025)
    assert status == "found" and fact["value"] == 54_633.0


def test_find_fact_footnote_marker_on_label(monkeypatch, mkfact):
    # Bloom Energy renders "Total revenue1" (footnote glued on).
    rows = [mkfact(raw_label="Total revenue1", value=2_023_994.0)]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    fact, status, _ = core.find_fact(CIK, "revenue", 2025)
    assert status == "found" and fact["value"] == 2_023_994.0


def test_find_fact_balance_annual_uses_company_fiscal_year_end(monkeypatch, mkfact):
    # Disney: annual operations end in Sept -> a balance Sept instant is the
    # year-end for an 'annual' cash request (month 9 would otherwise look like a
    # quarter-end under the fallback heuristic).
    rows = [
        mkfact(raw_label="Total revenues", statement="operations",
               period_kind="annual", fiscal_year=2025, period_end="2025-09-27",
               value=94_425.0),
        mkfact(raw_label="Cash and cash equivalents", statement="balance_sheet",
               period_kind="instant", fiscal_year=2025, period_end="2025-09-27",
               months=None, value=5_695.0),
    ]
    monkeypatch.setattr(bank, "load_facts", lambda path=None: rows)
    fact, status, _ = core.find_fact(CIK, "cash", 2025)
    assert status == "found" and fact["value"] == 5_695.0
