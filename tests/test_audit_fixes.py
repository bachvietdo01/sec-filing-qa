"""Regression tests for the 10-agent adversarial trust audit.

Each test pins one reproduced finding so it can't silently return. They run
OFFLINE (llm.available -> False), exercising the deterministic spine the audit
attacked, and read the real re-extracted Bank for the extraction-level fixes
(Disney fiscal quarters, Bloom Energy annual periods, decimal-filer precision,
share/equity units).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core      # noqa: E402
import edgar     # noqa: E402
import llm        # noqa: E402


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)


def ent(ticker: str):
    cands, _ = edgar.resolve_entity(ticker)
    return cands[0]


# --------------------------------------------------------------------------- #
# Growth / series robustness
# --------------------------------------------------------------------------- #
def test_growth_repeated_year_no_crash():          # [0] crash
    a = core.answer("How did Tesla revenue change from 2025 to 2025?", entity=ent("TSLA"))
    assert a.status in ("hedged", "refused") and "nan" not in a.text.lower()


def test_growth_zero_base_no_nan():                # [1] fabricated nan%
    a = core.answer("How did AST SpaceMobile revenue change from 2023 to 2024?",
                    entity=ent("ASTS"))
    assert "nan" not in a.text.lower()


def test_growth_negative_base_flagged():           # [9] misleading % on a loss
    a = core.answer("How did AST SpaceMobile net income change from 2023 to 2024?",
                    entity=ent("ASTS"))
    assert "negative base" in a.text.lower()


def test_growth_interim_labeled_not_bare_year():   # [10][25] interim mislabel
    a = core.answer("What was Tesla revenue growth in Q2 2026 versus Q2 2025?",
                    entity=ent("TSLA"))
    assert "Q2 2025" in a.text and "Q2 2026" in a.text


def test_growth_mixed_period_refused():            # [11] period-kind mix
    a = core.answer("What was Tesla revenue growth from Q2 2026 to full year 2025?",
                    entity=ent("TSLA"))
    assert a.status == "refused"


def test_growth_multi_metric():                    # [22] dropped 2nd metric
    a = core.answer("What is the revenue and net income growth from 2023 to 2024?",
                    entity=ent("GOOG"))
    assert "revenue" in a.text.lower() and "net income" in a.text.lower()


def test_cagr_is_annualized():                     # [23] cumulative mislabeled CAGR
    a = core.answer("What is the compound annual growth rate of revenue 2023 to 2025?",
                    entity=ent("GOOG"))
    assert "compound annual" in a.text.lower() and "31.0%" not in a.text


# --------------------------------------------------------------------------- #
# Non-December fiscal quarters (Disney, FY ends September)
# --------------------------------------------------------------------------- #
def test_disney_fiscal_q3_answered():              # [2][5][12] fiscal quarter
    a = core.answer("What was Disney revenue in Q3 2025?", entity=ent("DIS"))
    assert a.status == "answered" and "Q3" in a.text


def test_disney_calendar_q2_refused():             # the June quarter is NOT Q2
    a = core.answer("What was Disney revenue in Q2 2025?", entity=ent("DIS"))
    assert a.status in ("refused", "hedged")


def test_disney_nine_month_labeled_nine():         # [3][4] 9-mo not "6 months"
    a = core.answer("What was Disney revenue in the first nine months of 2025?",
                    entity=ent("DIS"))
    assert a.status == "answered" and "9 months" in a.text


def test_disney_six_month_refused():               # H1 isn't banked -> refuse
    a = core.answer("What was Disney revenue in the first six months of 2025?",
                    entity=ent("DIS"))
    assert a.status in ("refused", "hedged")


# --------------------------------------------------------------------------- #
# Bloom Energy annual income statement (was mis-tagged 'instant')
# --------------------------------------------------------------------------- #
def test_bloom_energy_annual_revenue():            # [16] annual data invisible
    a = core.answer("What was Bloom Energy revenue in 2025?", entity=ent("BE"))
    assert a.status == "answered"


def test_bloom_energy_coverage_reports_annual():   # [17] coverage misreport
    a = core.answer("What years and periods do you have for Bloom Energy?",
                    entity=ent("BE"))
    assert "annual" in a.text.lower()


# --------------------------------------------------------------------------- #
# Metric disambiguation
# --------------------------------------------------------------------------- #
def test_per_share_routes_to_eps():                # [6] EPS -> net-income TOTAL
    a = core.answer("What was SoFi's earnings (loss) per share - diluted in 2025?",
                    entity=ent("SOFI"))
    assert "thousand" not in a.text and "million" not in a.text  # a per-share $


def test_coreweave_net_income_answered():          # [18] spurious ambiguous hedge
    a = core.answer("What was CoreWeave net income in 2025?", entity=ent("CRWV"))
    assert a.status == "answered"


def test_parent_attribution_has_no_common_caveat():  # [29] caveat mislabel
    a = core.answer("What was Disney's net income in 2025?", entity=ent("DIS"))
    assert "attributable to common" not in a.text.lower()


def test_bare_net_income_has_no_caveat():          # [32] caveat on bare line
    a = core.answer("What was Nvidia net income in 2026?", entity=ent("NVDA"))
    assert "attributable to common" not in a.text.lower()


# --------------------------------------------------------------------------- #
# Computed metrics / non-GAAP labeling
# --------------------------------------------------------------------------- #
def test_ebitda_margin_no_crash():                 # [7] KeyError on computed margin
    a = core.answer("What is Tesla EBITDA margin in 2025?", entity=ent("TSLA"))
    assert a.status in ("answered", "refused")


def test_multi_computed_labeled_nongaap():         # [8] unlabeled derived figures
    a = core.answer("What were Tesla EBITDA and free cash flow in 2025?",
                    entity=ent("TSLA"))
    assert "non-gaap" in a.text.lower()


def test_adjusted_hyphen_hedged():                 # [21] hyphen slips the guard
    a = core.answer("What was adjusted-EBITDA in 2025?", entity=ent("TSLA"))
    assert a.status == "hedged"


# --------------------------------------------------------------------------- #
# Prose routing / section scoping
# --------------------------------------------------------------------------- #
def test_risk_question_routes_to_prose():          # [13] numeric hijack
    a = core.answer("What are the main risks to Tesla's revenue?", entity=ent("TSLA"))
    assert "94,827" not in a.text


def test_missing_mdna_section_refused():           # [26] section substitution
    a = core.answer("What liquidity trends does Disney's management discuss?",
                    entity=ent("DIS"))
    assert a.status == "refused"


# --------------------------------------------------------------------------- #
# Period / year validation & adversarial robustness
# --------------------------------------------------------------------------- #
def test_pre_2000_year_refused():                  # [15] silent latest-year
    a = core.answer("What was revenue in 1999?", entity=ent("GOOG"))
    assert a.status in ("refused", "hedged")


def test_invalid_quarter_hedged():                 # [28] Q5 -> annual
    a = core.answer("What was Tesla revenue in Q5 2025?", entity=ent("TSLA"))
    assert a.status == "hedged"


def test_trailing_twelve_months_hedged():          # [28] TTM -> annual
    a = core.answer("What was Tesla trailing twelve months revenue?", entity=ent("TSLA"))
    assert a.status == "hedged"


def test_understand_none_is_safe():                # [33] AttributeError on None
    assert core.understand(None) is not None


def test_multiword_company_mention_resolves():     # [31] mention collapse
    a = core.answer("What was American Airlines revenue in 2025?")
    assert a.status == "answered" and "American Airlines" in a.text


# --------------------------------------------------------------------------- #
# Grounding precision & unit integrity
# --------------------------------------------------------------------------- #
def test_decimal_filer_displayed_verbatim():       # [24][27][30][34] rounding
    a = core.answer("What was VRT revenue in 2025?", entity=ent("VRT"))
    cell = a.trace.citations[0]["cell"]
    assert cell in a.text          # the displayed figure is the filing's own cell


def test_total_equity_unit_is_usd():               # [19] USD mislabeled 'shares'
    a = core.answer("What was total equity in 2025?", entity=ent("TSLA"))
    assert a.trace.citations and a.trace.citations[0]["unit"] == "USD"


def test_no_share_count_misscaled():               # [20] impossible share counts
    d = json.loads((ROOT / "store" / "facts.json").read_text())
    bad = [f for f in d
           if f.get("unit") == "shares" and f.get("scale") == "millions"
           and len("".join(c for c in str(f.get("raw_cell_text", "")).split(".")[0]
                           if c.isdigit())) >= 8]
    assert not bad, f"{len(bad)} share counts still mis-scaled as millions"
