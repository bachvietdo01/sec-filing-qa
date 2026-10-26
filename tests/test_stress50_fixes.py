"""Regressions for the A-list fixes from the 50-agent end-to-end stress audit.
Offline + deterministic, against the real banked data."""
from __future__ import annotations

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


def ent(t):
    return edgar.resolve_entity(t)[0][0]


def test_a1_margin_zero_revenue_no_crash():            # was ZeroDivisionError
    a = core.answer("What was ASTS net margin in 2023?", entity=ent("ASTS"))
    assert a.status == "refused" and "nan" not in a.text.lower()


def test_a9_dash_cell_not_zero():                      # dash shown as "$0"
    a = core.answer("What was ASTS revenue in 2023?", entity=ent("ASTS"))
    assert a.status in ("hedged", "refused")
    assert "$0" not in a.text and ("dash" in a.text.lower() or "—" in a.text)


@pytest.mark.parametrize("q", [
    "What were ISRG costs of revenue in 2025?",        # plural
    "What was ISRG cost-of-revenue in 2025?",          # hyphen
    "What were GOOG sales and marketing expenses in 2024?",
])
def test_a2_wrong_metric_refused(q):
    tk = "ISRG" if "ISRG" in q else "GOOG"
    a = core.answer(q, entity=ent(tk))
    assert a.status == "refused" and ("don't track" in a.text or "won't substitute" in a.text)


def test_a3_cagr_acronym_computes_rate():
    a = core.answer("What was Disney's revenue CAGR from 2023 to 2025?", entity=ent("DIS"))
    assert a.status == "answered" and "compound annual growth rate" in a.text.lower()


def test_a4_mixed_period_kinds_answered_per_leg():
    # each leg in its OWN kind — the quarter is never coerced to the year
    a = core.answer("What was Tesla revenue in Q2 2025 and fiscal year 2025?", entity=ent("TSLA"))
    assert a.status == "answered"
    assert "Q2 2025 $22,496" in a.text and "FY2025 $94,827" in a.text


def test_a4_growth_across_period_kinds_refused():
    a = core.answer("Tesla revenue growth Q2 2025 vs fiscal 2025", entity=ent("TSLA"))
    assert a.status == "refused" and "mix" in a.text.lower()


def test_a5_crwv_net_income_is_consolidated_loss():    # not the basic-EPS numerator
    a = core.answer("What was CoreWeave net income in 2025?", entity=ent("CRWV"))
    assert a.status == "answered" and "1,167" in a.text and "1,196" not in a.text


def test_a6_roe_negative_equity_hedged():
    a = core.answer("What was CoreWeave ROE in 2024?", entity=ent("CRWV"))
    assert a.status == "hedged" and ("deficit" in a.text.lower() or "negative" in a.text.lower())
    assert "226" not in a.text


def test_a7_implausible_eps_refused_not_wrong():
    a = core.answer("What was CRWV basic EPS in Q2 2025?", entity=ent("CRWV"))
    assert a.status == "refused" and "290" not in a.text


def test_a7_legit_eps_still_answers():                 # guard must not over-reject
    a = core.answer("What was ISRG diluted EPS in 2025?", entity=ent("ISRG"))
    assert a.status == "answered" and "7.87" in a.text


def test_a8_say_about_risk_not_overview():
    a = core.answer("What does the company say about supply chain risk?", entity=ent("HSY"))
    assert "Here's what I have on" not in a.text   # not the overview boilerplate


def test_minor_out_of_window_year_refused():
    a = core.answer("What was revenue in 1850?", entity=ent("GOOG"))
    assert a.status in ("refused", "hedged")


@pytest.mark.parametrize("q,needle", [
    ("What was GOOG revenue in 2024?", "350,018"),
    ("What was Tesla diluted EPS in 2025?", "1.08"),
])
def test_controls_still_work(q, needle):
    tk = "TSLA" if "Tesla" in q else "GOOG"
    a = core.answer(q, entity=ent(tk))
    assert a.status == "answered" and needle in a.text
