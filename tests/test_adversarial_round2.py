"""Regressions from the 4-lens adversarial audit of the redesigned router.
Offline + deterministic, against the real banked data."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core      # noqa: E402
import edgar     # noqa: E402
import llm       # noqa: E402


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)


def ent(t):
    return edgar.resolve_entity(t)[0][0] if t else None


# --- never a look-alike: the consolidated figure must not stand in ---------- #
@pytest.mark.parametrize("q,tk,forbidden", [
    ("What was Tesla's energy revenue in 2025?", "TSLA", "94,827"),
    ("How much revenue did Tesla make from energy generation and storage in 2025?", "TSLA", "94,827"),
    ("What was Tesla's revenue for the automotive business in 2025?", "TSLA", "94,827"),
    ("What was Tesla's revenue in Q2 2026 from automotive?", "TSLA", "28,236"),
    ("What was Tesla's revenue minus automotive in 2025?", "TSLA", "94,827"),
    ("Revenue in the US for Tesla 2025", "TSLA", "94,827"),
    ("What were Tesla's American revenues in 2025?", "TSLA", "94,827"),
    ("What were Vertiv's Americas sales in 2025?", "VRT", "10,229"),
    ("What was SoFi's technology revenue in 2025?", "SOFI", "3,613"),
    ("What was Tesla's FSD revenue in 2025?", "TSLA", "94,827"),
    ("What was Tesla's gross profit from services in 2025?", "TSLA", "17,094"),
    ("What was Tesla's consolidated net income in 2025?", "TSLA", "3,794"),
    ("What was Disney's net income including noncontrolling interests in 2025?", "DIS", "12,404"),
    ("Tesla's earnings before interest and taxes 2025", "TSLA", "3,794"),
    ("Tesla's net income excluding the SpaceX gain in 2025", "TSLA", "3,794"),
    ("What was Tesla's owner earnings in 2025?", "TSLA", "3,794"),
    ("Tesla's revenue per employee 2025", "TSLA", "94,827"),
    ("What was Tesla's net change in cash in 2025?", "TSLA", "16,513"),
    ("What were Tesla's cash equivalents and investments at end of 2025?", "TSLA", "16,513"),
    ("What was Tesla's total cash flow in 2025?", "TSLA", "14,747"),
    ("What was Netflix net income 2024?", "TSLA", "7,091"),
    ("What was Delta revenue in 2024?", "AAL", "54,"),
    ("What was Apple's revenue in 2024?", "TSLA", "97,690"),
    ("crwv d&a latest quarter as a share of latest fiscal year", "CRWV", "2,454"),
])
def test_no_lookalike_substitution(q, tk, forbidden):
    a = core.answer(q, entity=ent(tk))
    assert a.status != "answered" and forbidden not in a.text, a.text


# --- periods: the period asked, in its own kind ------------------------------ #
@pytest.mark.parametrize("q,tk,needle", [
    ("Revenue in H1 2025", "TSLA", "41,831"),
    ("What was revenue for the 6 months ended June 30, 2025?", "TSLA", "41,831"),
    ("Revenue for the three months ended June 30, 2025", "TSLA", "22,496"),
    ("Revenue for the 9 months ended June 2025", "DIS", "71,961"),
    ("Revenue 3Q26", "DIS", "25,248"),
    ("Revenue Q2-25", "TSLA", "22,496"),
    ("Revenue Fiscal Year '24", "TSLA", "97,690"),
    ("Revenue for the six-month period ended June 30, 2026", "AAL", "30,647"),
    ("Revenue in the half year ended June 30, 2026", "TSLA", "50,623"),
    ("Revenue in calendar 2025", "TSLA", "94,827"),       # Dec filer: same period
    ("H1 2026 revenue vs H1 2025 revenue", "TSLA", "+21.0%"),
])
def test_period_phrasings(q, tk, needle):
    a = core.answer(q, entity=ent(tk))
    assert a.status == "answered" and needle in a.text, a.text


@pytest.mark.parametrize("q,tk", [
    ("Revenue for the quarter ended March 2026", "DIS"),      # fiscal Q2: not banked
    ("Revenue in the second fiscal quarter of 2026", "DIS"),
])
def test_quarter_never_substituted(q, tk):
    a = core.answer(q, entity=ent(tk))
    assert "25,248" not in a.text and a.status != "answered", a.text


def test_balance_sheet_instants_compare_across_kinds():
    a = core.answer("Cash and cash equivalents at the end of 2025 vs Q2 2026", entity=ent("TSLA"))
    assert a.status == "answered" and "16,513" in a.text and "15,219" in a.text


def test_margin_comparison_is_margins_not_dollar_growth():
    a = core.answer("Gross margin 2024 vs 2025", entity=ent("TSLA"))
    assert a.status == "answered" and "17.9%" in a.text and "18.0%" in a.text
    assert "percentage points" in a.text


def test_negative_base_has_no_percent():
    a = core.answer("Operating income Q2 2026 vs Q2 2025", entity=ent("BE"))
    assert "5302" not in a.text and "negative base" in a.text


def test_last_n_years():
    a = core.answer("Revenue for the last three years", entity=ent("TSLA"))
    assert all(y in a.text for y in ("FY2023", "FY2024", "FY2025"))


def test_pct_of_revenue_computed():
    a = core.answer("Broadcom R&D as a percent of revenue, FY2025", entity=ent("AVGO"))
    assert a.status == "answered" and "17.2%" in a.text


# --- metric labels across filers --------------------------------------------- #
@pytest.mark.parametrize("q,tk,needle", [
    ("What was diluted EPS in 2025?", "AVGO", "4.77"),
    ("Diluted net loss per share 2025", "CRWV", "2.81"),
    ("Operating cash flow 2024", "VRT", "1,319.3"),
    ("Capital expenditures 2025", "ISRG", "539.8"),
    ("Research and development 2025", "ASTS", "28,115"),
    ("Total equity at end of 2025", "SOFI", "10,489,495"),
    ("What was Broadcom's gross profit in fiscal 2025?", "AVGO", "43,294"),
    ("What is Nvidia's EPS in fiscal 2025, diluted?", None, "2.94"),
    ("What was Tesla's net income per diluted share in Q2 2026?", "TSLA", "0.32"),
])
def test_supported_metric_answers(q, tk, needle):
    a = core.answer(q, entity=ent(tk))
    assert a.status == "answered" and needle in a.text, a.text


# --- company resolution + lanes ---------------------------------------------- #
@pytest.mark.parametrize("q,tk,lane", [
    ("What is ASTS?", None, "overview"),
    ("What does Tesla do?", None, "overview"),
    ("What do you know about Hershey?", None, "overview"),
    ("Hey there! Who are you?", None, "capability"),
    ("goodbye", None, "smalltalk"),
    ("What did Tesla report as revenue in 2024 vs Rivian?", "TSLA", "multi_company"),
    ("Tesla vs GM revenue 2024", None, "multi_company"),
])
def test_lanes_round2(q, tk, lane):
    assert core.route(q, entity=ent(tk)) == lane


def test_typo_banked_name_resolves():
    a = core.answer("What was Telsa revenue in 2024?")
    assert a.status == "answered" and "97,690" in a.text


def test_chat_acronym_is_not_a_company():
    a = core.answer("IMO what was net income in 2024?", entity=ent("TSLA"))
    assert "IMPERIAL" not in a.text.upper()


def test_bloomberg_is_a_source_not_a_rival():
    a = core.answer("Compare Tesla revenue 2024 versus 2023 per Bloomberg")
    assert "compares Tesla, Inc. with Bloomberg" not in a.text


# --- multi-year growth leads with the whole period; follow-ups keep context -- #
def test_multi_year_growth_leads_with_whole_period():
    a = core.answer("what is the total revenue growth from 2023 to 2025 period ?", entity=ent("HSY"))
    assert a.status == "answered" and "+4.7% over the whole period" in a.text
    assert "compound annual growth rate" in a.text and a.headline[1] == "+4.7%"


def test_followup_entire_period_inherits_metric_and_years():
    e = ent("HSY")
    a = core.answer("what is the total revenue growth from 2023 to 2025 period ?", entity=e)
    b = core.answer("This is YoY right ? What is the growth rate for the entire period ?",
                    entity=e, context=a.query)
    assert b.status == "answered" and "+4.7%" in b.text and b.text.startswith("Yes")
    assert not b.passages                                  # not a prose dead-end


def test_followup_new_metric_same_periods():
    e = ent("HSY")
    a = core.answer("what is the total revenue growth from 2023 to 2025 period ?", entity=e)
    c = core.answer("what about net income?", entity=e, context=a.query)
    assert "FY2023" in c.text and "FY2025" in c.text and "Net income" in c.text


def test_context_does_not_hijack_other_lanes():
    e = ent("HSY")
    a = core.answer("what is the total revenue growth from 2023 to 2025 period ?", entity=e)
    assert core.answer("what are the main risks?", entity=e, context=a.query).passages
    assert "11,692,576" not in core.answer("thanks!", entity=e, context=a.query).text


# --- conversation frame: follow-ups fill empty slots, not trigger words ------ #
def _dialogue(tk, turns):
    e, ctx, out = ent(tk), None, []
    for t in turns:
        a = core.answer(t, entity=e, context=ctx)
        if a.query is not None and a.query.metrics and a.status != "refused":
            ctx = a.query
        out.append(a)
    return out


def test_frame_followup_each_year_growth():
    a, b = _dialogue("KTOS", ["What's YoY revenue growth from 2023-2025 ?",
                              "Okay for each year growth ?"])
    assert a.text.split("year over year:")[1].strip().startswith("FY2023→FY2024 +9.6%")
    assert b.status == "answered" and "+9.6%" in b.text and "+18.5%" in b.text
    assert b.text.startswith("Following up on your last question (Total revenue, FY2023–FY2025)")


def test_frame_new_metric_keeps_periods_and_loss_base_is_honest():
    _a, b = _dialogue("KTOS", ["What's YoY revenue growth from 2023-2025 ?",
                               "what about net income?"])
    assert "FY2023" in b.text and "FY2025" in b.text and "isn't meaningful from a loss" in b.text


def test_frame_needs_a_factual_signal():
    _a, b, c = _dialogue("KTOS", ["What's YoY revenue growth from 2023-2025 ?",
                                  "hmm interesting", "thanks!"])
    assert "1,346.8" not in b.text and "1,346.8" not in c.text
    assert c.text.startswith("You're welcome")
