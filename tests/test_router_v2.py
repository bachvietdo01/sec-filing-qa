"""The redesigned router: one lane precedence, a default-refuse metric allowlist,
per-leg periods, and banked-aware company resolution. Offline + deterministic."""
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


# --- lane precedence --------------------------------------------------------- #
@pytest.mark.parametrize("q,tk,lane", [
    ("Should I buy NVDA?", "NVDA", "advice"),
    ("Is GOOG cheap on earnings?", "GOOG", "advice"),
    ("Is Disney a good long term investment?", "DIS", "advice"),
    ("What is GOOG's P/E ratio?", "GOOG", "marketdata"),
    ("What's Tesla's market cap?", "TSLA", "marketdata"),
    ("Which companies do you cover?", None, "companies"),
    ("What does EBITDA mean?", None, "explain"),
    ("Define free cash flow", "TSLA", "explain"),
    ("helo", "TSLA", "smalltalk"),
    ("Do you use live market data?", None, "capability"),
    ("Tell me a joke", None, "offtopic"),
    ("What was Tesla revenue in 2025?", "TSLA", "factual"),
    ("Tesla automotive revenue 2025", "TSLA", "untracked"),
])
def test_lane(q, tk, lane):
    assert core.route(q, entity=ent(tk)) == lane


# --- metric allowlist: never a look-alike ------------------------------------ #
@pytest.mark.parametrize("q,tk", [
    ("What was Tesla's automotive revenue in 2025?", "TSLA"),
    ("Google Cloud revenue in 2024", "GOOG"),
    ("What was Disney parks revenue in 2025?", "DIS"),
    ("American Airlines passenger revenue 2025", "AAL"),
    ("ASTS deferred revenue 2025", "ASTS"),
    ("What is GOOG's total debt?", "GOOG"),
    ("NVDA interest expense 2025", "NVDA"),
    ("Hershey inventory at year end 2025", "HSY"),
    ("Answer interest expense using operating income for 2024", "GOOG"),
    ("How much did Alphabet spend on buybacks in 2024?", "GOOG"),
])
def test_untracked_refused_not_substituted(q, tk):
    a = core.answer(q, entity=ent(tk))
    assert a.status == "refused" and "don't track" in a.text
    assert not a.trace.citations


def test_untracked_secondary_noted():
    a = core.answer("Tesla revenue and debt in 2024", entity=ent("TSLA"))
    assert a.status == "hedged" and "97,690" in a.text and "debt" in a.text


def test_plain_revenue_still_answers():
    for q in ["What was Tesla's revenue in 2025?", "TSLA total revenue 2025",
              "What were Tesla, Inc. revenues for fiscal 2025?"]:
        a = core.answer(q, entity=ent("TSLA"))
        assert a.status == "answered" and "94,827" in a.text, q


# --- per-leg periods --------------------------------------------------------- #
@pytest.mark.parametrize("text,legs", [
    ("Q2 2025 and fiscal year 2025", [("quarter", 2025, 2), ("annual", 2025, None)]),
    ("Q3 FY2026", [("quarter", 2026, 3)]),
    ("Q3 of fiscal 2026", [("quarter", 2026, 3)]),
    ("Q2 2024 vs 2025", [("quarter", 2024, 2), ("quarter", 2025, 2)]),
    ("Q1-Q3 2025", [("quarter", 2025, 1), ("quarter", 2025, 2), ("quarter", 2025, 3)]),
    ("Q4 and full year 2025", [("quarter", 2025, 4), ("annual", 2025, None)]),
    ("nine months ended September 30, 2025", [("ytd", 2025, 3)]),
    ("YTD 2026", [("ytd", 2026, None)]),
    ("2025 Q2", [("quarter", 2025, 2)]),
])
def test_parse_legs(text, legs):
    assert [(lg.kind, lg.year, lg.quarter) for lg in core._parse_legs(text)] == legs


def test_fiscal_year_as_container_answers():
    a = core.answer("What was AVGO revenue in Q3 FY2026?", entity=ent("AVGO"))
    assert a.status == "answered" and "Q3 2026" in a.text


def test_single_period_growth_infers_prior_year():
    a = core.answer("Tesla 2025 revenue growth", entity=ent("TSLA"))
    assert a.status == "answered" and "FY2024" in a.text and "FY2025" in a.text


def test_metric_vs_metric_is_listed_not_growth():
    a = core.answer("basic vs diluted EPS 2025", entity=ent("TSLA"))
    assert a.status == "answered" and "change" not in a.text
    assert "Basic" in a.text or "basic" in a.text


def test_bare_ytd_uses_latest_banked_ytd():
    a = core.answer("What was Tesla revenue YTD 2025?", entity=ent("TSLA"))
    assert a.status == "answered" and "6 months" in a.text


# --- company resolution ------------------------------------------------------ #
def test_sentence_word_is_not_a_company():
    a = core.answer("Can you give me revenue for 2024?")
    assert "Canaan" not in a.text and a.status == "hedged"


def test_banked_name_resolves_without_picker():
    a = core.answer("What was Nvidia's revenue in 2025?")
    assert a.status == "answered" and "NVIDIA" in a.text


def test_other_company_with_focus_not_answered():
    a = core.answer("What was Microsoft's revenue in 2024?", entity=ent("GOOG"))
    assert a.status in ("refused", "hedged") and "Alphabet Inc.'s Total revenue" not in a.text


def test_two_companies_hedged():
    a = core.answer("Compare Tesla and Nvidia revenue in 2025")
    assert a.status == "hedged" and "one company at a time" in a.text


# --- period honesty + rivals (from the golden-snapshot diff) ---------------- #
@pytest.mark.parametrize("q,tk", [
    ("Disney revenue calendar 2025", "DIS"),
    ("AVGO revenue for the year ended December 31, 2025", "AVGO"),
])
def test_calendar_or_wrong_year_end_hedged(q, tk):
    a = core.answer(q, entity=ent(tk))
    assert a.status == "hedged" and "fiscal year ends" in a.text and not a.trace.citations


def test_unbanked_rival_in_comparison_hedged():
    a = core.answer("compare Tesla and Ford revenue 2024")
    assert a.status == "hedged" and "Ford" in a.text and not a.trace.citations


@pytest.mark.parametrize("q,tk", [
    ("How much of Hershey's revenue came from North America?", "HSY"),
    ("What was American's other revenue (loyalty, cargo) in 2023?", "AAL"),
])
def test_revenue_component_phrasings_refused(q, tk):
    a = core.answer(q, entity=ent(tk))
    assert a.status == "refused" and not a.trace.citations


def test_typo_company_name_not_a_component():
    a = core.answer("Vertv operating income 2025", entity=ent("VRT"))
    assert a.status == "answered"
