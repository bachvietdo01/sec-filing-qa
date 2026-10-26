"""Conversational-router regressions, from the 5-agent router stress audit.
Offline + deterministic (the router guards run on the raw question regardless of
the LLM path). Covers: no-company greetings must not resolve to an arbitrary
ticker; off-topic + a prose-cue word must steer back, not leak/dead-end; identity
questions; banked-only anchoring; and no $-figure leaking into a chat reply."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core      # noqa: E402
import edgar     # noqa: E402
import llm        # noqa: E402

_DOLLAR = re.compile(r"\$\s?\d")       # a specific dollar figure


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)


def ent(t):
    return edgar.resolve_entity(t)[0][0]


# --- no-company greetings must NOT anchor to a stray ticker / cold-refuse -------
@pytest.mark.parametrize("greeting", ["GM", "Morning", "Namaste", "Hola",
                                      "You're awesome!", "Evening!", "gm", "heya"])
def test_nocompany_greeting_is_warm_not_wrong_company(greeting):
    a = core.answer(greeting, entity=None)
    assert a.status in ("answered", "hedged")
    # must not name/claim an unbanked company it silently resolved
    for bad in ("Morningstar", "General Motors", "C3.ai", "Scholastic"):
        assert bad not in a.text
    assert not _DOLLAR.search(a.text)      # a greeting asserts no figure


def test_nocompany_capability_is_warm():
    for q in ("are you an AI?", "what data do you have?", "what can you do?"):
        a = core.answer(q, entity=None)
        assert a.status == "answered" and not _DOLLAR.search(a.text)


def test_named_unbanked_company_refused():
    a = core.answer("What was Microsoft's revenue in 2025?", entity=None)
    assert a.status == "refused" and "banked" in a.text.lower()


# --- off-topic + a prose-cue word must steer back, not leak or dead-end --------
@pytest.mark.parametrize("q", [
    "explain the litigation process in a divorce case",
    "what's the outlook for tonight's football game?",
    "what is your strategy for winning at chess?",
    "discuss the plot of your favorite book",
])
def test_offtopic_with_prose_word_steers_back(q):
    a = core.answer(q, entity=ent("NVDA"))
    # no raw risk-factor passage leaked, no cold "couldn't find a passage" dead-end
    assert "Item 1A" not in a.text and "couldn't find a passage" not in a.text
    assert not _DOLLAR.search(a.text)


# --- identity / overview, and a legit prose question still routes --------------
def test_identity_question_gives_overview():
    a = core.answer("what is ASTS?", entity=ent("ASTS"))
    assert a.status == "answered" and "AST SpaceMobile" in a.text


def test_what_is_company_doing_not_misleading_prose():
    a = core.answer("what's this company doing?", entity=ent("ASTS"))
    # the honest overview, not a risk-factor passage narrated as the business
    assert a.status == "answered" and "Item 1A" not in a.text


def test_legit_risk_question_routes_to_prose():
    a = core.answer("What are the biggest competition risks Tesla faces?", entity=ent("TSLA"))
    assert a.status in ("answered", "refused")   # prose lane (not a chat reply)
    assert "I answer questions about" not in a.text  # not the capability template
