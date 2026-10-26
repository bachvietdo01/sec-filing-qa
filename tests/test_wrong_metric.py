"""Silent-wrong-metric guard: a metric we don't bank must REFUSE, never be
answered with a look-alike. These fired on banked companies (GOOG), so they are
shipped-engine bugs, not just new-filer generalization. From the 20-company
generalization stress audit."""
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


@pytest.mark.parametrize("q", [
    "What was GOOG cost of revenue in 2024?",      # was -> Total revenue
    "What was GOOG dividend per share in 2024?",   # was -> Diluted EPS
    "What were GOOG retained earnings in 2024?",   # was -> Net income
    "What was GOOG book value per share in 2024?", # was -> EPS + equity
    "What was GOOG revenue per share in 2024?",    # was -> revenue
])
def test_unsupported_metric_refuses_not_substitutes(q):
    a = core.answer(q, entity=ent("GOOG"))
    assert a.status == "refused"
    assert "don't track" in a.text or "won't substitute" in a.text


@pytest.mark.parametrize("q,needle", [
    ("What was GOOG diluted EPS in 2024?", "EPS"),
    ("What was GOOG net income in 2024?", "Net income"),
    ("What was GOOG book value in 2024?", "equity"),  # "book value" == total equity
])
def test_legit_metrics_still_answer(q, needle):
    a = core.answer(q, entity=ent("GOOG"))
    assert a.status == "answered" and needle.lower() in a.text.lower()
