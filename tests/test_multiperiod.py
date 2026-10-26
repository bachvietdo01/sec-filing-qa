"""Multi-period decomposition: a multi-year request fans out over a single-period
atom and composes one answer. Pins the margin-series gap (margin used to answer
only the first year) and year-range expansion ("2023-2025" -> all three)."""
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


def ent(ticker):
    return edgar.resolve_entity(ticker)[0][0]


def test_year_range_expands():
    assert core._expand_year_range("revenue from 2023 to 2025", [2023, 2025]) == [2023, 2024, 2025]
    assert core._expand_year_range("margin for each year 2023-2025", [2023, 2025]) == [2023, 2024, 2025]
    # NOT a range: two explicitly listed years stay as-is
    assert core._expand_year_range("revenue in 2023 and 2025", [2023, 2025]) == [2023, 2025]
    # a single year is untouched
    assert core._expand_year_range("revenue in 2024", [2024]) == [2024]


def test_multi_year_margin_lists_each_year():
    a = core.answer("operating margin for each year from 2023-2025", entity=ent("GOOG"))
    assert a.status == "answered"
    # every year present and distinct, not just the first
    assert "2023" in a.text and "2024" in a.text and "2025" in a.text
    assert "27.4%" in a.text and "32.0%" in a.text
    # three operating-income + three revenue citations
    assert len(a.trace.citations) == 6


def test_multi_year_level_series():
    a = core.answer("revenue from 2023 to 2025", entity=ent("GOOG"))
    assert a.status == "answered"
    assert "307,394" in a.text and "350,018" in a.text and "402,836" in a.text


def test_single_year_margin_keeps_detail():
    a = core.answer("operating margin in 2024", entity=ent("GOOG"))
    assert a.status == "answered" and "32.1%" in a.text and "/" in a.text  # shows the fraction
