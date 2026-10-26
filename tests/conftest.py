"""Shared test harness and fixtures for the d71 take-home unit tests.

Design goals for the suite:
  * hermetic — never touch the real store/ or the network; the Bank is
    redirected to an in-memory/temp fixture, EDGAR and the LLM are monkeypatched.
  * trust-focused — exercise the invariants that matter (grounding, label
    disambiguation, scale/period handling, honest refusal), not just happy paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# make the project modules importable (tests/ is one level below the root)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CIK = 1318605          # Tesla's real CIK; arbitrary but consistent across tests
TSLA_PDF = ROOT / "filings" / "TSLA_10K_2025-12-31.pdf"


# --------------------------------------------------------------------------- #
# Fact factory + a small, realistic sample Fact Store
# --------------------------------------------------------------------------- #
def _base_fact(**over) -> dict:
    """Build one Fact dict with all required fields; override via kwargs."""
    f = dict(
        company="Tesla, Inc.", cik=CIK, ticker="TSLA",
        filing_stem="TSLA_10K_2025-12-31", form="10-K",
        accession_no="0001628280-25-000000",
        statement="operations", raw_label="Total revenues",
        period="FY2025", period_type="duration", period_kind="annual",
        period_end="2025-12-31", fiscal_year=2025, fiscal_quarter=None,
        months=12, value=94_827.0, unit="USD", scale="millions",
        is_negative=False, is_dash=False, gaap_flag=True,
        page_no=82, raw_cell_text="94,827", extraction_method="pdfplumber_text",
        grounding_ok=True, confidence=0.98, section="Revenues",
        source_line="Total revenues 94,827 97,690 96,773",
    )
    f.update(over)
    return f


@pytest.fixture
def mkfact():
    """Expose the fact factory to tests that need bespoke rows."""
    return _base_fact


@pytest.fixture
def sample_facts() -> list[dict]:
    """A compact Fact Store covering the disambiguation-critical cases:
      * revenue (two years, for growth)
      * net income attributable to common stockholders  AND  total 'Net income'
        (the GAAP subtotal look-alike)
      * operating income
      * diluted/basic EPS in $/share  AND  a 'Diluted' share-count row in
        millions (the reused-label collision)
      * cash on the balance sheet (two as-of dates)
    """
    F = _base_fact
    return [
        F(raw_label="Total revenues", period="FY2025", fiscal_year=2025,
          value=94_827.0, raw_cell_text="94,827"),
        F(raw_label="Total revenues", period="FY2024", fiscal_year=2024,
          value=97_690.0, raw_cell_text="97,690"),
        F(raw_label="Income from operations", period="FY2025", fiscal_year=2025,
          value=4_355.0, raw_cell_text="4,355", section="Operating expenses"),
        F(raw_label="Net income attributable to common stockholders",
          period="FY2025", fiscal_year=2025, value=3_794.0,
          raw_cell_text="3,794", section="Net income"),
        F(raw_label="Net income attributable to common stockholders",
          period="FY2024", fiscal_year=2024, value=7_091.0,
          raw_cell_text="7,091", section="Net income"),
        F(raw_label="Net income", period="FY2025", fiscal_year=2025,
          value=3_855.0, raw_cell_text="3,855", section="Net income"),
        F(raw_label="Diluted", period="FY2025", fiscal_year=2025, value=1.08,
          unit="USD", scale="units", raw_cell_text="1.08",
          section="Net income per share of common stock attributable to "
                  "common stockholders"),
        F(raw_label="Basic", period="FY2025", fiscal_year=2025, value=1.18,
          unit="USD", scale="units", raw_cell_text="1.18",
          section="Net income per share of common stock attributable to "
                  "common stockholders"),
        F(raw_label="Diluted", period="FY2025", fiscal_year=2025, value=3_528.0,
          unit="shares", scale="millions", raw_cell_text="3,528",
          section="Weighted average shares used in computing net income per "
                  "share of common stock"),
        F(statement="balance_sheet", raw_label="Cash and cash equivalents",
          period="ASOF2025-12-31", period_type="instant", period_kind="instant",
          period_end="2025-12-31", fiscal_year=2025, months=None, value=16_513.0,
          raw_cell_text="16,513", page_no=80, section="Current assets"),
        F(statement="balance_sheet", raw_label="Cash and cash equivalents",
          period="ASOF2024-12-31", period_type="instant", period_kind="instant",
          period_end="2024-12-31", fiscal_year=2024, months=None, value=16_139.0,
          raw_cell_text="16,139", page_no=80, section="Current assets"),
    ]


# --------------------------------------------------------------------------- #
# Entity + monkeypatch fixtures (keep tests hermetic)
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_entity():
    import edgar
    return edgar.Entity(cik=CIK, ticker="TSLA", title="Tesla, Inc.")


@pytest.fixture
def banked(monkeypatch, sample_facts):
    """Redirect the Bank's reads to sample_facts (no disk, no real store/)."""
    import bank
    monkeypatch.setattr(bank, "load_facts", lambda path=None: sample_facts)
    return sample_facts


@pytest.fixture
def fake_resolve(monkeypatch, fake_entity):
    """Make core's entity resolution deterministic and offline.

    'Tesla'/'TSLA' -> exact fake_entity; '' -> not_found; 'AMBIG' -> ambiguous.
    Tests can re-patch edgar.resolve_entity for other scenarios.
    """
    import edgar

    def _resolve(mention: str):
        m = (mention or "").strip().lower()
        if m in ("tesla", "tsla", "tesla, inc.", "tesla inc"):
            return [fake_entity], "exact"
        if m == "ambig":
            return [fake_entity, edgar.Entity(2, "AMB", "Ambi Corp")], "ambiguous"
        return [], "not_found"

    monkeypatch.setattr(edgar, "resolve_entity", _resolve)
    return _resolve


@pytest.fixture
def no_llm(monkeypatch):
    """Force the offline path: llm.available() -> False."""
    import llm
    monkeypatch.setattr(llm, "available", lambda: False)


@pytest.fixture
def tsla_pdf():
    """The real rendered 10-K, for integration-style extraction tests."""
    if not TSLA_PDF.exists():
        pytest.skip(f"rendered PDF not present: {TSLA_PDF}")
    return TSLA_PDF


@pytest.fixture
def fake_tickers(monkeypatch):
    """Patch edgar._load_tickers with a small fixed universe that includes an
    ambiguous substring ('Tech' matches two) and a clean exact ticker."""
    import edgar
    rows = [
        {"cik_str": CIK, "ticker": "TSLA", "title": "Tesla, Inc."},
        {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        {"cik_str": 111, "ticker": "TECA", "title": "Tech Alpha Corp"},
        {"cik_str": 222, "ticker": "TECB", "title": "Tech Beta Corp"},
    ]
    monkeypatch.setattr(edgar, "_load_tickers", lambda: rows)
    return rows
