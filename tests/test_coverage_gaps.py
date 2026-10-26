"""Targeted tests closing the gaps the verify/critic pass flagged.

Focus: negative/branch coverage of trust invariants that the happy-path suite
left implicit — especially the GROUNDING-FAILURE path, which is the single most
important untested invariant in the system.
"""
from __future__ import annotations

import json

import pytest

import config
import edgar
import extract
import bank


# --------------------------------------------------------------------------- #
# extract._is_grounded — the trust gate's negative branch
# --------------------------------------------------------------------------- #
def test_is_grounded_true_when_token_verbatim_on_page():
    page = "Total revenues 94,827 97,690 96,773"
    assert extract._is_grounded("94,827", page, is_dash=False) is True


def test_is_grounded_false_when_token_absent():
    # A number the parser produced that is NOT verbatim on the page must fail
    # the gate — this is what stops a hallucinated/mis-read figure.
    page = "Total revenues 94,827 97,690 96,773"
    assert extract._is_grounded("12,345", page, is_dash=False) is False


def test_is_grounded_dash_is_nil_and_needs_no_lookup():
    assert extract._is_grounded("—", "", is_dash=True) is True


def test_ungrounded_fact_is_rejected_by_the_bank(mkfact, tmp_path):
    # End-to-end of the invariant: an ungrounded fact can never be banked.
    bad = mkfact(grounding_ok=False, confidence=0.0, raw_cell_text="99,999")
    with pytest.raises(ValueError):
        bank.write_facts([bad], path=tmp_path / "facts.json")


# --------------------------------------------------------------------------- #
# edgar.resolve_entity — the ambiguity branches (duplicate ticker/title, cap)
# --------------------------------------------------------------------------- #
def _patch_rows(monkeypatch, rows):
    monkeypatch.setattr(edgar, "_load_tickers", lambda: rows)


def test_resolve_entity_duplicate_ticker_is_ambiguous(monkeypatch):
    _patch_rows(monkeypatch, [
        {"cik_str": 1, "ticker": "DUP", "title": "Dup One Inc."},
        {"cik_str": 2, "ticker": "DUP", "title": "Dup Two Inc."},
    ])
    cands, status = edgar.resolve_entity("DUP")
    assert status == "ambiguous" and len(cands) == 2


def test_resolve_entity_duplicate_title_is_ambiguous(monkeypatch):
    _patch_rows(monkeypatch, [
        {"cik_str": 1, "ticker": "AAA", "title": "Same Name Inc."},
        {"cik_str": 2, "ticker": "BBB", "title": "Same Name Inc."},
    ])
    cands, status = edgar.resolve_entity("Same Name Inc.")
    assert status == "ambiguous" and len(cands) == 2


def test_resolve_entity_substring_match_is_capped_at_ten(monkeypatch):
    rows = [{"cik_str": i, "ticker": f"BK{i}", "title": f"Bank {i} Corp"}
            for i in range(12)]
    _patch_rows(monkeypatch, rows)
    cands, status = edgar.resolve_entity("bank")
    assert status == "ambiguous" and len(cands) == 10  # contains[:10]


# --------------------------------------------------------------------------- #
# edgar._load_tickers — the real download + cache + parse path
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def raise_for_status(self):
        pass


def test_load_tickers_downloads_once_then_reads_cache(monkeypatch, tmp_path):
    payload = json.dumps({
        "0": {"cik_str": 1318605, "ticker": "TSLA", "title": "Tesla, Inc."},
        "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    })
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    n = {"count": 0}

    def fake_get(url, headers=None, timeout=None):
        n["count"] += 1
        assert "User-Agent" in headers           # polite UA sent
        return _FakeResp(payload)

    monkeypatch.setattr(edgar.requests, "get", fake_get)

    rows1 = edgar._load_tickers()                # downloads + caches
    rows2 = edgar._load_tickers()                # reads cache, no new request
    assert n["count"] == 1
    assert rows1 == rows2
    assert {r["ticker"] for r in rows1} == {"TSLA", "AAPL"}
    assert (tmp_path / "company_tickers.json").exists()


# --------------------------------------------------------------------------- #
# edgar.discover_filings — empty 'recent' returns []
# --------------------------------------------------------------------------- #
def test_discover_filings_empty_recent(monkeypatch):
    class R:
        def raise_for_status(self): pass
        def json(self):
            return {"filings": {"recent": {
                "form": [], "accessionNumber": [], "filingDate": [],
                "reportDate": [], "primaryDocument": [], "primaryDocDescription": []}}}

    monkeypatch.setattr(edgar.time, "sleep", lambda *_: None)
    monkeypatch.setattr(edgar.requests, "get", lambda *a, **k: R())
    assert edgar.discover_filings(1318605) == []


# --------------------------------------------------------------------------- #
# bank.prose_for — prefix filtering of prose chunks
# --------------------------------------------------------------------------- #
def test_prose_for_filters_by_stem_prefix(monkeypatch):
    chunks = [
        {"filing_stem": "TSLA_10K_2025-12-31", "section": "risk", "page_no": 10,
         "text": "a"},
        {"filing_stem": "AAPL_10K_2024-09-28", "section": "risk", "page_no": 5,
         "text": "b"},
    ]
    monkeypatch.setattr(bank, "load_prose", lambda path=None: chunks)
    got = bank.prose_for("TSLA")
    assert len(got) == 1 and got[0]["filing_stem"].startswith("TSLA")
