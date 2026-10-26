"""Unit tests for edgar.py — deterministic entity resolution + throttled EDGAR
access. All network and sleeps are monkeypatched; nothing leaves the process."""
from __future__ import annotations

import pytest

import config
import edgar


# --------------------------------------------------------------------------- #
# resolve_entity — the trust invariant: a CIK is matched, never guessed.
# --------------------------------------------------------------------------- #
def test_resolve_exact_ticker(fake_tickers):
    cands, status = edgar.resolve_entity("TSLA")
    assert status == "exact"
    assert len(cands) == 1
    assert cands[0].cik == 1318605
    assert cands[0].ticker == "TSLA"


def test_resolve_ticker_case_insensitive(fake_tickers):
    cands, status = edgar.resolve_entity("tsla")
    assert status == "exact"
    assert cands[0].cik == 1318605


def test_resolve_single_title_contains(fake_tickers):
    # "Tesla" is not a ticker and not an exact title, matches one title substring.
    cands, status = edgar.resolve_entity("Tesla")
    assert status == "exact"
    assert cands[0].ticker == "TSLA"
    assert cands[0].title == "Tesla, Inc."


def test_resolve_exact_title(fake_tickers):
    cands, status = edgar.resolve_entity("Apple Inc.")
    assert status == "exact"
    assert cands[0].cik == 320193
    assert cands[0].ticker == "AAPL"


def test_resolve_exact_title_case_insensitive(fake_tickers):
    cands, status = edgar.resolve_entity("APPLE INC.")
    assert status == "exact"
    assert cands[0].ticker == "AAPL"


def test_resolve_ambiguous_substring(fake_tickers):
    # "Tech" is a substring of two distinct titles -> caller must disambiguate.
    cands, status = edgar.resolve_entity("Tech")
    assert status == "ambiguous"
    assert len(cands) == 2
    assert {c.ticker for c in cands} == {"TECA", "TECB"}


def test_resolve_not_found_returns_no_candidates(fake_tickers):
    cands, status = edgar.resolve_entity("Nonexistent Co")
    assert status == "not_found"
    assert cands == []  # invariant: never fabricate a CIK


def test_resolve_cik_is_int(fake_tickers):
    cands, _ = edgar.resolve_entity("TSLA")
    assert isinstance(cands[0].cik, int)


# --------------------------------------------------------------------------- #
# document_url — dash-stripping + correct interpolation
# --------------------------------------------------------------------------- #
def test_document_url_strips_dashes_and_interpolates():
    url = edgar.document_url(1318605, "0001628280-25-000000", "tsla-10k.htm")
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/"
        "1318605/000162828025000000/tsla-10k.htm"
    )
    assert "-25-" not in url.split("/")[-2]  # accession segment has no dashes


# --------------------------------------------------------------------------- #
# discover_filings — one throttled submissions lookup, forms filtered
# --------------------------------------------------------------------------- #
def _submissions_payload():
    return {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K", "10-Q", "10-K"],
                "accessionNumber": ["acc0", "acc1", "acc2", "acc3"],
                "filingDate": ["2025-01-01", "2025-02-01",
                               "2024-07-01", "2024-01-01"],
                "reportDate": ["2024-12-31", "2025-01-15",
                               "2024-06-30", "2023-12-31"],
                "primaryDocument": ["d0.htm", "d1.htm", "d2.htm", "d3.htm"],
                "primaryDocDescription": ["10-K", "8-K", "10-Q", "10-K"],
            }
        }
    }


class _FakeResp:
    def __init__(self, json_data=None, content=None):
        self._json = json_data
        self.content = content
        self.raised = False

    def raise_for_status(self):
        self.raised = True

    def json(self):
        return self._json


def _patch_get(monkeypatch, resp, rec):
    def fake_get(url, headers=None, timeout=None):
        rec["url"] = url
        rec["headers"] = headers
        rec["timeout"] = timeout
        return resp
    monkeypatch.setattr(edgar.requests, "get", fake_get)
    sleeps = []
    monkeypatch.setattr(edgar.time, "sleep", lambda s: sleeps.append(s))
    rec["sleeps"] = sleeps


def test_discover_filings_filters_forms(monkeypatch):
    rec = {}
    resp = _FakeResp(json_data=_submissions_payload())
    _patch_get(monkeypatch, resp, rec)

    out = edgar.discover_filings(1318605)
    # 8-K filtered out; 10-K/10-Q kept, original order preserved
    assert [f["form"] for f in out] == ["10-K", "10-Q", "10-K"]


def test_discover_filings_maps_fields(monkeypatch):
    rec = {}
    resp = _FakeResp(json_data=_submissions_payload())
    _patch_get(monkeypatch, resp, rec)

    out = edgar.discover_filings(1318605)
    first = out[0]
    assert first == {
        "form": "10-K",
        "accession_no": "acc0",
        "filing_date": "2025-01-01",
        "report_date": "2024-12-31",
        "primary_document": "d0.htm",
        "primary_doc_desc": "10-K",
    }


def test_discover_filings_honors_limit(monkeypatch):
    rec = {}
    resp = _FakeResp(json_data=_submissions_payload())
    _patch_get(monkeypatch, resp, rec)

    out = edgar.discover_filings(1318605, limit=2)
    assert len(out) == 2


def test_discover_filings_only_requested_forms(monkeypatch):
    rec = {}
    resp = _FakeResp(json_data=_submissions_payload())
    _patch_get(monkeypatch, resp, rec)

    out = edgar.discover_filings(1318605, forms=("10-Q",))
    assert [f["form"] for f in out] == ["10-Q"]


def test_discover_filings_url_ua_and_throttle(monkeypatch):
    rec = {}
    resp = _FakeResp(json_data=_submissions_payload())
    _patch_get(monkeypatch, resp, rec)

    edgar.discover_filings(1318605)
    assert rec["url"] == config.EDGAR_SUBMISSIONS_URL.format(cik10="0001318605")
    assert rec["headers"]["User-Agent"] == config.EDGAR_UA
    assert resp.raised is True            # raise_for_status() was called
    assert rec["sleeps"] == [config.EDGAR_MIN_INTERVAL_S]   # throttled once


def test_discover_filings_cik_zero_padded(monkeypatch):
    rec = {}
    resp = _FakeResp(json_data=_submissions_payload())
    _patch_get(monkeypatch, resp, rec)

    edgar.discover_filings(111)
    assert "CIK0000000111.json" in rec["url"]


# --------------------------------------------------------------------------- #
# fetch_document — bytes back, correct URL + UA, throttled
# --------------------------------------------------------------------------- #
def test_fetch_document_returns_bytes(monkeypatch):
    rec = {}
    resp = _FakeResp(content=b"<html>raw filing</html>")
    _patch_get(monkeypatch, resp, rec)

    data = edgar.fetch_document(1318605, "0001628280-25-000000", "tsla.htm")
    assert data == b"<html>raw filing</html>"
    assert resp.raised is True


def test_fetch_document_url_and_ua(monkeypatch):
    rec = {}
    resp = _FakeResp(content=b"x")
    _patch_get(monkeypatch, resp, rec)

    edgar.fetch_document(1318605, "0001628280-25-000000", "tsla.htm")
    assert rec["url"] == edgar.document_url(
        1318605, "0001628280-25-000000", "tsla.htm"
    )
    assert rec["headers"]["User-Agent"] == config.EDGAR_UA
    assert rec["sleeps"] == [config.EDGAR_MIN_INTERVAL_S]
