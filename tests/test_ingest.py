"""Unit tests for ingest.py — fetch primary doc + render to PDF. subprocess,
EDGAR fetch, and the renderer are all monkeypatched; writes go to tmp_path."""
from __future__ import annotations

import pytest

import config
import edgar
import ingest


@pytest.fixture
def filing():
    return {
        "form": "10-K",
        "accession_no": "0001628280-25-000000",
        "filing_date": "2025-01-27",
        "report_date": "2025-12-31",
        "primary_document": "tsla.htm",
        "primary_doc_desc": "10-K",
    }


# --------------------------------------------------------------------------- #
# ingest_filing
# --------------------------------------------------------------------------- #
def test_ingest_filing_fetches_and_renders(monkeypatch, tmp_path,
                                            fake_entity, filing):
    monkeypatch.setattr(config, "FILINGS_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)

    fetched = {}

    def fake_fetch(cik, acc, doc):
        fetched["args"] = (cik, acc, doc)
        return b"<html>raw tesla 10-K</html>"
    monkeypatch.setattr(edgar, "fetch_document", fake_fetch)

    rendered = {}

    def fake_render(htm_path, pdf_path, timeout=180):
        rendered["args"] = (htm_path, pdf_path)
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        return pdf_path
    monkeypatch.setattr(ingest, "render_html_to_pdf", fake_render)

    res = ingest.ingest_filing(fake_entity, filing)

    stem = "TSLA_10K_2025-12-31"
    assert res["stem"] == stem
    assert res["htm"] == tmp_path / f"{stem}.htm"
    assert res["pdf"] == tmp_path / f"{stem}.pdf"
    assert res["entity"] is fake_entity
    assert res["filing"] is filing

    # fetch was called with entity CIK + filing identifiers; htm written
    assert fetched["args"] == (fake_entity.cik, "0001628280-25-000000", "tsla.htm")
    assert res["htm"].read_bytes() == b"<html>raw tesla 10-K</html>"
    # renderer got (htm, pdf) and produced the pdf
    assert rendered["args"] == (res["htm"], res["pdf"])
    assert res["pdf"].exists()


def test_ingest_filing_idempotent(monkeypatch, tmp_path, fake_entity, filing):
    monkeypatch.setattr(config, "FILINGS_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)

    stem = "TSLA_10K_2025-12-31"
    (tmp_path / f"{stem}.htm").write_bytes(b"already here")
    (tmp_path / f"{stem}.pdf").write_bytes(b"%PDF already")

    def boom_fetch(*a, **k):
        raise AssertionError("fetch_document must not be called when files exist")
    monkeypatch.setattr(edgar, "fetch_document", boom_fetch)

    def boom_render(*a, **k):
        raise AssertionError("render must not be called when files exist")
    monkeypatch.setattr(ingest, "render_html_to_pdf", boom_render)

    res = ingest.ingest_filing(fake_entity, filing)
    assert res["stem"] == stem
    # untouched
    assert res["htm"].read_bytes() == b"already here"
    assert res["pdf"].read_bytes() == b"%PDF already"


def test_ingest_filing_renders_when_only_htm_present(monkeypatch, tmp_path,
                                                     fake_entity, filing):
    monkeypatch.setattr(config, "FILINGS_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)

    stem = "TSLA_10K_2025-12-31"
    (tmp_path / f"{stem}.htm").write_bytes(b"cached htm")

    def boom_fetch(*a, **k):
        raise AssertionError("fetch must be skipped when htm exists")
    monkeypatch.setattr(edgar, "fetch_document", boom_fetch)

    called = {"render": False}

    def fake_render(htm_path, pdf_path, timeout=180):
        called["render"] = True
        pdf_path.write_bytes(b"%PDF")
        return pdf_path
    monkeypatch.setattr(ingest, "render_html_to_pdf", fake_render)

    ingest.ingest_filing(fake_entity, filing)
    assert called["render"] is True


# --------------------------------------------------------------------------- #
# render_html_to_pdf
# --------------------------------------------------------------------------- #
def test_render_returns_path_when_pdf_created(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    html = tmp_path / "doc.htm"
    html.write_text("<html><body>hi</body></html>")
    pdf = tmp_path / "doc.pdf"

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        pdf.write_bytes(b"%PDF-1.4")   # Chrome "produces" the pdf

        class _R:
            returncode = 0
        return _R()
    monkeypatch.setattr(ingest.subprocess, "run", fake_run)

    out = ingest.render_html_to_pdf(html, pdf)
    assert out == pdf
    assert pdf.exists()

    argv = captured["cmd"]
    assert any("--headless" in a for a in argv)
    assert any("--print-to-pdf" in a for a in argv)
    # the target pdf path is wired into the --print-to-pdf flag
    assert any(f"--print-to-pdf={pdf}" == a for a in argv)
    assert captured["kwargs"].get("check") is True


def test_render_raises_when_pdf_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    html = tmp_path / "doc.htm"
    html.write_text("<html></html>")
    pdf = tmp_path / "missing.pdf"

    def fake_run(cmd, **kwargs):
        class _R:
            returncode = 0
        return _R()                      # does NOT create the pdf
    monkeypatch.setattr(ingest.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        ingest.render_html_to_pdf(html, pdf)
