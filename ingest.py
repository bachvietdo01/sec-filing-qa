"""Ingestion: fetch a filing's primary document and render it to PDF.

We render the EDGAR HTML to PDF (headless Chrome) on purpose: the assignment's
hard problem is messy PDF tables, not HTML <table> tags. Parsing a rendered PDF
confronts the real layout problem rather than dodging it via HTML structure.

This runs at INGESTION time only and writes raw source docs into filings/.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import config
import edgar


def render_html_to_pdf(html_path: Path, pdf_path: Path, timeout: int = 180) -> Path:
    """Render a local HTML file to PDF using installed headless Chrome."""
    profile = config.CACHE_DIR / "chrome-profile"  # isolated; don't touch user's profile
    cmd = [
        config.CHROME_BIN,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--user-data-dir={profile}",
        "--no-pdf-header-footer",
        "--virtual-time-budget=20000",  # let the page settle, then stop
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(
        cmd, check=True, timeout=timeout,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not pdf_path.exists():
        raise RuntimeError(f"Chrome did not produce {pdf_path}")
    return pdf_path


def ingest_filing(entity: edgar.Entity, filing: dict) -> dict:
    """Fetch the primary .htm and render it to .pdf. Idempotent (skips if present)."""
    stem = f"{entity.ticker}_{filing['form'].replace('-', '')}_{filing['report_date']}"
    htm_path = config.FILINGS_DIR / f"{stem}.htm"
    pdf_path = config.FILINGS_DIR / f"{stem}.pdf"

    if not htm_path.exists():
        data = edgar.fetch_document(
            entity.cik, filing["accession_no"], filing["primary_document"]
        )
        htm_path.write_bytes(data)

    if not pdf_path.exists():
        render_html_to_pdf(htm_path, pdf_path)

    return {"stem": stem, "htm": htm_path, "pdf": pdf_path,
            "entity": entity, "filing": filing}


if __name__ == "__main__":
    cands, status = edgar.resolve_entity("Tesla")
    ent = cands[0]
    filings = edgar.discover_filings(ent.cik, limit=10)
    tenk = next(f for f in filings if f["form"] == "10-K")  # FY2025 10-K
    print(f"ingesting {tenk['form']} {tenk['report_date']} ({tenk['primary_document']}) ...")
    res = ingest_filing(ent, tenk)
    print("htm:", res["htm"].name, f"{res['htm'].stat().st_size/1e6:.2f} MB")
    print("pdf:", res["pdf"].name, f"{res['pdf'].stat().st_size/1e6:.2f} MB")

    import pdfplumber
    with pdfplumber.open(res["pdf"]) as pdf:
        print("pdf pages:", len(pdf.pages))
        # peek: find a page mentioning 'Net income' to confirm tables rendered
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            if "Total revenues" in txt or "Net income" in txt:
                print(f"  first financial-statement hit on page {i+1}")
                break
