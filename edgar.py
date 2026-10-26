"""Sparing EDGAR access — the ingestion front-end.

Deterministic only, NO LLM. Three jobs:
  1. resolve_entity(): company mention -> CIK, from SEC's official STATIC ticker
     map (downloaded once, cached). A CIK is never guessed/hallucinated.
  2. discover_filings(): which 10-K/10-Q filings exist for a CIK (one low-volume
     submissions lookup).
  3. fetch_document(): download a single primary filing document.

EDGAR is a constrained resource: a cached static file + polite, throttled lookups.
It is used at INGESTION time only, never in the per-question answer path.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

import config


@dataclass
class Entity:
    cik: int
    ticker: str
    title: str


# --------------------------------------------------------------------------- #
# 1. Entity resolution (static CIK map)
# --------------------------------------------------------------------------- #
def _tickers_path() -> Path:
    return config.CACHE_DIR / "company_tickers.json"


def _load_tickers() -> list[dict]:
    """Download SEC's static ticker->CIK map once, then read from cache."""
    p = _tickers_path()
    if not p.exists():
        r = requests.get(
            config.EDGAR_TICKERS_URL,
            headers={"User-Agent": config.EDGAR_UA},
            timeout=30,
        )
        r.raise_for_status()
        p.write_text(r.text)
    raw = json.loads(p.read_text())
    # raw is {"0": {"cik_str": int, "ticker": str, "title": str}, ...}
    return list(raw.values())


def resolve_entity(mention: str) -> tuple[list[Entity], str]:
    """Map a company mention to candidate SEC entities, deterministically.

    Returns (candidates, status) where status is one of:
      'exact'     -> exactly one confident match (candidates[0])
      'ambiguous' -> multiple plausible matches -> caller must disambiguate
      'not_found' -> no match -> caller surfaces honestly

    The LLM may extract the mention upstream; it never produces the CIK.
    """
    if not mention or not mention.strip():
        return [], "not_found"   # no mention -> never match (an empty string is
                                 # a substring of every title); refuse honestly
    rows = _load_tickers()
    m_up = mention.strip().upper()
    m_low = mention.strip().lower()

    def mk(r: dict) -> Entity:
        return Entity(cik=int(r["cik_str"]), ticker=r["ticker"], title=r["title"])

    def resolve(cands: list[Entity]) -> tuple[list[Entity], str]:
        # Candidates that share ONE CIK are the same legal entity under multiple
        # share classes (Alphabet GOOG/GOOGL) — that is an exact, not ambiguous,
        # resolution. Only a genuine multi-CIK set is ambiguous.
        if not cands:
            return [], "not_found"
        if len({c.cik for c in cands}) == 1:
            return [cands[0]], "exact"
        return cands[:10], "ambiguous"

    # 1) exact ticker match (e.g. "TSLA")
    exact_ticker = [mk(r) for r in rows if r["ticker"].upper() == m_up]
    if exact_ticker:
        return resolve(exact_ticker)

    # 2) exact title match (e.g. "Tesla, Inc.")
    exact_title = [mk(r) for r in rows if r["title"].lower() == m_low]
    if exact_title:
        return resolve(exact_title)

    # 3) title contains the mention (e.g. "Tesla"). Rank so the most relevant
    # survive the candidate cap: a title that STARTS with the mention first
    # ("American Airlines..." for "American"), then shorter titles (a closer
    # match), then alphabetically — otherwise raw SEC-dict order could truncate
    # the intended company out of the list entirely.
    contains = [mk(r) for r in rows if m_low in r["title"].lower()]
    if contains:
        contains.sort(key=lambda c: (not c.title.lower().startswith(m_low),
                                     len(c.title), c.title.lower()))
        return resolve(contains)

    return [], "not_found"


# --------------------------------------------------------------------------- #
# 2. Filing discovery
# --------------------------------------------------------------------------- #
def discover_filings(
    cik: int,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    limit: int = 10,
) -> list[dict]:
    """Return recent filings of the given forms for a CIK (one submissions lookup)."""
    cik10 = f"{cik:010d}"
    url = config.EDGAR_SUBMISSIONS_URL.format(cik10=cik10)
    time.sleep(config.EDGAR_MIN_INTERVAL_S)
    r = requests.get(url, headers={"User-Agent": config.EDGAR_UA}, timeout=30)
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]

    out: list[dict] = []
    for i, form in enumerate(recent["form"]):
        if form not in forms:
            continue
        out.append(
            {
                "form": form,
                "accession_no": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "primary_document": recent["primaryDocument"][i],
                "primary_doc_desc": recent["primaryDocDescription"][i],
            }
        )
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# 3. Document fetch
# --------------------------------------------------------------------------- #
def document_url(cik: int, accession_no: str, primary_document: str) -> str:
    acc_nodash = accession_no.replace("-", "")
    return config.EDGAR_ARCHIVE_URL.format(
        cik=cik, acc_nodash=acc_nodash, doc=primary_document
    )


def fetch_document(cik: int, accession_no: str, primary_document: str) -> bytes:
    """Download one primary filing document (bytes). Throttled + UA header."""
    url = document_url(cik, accession_no, primary_document)
    time.sleep(config.EDGAR_MIN_INTERVAL_S)
    r = requests.get(url, headers={"User-Agent": config.EDGAR_UA}, timeout=60)
    r.raise_for_status()
    return r.content


if __name__ == "__main__":
    # smoke test
    cands, status = resolve_entity("Tesla")
    print("resolve_entity('Tesla') ->", status)
    for c in cands[:3]:
        print("  ", c)
    if status == "exact":
        fs = discover_filings(cands[0].cik, limit=5)
        print(f"recent 10-K/10-Q for {cands[0].ticker}:")
        for f in fs:
            print("  ", f["form"], f["filing_date"], f["report_date"], f["primary_document"])
