"""Ingestion pipeline (PDF -> facts dataset) with a CSV processing/error log.

Every filing we try to process is recorded as one row in
`store/ingest_log.csv`, with a `status`:
  * ok    — rendered, extracted, all three statements present
  * warn  — extracted, but a statement is missing (partial coverage)
  * error — an exception at some stage (resolve / discover / render / extract)

So at any point you can open the CSV and see exactly which PDFs failed and why
— the operational companion to the grounding invariant.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import config
import edgar
import extract
import ingest

LOG_PATH = config.STORE_DIR / "ingest_log.csv"
_FIELDS = ["timestamp", "ticker", "cik", "form", "report_date", "stem", "stage",
           "status", "n_facts", "statements", "grounded", "ungrounded",
           "error_type", "error"]
_ALL_STMTS = {"operations", "balance_sheet", "cash_flows"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_log(rec: dict, path: Path = LOG_PATH) -> None:
    is_new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow({k: rec.get(k, "") for k in _FIELDS})


def read_log(path: Path = LOG_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def process_filing(entity, filing: dict, log: bool = True) -> dict:
    """Render + extract one filing. Returns {record, facts}; never raises —
    a failure is captured in the record (status='error') and logged."""
    rec = {"timestamp": _now(), "ticker": entity.ticker, "cik": entity.cik,
           "form": filing.get("form", ""),
           "report_date": filing.get("report_date", ""), "stage": "start",
           "status": "ok"}
    facts: list = []
    try:
        rec["stage"] = "render"
        res = ingest.ingest_filing(entity, filing)
        rec["stem"] = res["stem"]
        rec["stage"] = "extract"
        facts = extract.extract_facts(res["pdf"], entity, filing)
        stmts = sorted({f.statement for f in facts})
        grounded = sum(f.grounding_ok for f in facts)
        rec.update(stage="done", n_facts=len(facts), statements="|".join(stmts),
                   grounded=grounded, ungrounded=len(facts) - grounded)
        missing = _ALL_STMTS - set(stmts)
        if missing:
            rec["status"] = "warn"
            rec["error"] = "missing statements: " + ",".join(sorted(missing))
    except Exception as e:
        rec.update(status="error", error_type=type(e).__name__, error=str(e)[:300])
    if log:
        append_log(rec)
    return {"record": rec, "facts": facts}


def ingest_and_bank(tickers, forms=("10-K", "10-Q"), limit: int = 12,
                    do_bank: bool = True) -> list[dict]:
    """Resolve, fetch, render, extract and (optionally) bank the latest filing
    of each requested form for each ticker. Logs every attempt to the CSV,
    including resolve/discover failures (e.g. a foreign filer with no 10-K)."""
    import bank
    collected: list[dict] = []
    for tk in tickers:
        cands, status = edgar.resolve_entity(tk)
        if status != "exact":
            append_log({"timestamp": _now(), "ticker": tk, "stage": "resolve",
                        "status": "error", "error_type": f"resolve_{status}",
                        "error": f"{status}: {[c.title for c in cands[:3]]}"})
            continue
        ent = cands[0]
        found = edgar.discover_filings(ent.cik, forms=forms, limit=limit)
        picks = [nf for nf in (next((x for x in found if x["form"] == form), None)
                               for form in forms) if nf]
        if not picks:
            append_log({"timestamp": _now(), "ticker": tk, "cik": ent.cik,
                        "stage": "discover", "status": "error",
                        "error_type": "no_filings",
                        "error": f"no {list(forms)} found (foreign filer? files 20-F?)"})
            continue
        for f in picks:
            out = process_filing(ent, f)
            if out["record"]["status"] in ("ok", "warn"):
                collected.extend(extract.facts_to_dicts(out["facts"]))
    if do_bank and collected:
        bank.append_facts(collected)
    return collected


if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] or ["AAL"]
    ingest_and_bank(tickers)
    print(f"log -> {LOG_PATH}")
    for r in read_log():
        print(f"  {r['ticker']:5} {r['form']:5} {r['status']:5} "
              f"{r.get('statements',''):40} {r.get('error','')[:50]}")
