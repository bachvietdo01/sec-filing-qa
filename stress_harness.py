"""Isolated stress-test harness for the adversarial generalization audit.

Ingests a NEW company (not in the shared Bank) into a PRIVATE temp directory —
its own filings/ and its own Chrome profile/cache, so many agents can ingest in
parallel without colliding — and answers questions against ONLY that company's
facts by patching `bank.facts_for` in-process. The real `store/facts.json` is
never read or written.

Usage (offline, deterministic — no API spend):
    import stress_harness as H
    h = H.ingest("PLTR")              # fetch + render + extract into a temp bank
    print(h.n_facts, h.statements)    # coverage
    a = h.answer("What was revenue in 2025?")
    print(a.status, a.text)
    for c in a.trace.citations: ...   # grounding
"""
from __future__ import annotations

import tempfile
import unittest.mock as mock
from pathlib import Path

import config
import edgar


class Harness:
    def __init__(self, entity, facts, prose_chunks, statements, log):
        self.entity = entity
        self.facts = facts
        self.prose = prose_chunks
        self.statements = statements
        self.n_facts = len(facts)
        self.log = log            # ingest notes (per-form status / errors)

    def answer(self, question):
        """Run the real engine against ONLY this company's facts, OFFLINE."""
        import bank, core, llm
        mine = [f for f in self.facts if f.get("cik") == self.entity.cik]
        myprose = [c for c in self.prose if c.get("cik") == self.entity.cik]
        with mock.patch.object(bank, "facts_for", lambda cik, *a, **k: mine), \
             mock.patch.object(bank, "load_prose", lambda *a, **k: myprose), \
             mock.patch.object(llm, "available", lambda: False):
            return core.answer(question, entity=self.entity)


def ingest(ticker: str, forms=("10-K", "10-Q")) -> Harness:
    """Ingest a company into a private temp bank. Raises if it can't resolve."""
    import pipeline
    import extract
    import prose as prose_mod

    tmp = Path(tempfile.mkdtemp(prefix=f"stress_{ticker}_"))
    orig_cache, orig_filings = config.CACHE_DIR, config.FILINGS_DIR
    # redirect the Chrome profile, the EDGAR ticker cache, and rendered filings
    # into this agent's private temp dir (parallel-safe).
    config.CACHE_DIR = tmp / "cache"
    config.FILINGS_DIR = tmp / "filings"
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    try:
        cands, status = edgar.resolve_entity(ticker)
        if status != "exact" or not cands:
            raise RuntimeError(f"resolve {ticker}: {status} ({[c.ticker for c in cands[:3]]})")
        ent = cands[0]
        found = edgar.discover_filings(ent.cik, forms=forms, limit=12)
        facts, chunks = [], []
        for form in forms:
            f = next((x for x in found if x["form"] == form), None)
            if not f:
                log.append(f"{form}: none found")
                continue
            out = pipeline.process_filing(ent, f, log=False)
            rec = out["record"]
            log.append(f"{form} {rec.get('report_date')}: {rec['status']} "
                       f"stmts={rec.get('statements')} n={rec.get('n_facts')} "
                       f"{rec.get('error','')}")
            if rec["status"] in ("ok", "warn"):
                facts += extract.facts_to_dicts(out["facts"])
                try:                        # prose is best-effort (10-K only here)
                    chunks += prose_mod.extract_prose(
                        config.FILINGS_DIR / f"{rec['stem']}.pdf", ent, f)
                except Exception as e:
                    log.append(f"{form} prose: {type(e).__name__}")
        stmts = sorted({fct["statement"] for fct in facts})
        return Harness(ent, facts, chunks, stmts, log)
    finally:
        config.CACHE_DIR, config.FILINGS_DIR = orig_cache, orig_filings


if __name__ == "__main__":
    import sys
    h = ingest(sys.argv[1] if len(sys.argv) > 1 else "PLTR")
    print("entity:", h.entity.ticker, h.entity.cik, "| facts:", h.n_facts,
          "| stmts:", h.statements)
    for ln in h.log:
        print("  ", ln)
    for q in ["What was revenue in 2025?", "What was net income in 2025?",
              "What was diluted EPS in 2025?", "How did revenue grow from 2024 to 2025?"]:
        a = h.answer(q)
        print(f"  [{a.status:8}] {q} -> {a.text[:90]}")
