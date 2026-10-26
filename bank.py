"""THE BANK: the system of record that sits between ingestion and answering.

Two stores, written once at ingestion, read fast and deterministically per
question:

  * Fact Store  - structured, grounded statement facts (from extract.py). This
                  is what every numeric answer is drawn from. Numbers are matched
                  here by label + period, never re-read from the PDF per question.
  * Prose Store - risk / MD&A text chunks with page provenance, for the prose
                  questions. Embeddings are attached lazily by the prose layer;
                  the Bank just persists chunks + (optional) vectors.

At 3-filing scale this is JSON on disk + in-memory filtering. At full scale the
same interface backs onto columnar storage (DuckDB/Parquet) for the facts and a
vector index for the prose; callers would not change.
"""
from __future__ import annotations

import json
from pathlib import Path

import config

FACTS_PATH = config.STORE_DIR / "facts.json"
PROSE_PATH = config.STORE_DIR / "prose.json"


# --------------------------------------------------------------------------- #
# Fact Store
# --------------------------------------------------------------------------- #
def write_facts(fact_dicts: list[dict], path: Path = FACTS_PATH) -> Path:
    """Persist facts, refusing to bank an ungrounded number (trust invariant)."""
    ungrounded = [f for f in fact_dicts if not f.get("grounding_ok")]
    if ungrounded:
        raise ValueError(
            f"refusing to bank {len(ungrounded)} ungrounded fact(s); "
            "every number must be verbatim in its source page"
        )
    path.write_text(json.dumps(fact_dicts, indent=2))
    return path


def append_facts(fact_dicts: list[dict], path: Path = FACTS_PATH) -> Path:
    """Add facts from another filing, de-duplicated on (cik, filing, stmt,
    label, period)."""
    existing = load_facts(path)
    seen = {_fact_key(f) for f in existing}
    merged = existing + [f for f in fact_dicts if _fact_key(f) not in seen]
    return write_facts(merged, path)


def load_facts(path: Path = FACTS_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _fact_key(f: dict) -> tuple:
    return (f["cik"], f["filing_stem"], f["statement"], f["raw_label"], f["period"])


def facts_for(cik: int, path: Path = FACTS_PATH) -> list[dict]:
    """All banked facts for one company (by CIK — the resolved identity)."""
    return [f for f in load_facts(path) if f["cik"] == cik]


def distinct_labels(cik: int, statement: str | None = None,
                    path: Path = FACTS_PATH) -> list[str]:
    """Unique line-item labels available for a company (optionally one stmt).
    Used by the matcher to offer real candidates instead of guessing."""
    out: list[str] = []
    for f in facts_for(cik, path):
        if statement and f["statement"] != statement:
            continue
        if f["raw_label"] not in out:
            out.append(f["raw_label"])
    return out


def periods_for(cik: int, path: Path = FACTS_PATH) -> list[str]:
    seen: list[str] = []
    for f in facts_for(cik, path):
        if f["period"] not in seen:
            seen.append(f["period"])
    return sorted(seen)


# --------------------------------------------------------------------------- #
# Prose Store
# --------------------------------------------------------------------------- #
def write_prose(chunks: list[dict], path: Path = PROSE_PATH) -> Path:
    """Persist prose chunks. Each: {filing_stem, section, page_no, text,
    embedding?}."""
    path.write_text(json.dumps(chunks, indent=2))
    return path


def load_prose(path: Path = PROSE_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def prose_for(cik_stem_prefix: str, path: Path = PROSE_PATH) -> list[dict]:
    return [c for c in load_prose(path)
            if c.get("filing_stem", "").startswith(cik_stem_prefix)]


if __name__ == "__main__":
    import edgar
    import extract

    cands, _ = edgar.resolve_entity("Tesla")
    ent = cands[0]

    # Bank the annual 10-K and the interim 10-Q (if present). Different filings
    # coexist: they carry distinct period labels (FY2025 vs 2026Q2 / 2026YTD6M).
    filings = [
        ("10-K", config.FILINGS_DIR / "TSLA_10K_2025-12-31.pdf"),
        ("10-Q", config.FILINGS_DIR / "TSLA_10Q_2026-06-30.pdf"),
    ]
    all_facts: list[dict] = []
    for form, pdf in filings:
        if not pdf.exists():
            print(f"(skip {form}: {pdf.name} not present)")
            continue
        fs = extract.facts_to_dicts(
            extract.extract_facts(pdf, ent, {"form": form, "accession_no": ""}))
        all_facts.extend(fs)
        print(f"{form}: {len(fs)} facts from {pdf.name}")
    write_facts(all_facts)
    print(f"banked {len(all_facts)} facts -> {FACTS_PATH}")
    print("periods:", periods_for(ent.cik))
