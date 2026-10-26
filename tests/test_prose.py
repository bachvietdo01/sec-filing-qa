"""Unit tests for prose.py — grounded prose retrieval + narration.

Hermetic: section detection and chunking run on synthetic line lists; the
retriever is exercised on tiny chunk sets; the LLM narration path is
monkeypatched. One integration test uses the real rendered PDF (tsla_pdf).
"""
from __future__ import annotations

import prose
import llm


def _prose_lines(page, n=6, word="important disclosure about operations and "
                                  "results that is clearly body prose text"):
    """n prose-like lines (long, word-rich) on a given page."""
    return [(page, f"{word} number {i} continuing the paragraph onwards here")
            for i in range(n)]


# --------------------------------------------------------------------------- #
# header detection: real section start vs table-of-contents row
# --------------------------------------------------------------------------- #
def _doc():
    lines = [
        (1, "Table of Contents"),
        (1, "Item 1A. Risk Factors 12"),          # TOC row (no prose after)
        (1, "Item 1B. Unresolved Staff Comments 27"),
        (1, "Item 7. Management's Discussion 40"),
    ]
    lines += _prose_lines(2, n=2)                  # a little filler (not enough)
    lines += [(5, "ITEM 1A. RISK FACTORS")]        # REAL start of 1A
    lines += _prose_lines(5, n=6)
    lines += [(8, "ITEM 1B. UNRESOLVED STAFF COMMENTS")]  # REAL start of 1B
    lines += _prose_lines(8, n=6)
    lines += [(10, "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS")]
    lines += _prose_lines(10, n=6)
    lines += [(14, "ITEM 7A. QUANTITATIVE DISCLOSURES")]
    lines += _prose_lines(14, n=6)
    return lines


def test_item_re_matches_header_line():
    assert prose._item_re("1a").match("ITEM 1A. RISK FACTORS")
    assert prose._item_re("7").match("Item 7. Management's Discussion")
    assert not prose._item_re("1a").match("Item 1B. Something")


def test_real_start_picks_body_not_toc():
    lines = _doc()
    idx, pg = prose._real_start(lines, "1a")
    assert pg == 5                              # the real section, not the TOC p1
    assert lines[idx][1].upper().startswith("ITEM 1A")


def test_real_start_none_when_only_toc():
    lines = [(1, "Item 1A. Risk Factors 12"), (1, "Item 1B. Comments 27")]
    assert prose._real_start(lines, "1a") is None


def test_section_span_bounds_at_next_item():
    lines = _doc()
    s_idx, e_idx, s_pg = prose._section_span(lines, "1a", ("1b", "1c", "2"))
    assert s_pg == 5
    # the span must stop at the real Item 1B start, not run to the end
    assert lines[e_idx][1].upper().startswith("ITEM 1B")


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
def test_chunk_section_provenance_and_size():
    lines = _doc()
    s_idx, e_idx, _ = prose._section_span(lines, "1a", ("1b",))
    chunks = prose._chunk_section(lines, s_idx, e_idx, "risk_factors",
                                  "Item 1A. Risk Factors", target_chars=120)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c["section"] == "risk_factors"
        assert c["page_start"] <= c["page_end"]
        assert len(c["text"]) >= 80
        assert "ITEM 1A" not in c["text"].upper()   # header line excluded


# --------------------------------------------------------------------------- #
# tokenizer + TF-IDF retriever
# --------------------------------------------------------------------------- #
def test_tokenize_drops_stopwords_and_shorts():
    toks = prose._tokenize("The competition in the EV market is intense a")
    assert "the" not in toks and "in" not in toks and "is" not in toks
    assert "competition" in toks and "market" in toks and "ev" in toks
    assert "a" not in toks


def test_index_ranks_relevant_chunk_first():
    chunks = [
        {"text": "We face intense competition from other electric vehicle "
                 "manufacturers which could reduce our margins and market share"},
        {"text": "Cash flows from financing activities included repayments of "
                 "debt and proceeds from the issuance of common stock"},
    ]
    idx = prose.ProseIndex(chunks)
    hits = idx.search("competition from electric vehicle rivals", k=2)
    assert hits[0][0] is chunks[0]               # relevant chunk ranks first
    assert hits[0][1] > 0


def test_index_returns_empty_on_no_term_overlap():
    chunks = [{"text": "electric vehicle battery manufacturing capacity"}]
    idx = prose.ProseIndex(chunks)
    assert idx.search("xyzzy plugh frobnicate") == []


def test_index_respects_k():
    chunks = [{"text": f"competition risk factor number {i} affecting margins"}
              for i in range(5)]
    idx = prose.ProseIndex(chunks)
    assert len(idx.search("competition margins risk", k=3)) == 3


# --------------------------------------------------------------------------- #
# answer_prose: refuse / extractive / narrate / fallback
# --------------------------------------------------------------------------- #
def _chunks(cik):
    return [
        {"section": "risk_factors", "section_name": "Item 1A. Risk Factors",
         "page_start": 23, "page_end": 23, "filing_stem": "TSLA_10K_2025-12-31",
         "form": "10-K", "cik": cik, "ticker": "TSLA", "company": "Tesla, Inc.",
         "text": "We face intense competition in the electric vehicle market "
                 "that could reduce our sales, margins and profitability."},
    ]


def test_answer_prose_refuses_without_chunks(monkeypatch):
    import bank
    monkeypatch.setattr(bank, "load_prose", lambda path=None: [])
    a = prose.answer_prose("What are the risks?", cik=1318605)
    assert a.status == "refused" and a.confidence == 0.0


def test_answer_prose_refuses_on_no_overlap(monkeypatch):
    import bank
    monkeypatch.setattr(bank, "load_prose", lambda path=None: _chunks(1318605))
    a = prose.answer_prose("xyzzy plugh frobnicate", cik=1318605)
    assert a.status == "refused"


def test_answer_prose_extractive_when_offline(monkeypatch, no_llm):
    import bank
    monkeypatch.setattr(bank, "load_prose", lambda path=None: _chunks(1318605))
    a = prose.answer_prose("competition risks in the EV market", cik=1318605)
    assert a.status == "answered"
    assert a.passages and a.citations
    assert a.citations[0]["page_start"] == 23
    assert "no LLM configured" in a.text              # honest extractive label


def test_answer_prose_narrates_when_llm_available(monkeypatch):
    import bank
    monkeypatch.setattr(bank, "load_prose", lambda path=None: _chunks(1318605))
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "complete",
                        lambda sys, user, max_tokens=400: "Tesla cites EV "
                        "competition as a key risk [p.23].")
    a = prose.answer_prose("competition risks", cik=1318605)
    assert a.status == "answered"
    assert a.text == "Tesla cites EV competition as a key risk [p.23]."


def test_answer_prose_falls_back_when_llm_errors(monkeypatch):
    import bank
    monkeypatch.setattr(bank, "load_prose", lambda path=None: _chunks(1318605))
    monkeypatch.setattr(llm, "available", lambda: True)

    def _boom(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(llm, "complete", _boom)
    a = prose.answer_prose("competition risks", cik=1318605)
    assert a.status == "answered"
    assert "no LLM configured" in a.text              # extractive fallback


# --------------------------------------------------------------------------- #
# integration: real rendered 10-K
# --------------------------------------------------------------------------- #
def test_extract_prose_integration(tsla_pdf, fake_entity):
    filing = {"form": "10-K", "accession_no": "0001628280-25-000000"}
    chunks = prose.extract_prose(tsla_pdf, fake_entity, filing)
    sections = {c["section"] for c in chunks}
    assert "risk_factors" in sections and "mdna" in sections
    assert len(chunks) > 20
    for c in chunks[:5]:
        assert c["cik"] == fake_entity.cik
        assert c["page_start"] > 0 and c["text"]
