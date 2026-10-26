"""Prose side of the Bank: grounded retrieval + narration over filing text.

The numeric spine answers quantitative questions from the Fact Store. Prose
questions ("what are the main risks?", "what does management say about margins?")
are answered from the narrative sections — Item 1A (Risk Factors) and Item 7
(MD&A) — which hold text, not tables.

Design (the prose mirror of "code finds the number, LLM phrases it"):
    extract_prose   pull the narrative sections out of the SAME rendered PDF,
                    chunk them with page provenance              (code)
    ProseIndex      rank chunks against the question             (code)
    answer_prose    LLM narrates ONLY over the retrieved chunks, cites pages,
                    refuses when nothing relevant is retrieved   (LLM, fenced)

Retrieval is vector-space similarity over prose — TF-IDF cosine here, which is
dependency-light and fully inspectable (you can see which terms matched). Dense
neural embeddings are the drop-in upgrade at scale; the interface would not
change. We deliberately never embed numbers (magnitude-blind, topically
collapsed); that is why this module is prose-only.

Trust honesty: prose grounding is SOFTER than the numeric side. Numbers are
matched verbatim; prose is paraphrased by the LLM, so three cracks remain
(overstatement, smuggled outside knowledge, wrong-passage retrieval). The
mitigation is to return the cited source passages alongside the answer so a
human can verify in one glance — never narration without its sources.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import bank
import extract
import llm

# --------------------------------------------------------------------------- #
# The narrative sections we index, with the item headers that bound each one.
# --------------------------------------------------------------------------- #
SECTIONS = {
    "risk_factors": {
        "name": "Item 1A. Risk Factors",
        "start": "1a", "end": ("1b", "1c", "2"),
    },
    "mdna": {
        "name": "Item 7. Management's Discussion and Analysis",
        "start": "7", "end": ("7a", "8"),
    },
}
_BOUNDARY_KEYS = ("1a", "1b", "1c", "2", "7", "7a", "8")


def _item_re(key: str) -> re.Pattern:
    # match an item header at the start of a line, e.g. "ITEM 1A. RISK FACTORS"
    return re.compile(rf"^item\s+{re.escape(key)}[.\s)]", re.I)


def _prose_after(lines: list[tuple[int, str]], k: int, span: int = 20) -> int:
    """Count prose-like lines just after index k. Distinguishes a real section
    start (lots of prose follows) from a table-of-contents row (short, dotted,
    or more item rows follow)."""
    n = 0
    for _pg, nl in lines[k + 1:k + 1 + span]:
        t = nl.strip()
        if (len(t) > 60 and t.count(" ") >= 10 and "....." not in t
                and not t.lower().startswith("item ") and not t.isupper()):
            n += 1
    return n


def _real_start(lines: list[tuple[int, str]], key: str) -> tuple[int, int] | None:
    """Return (line_index, page) of the REAL start of an item section: the
    header occurrence with the most prose following it (the TOC rows have ~none).
    """
    pat = _item_re(key)
    best = None  # (prose_after, idx, page)
    for k, (pg, ln) in enumerate(lines):
        if pat.match(ln.strip()):
            pa = _prose_after(lines, k)
            if best is None or pa > best[0]:
                best = (pa, k, pg)
    if best is None or best[0] < 5:   # no occurrence with real body text
        return None
    return best[1], best[2]


def _section_span(lines, start_key, end_keys) -> tuple[int, int, int] | None:
    """(start_idx, end_idx_exclusive, start_page) for one section, or None."""
    start = _real_start(lines, start_key)
    if start is None:
        return None
    s_idx, s_pg = start
    ends = [_real_start(lines, e) for e in end_keys]
    ends = [e[0] for e in ends if e is not None and e[0] > s_idx]
    e_idx = min(ends) if ends else len(lines)
    return s_idx, e_idx, s_pg


# --------------------------------------------------------------------------- #
# Chunking (page-provenanced passages)
# --------------------------------------------------------------------------- #
def _chunk_section(lines, s_idx, e_idx, section_key, section_name,
                   target_chars: int = 800) -> list[dict]:
    """Group the section's lines into ~target_chars passages, each tagged with
    the page range it spans. One line of overlap keeps sentences from being
    orphaned across a boundary."""
    chunks: list[dict] = []
    buf: list[str] = []
    buf_pages: list[int] = []
    size = 0

    def flush():
        nonlocal buf, buf_pages, size
        text = " ".join(buf).strip()
        if len(text) >= 80:  # skip stub fragments
            chunks.append({
                "section": section_key, "section_name": section_name,
                "page_start": min(buf_pages), "page_end": max(buf_pages),
                "text": text,
            })

    for pg, ln in lines[s_idx + 1:e_idx]:   # skip the header line itself
        t = ln.strip()
        if not t or t.isupper() and len(t) < 60:   # drop sub-headers / blanks
            continue
        buf.append(t)
        buf_pages.append(pg)
        size += len(t) + 1
        if size >= target_chars:
            flush()
            buf, buf_pages, size = ([buf[-1]], [buf_pages[-1]], len(buf[-1]))  # 1-line overlap
    if buf:
        flush()
    return chunks


def extract_prose(pdf_path, entity, filing) -> list[dict]:
    """Parse the narrative sections of a rendered 10-K/10-Q into prose chunks
    with provenance. Mirror of extract.extract_facts; writes nothing."""
    import pdfplumber
    stem = str(pdf_path).rsplit("/", 1)[-1].rsplit(".", 1)[0]
    out: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        lines = extract._page_lines(pdf)
    for key, spec in SECTIONS.items():
        span = _section_span(lines, spec["start"], spec["end"])
        if span is None:
            continue
        s_idx, e_idx, _ = span
        for ch in _chunk_section(lines, s_idx, e_idx, key, spec["name"]):
            ch.update(company=entity.title, cik=entity.cik, ticker=entity.ticker,
                      filing_stem=stem, form=filing["form"],
                      accession_no=filing.get("accession_no", ""))
            out.append(ch)
    return out


# --------------------------------------------------------------------------- #
# Retrieval: TF-IDF cosine (pure Python, inspectable, zero heavy deps)
# --------------------------------------------------------------------------- #
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "at", "by",
    "is", "are", "was", "were", "be", "been", "as", "that", "this", "these",
    "those", "it", "its", "we", "our", "us", "with", "from", "which", "such",
    "may", "will", "can", "could", "would", "has", "have", "had", "not", "but",
    "if", "than", "then", "there", "their", "also", "any", "all", "about",
    "what", "does", "do", "how", "when", "where", "who", "why",
}


def _tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if len(t) >= 2 and t not in _STOP]


@dataclass
class ProseIndex:
    """In-memory TF-IDF index over prose chunks (one filing's worth here;
    at scale this is where a persisted vector index would sit)."""
    chunks: list[dict]
    idf: dict = field(default_factory=dict)
    vecs: list[dict] = field(default_factory=list)   # normalized tf-idf per chunk

    def __post_init__(self):
        n = len(self.chunks)
        df: dict[str, int] = {}
        toks_per = []
        for c in self.chunks:
            toks = _tokenize(c["text"])
            toks_per.append(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}
        self.vecs = [self._vec(toks) for toks in toks_per]

    def _vec(self, toks: list[str]) -> dict:
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        vec = {t: (1 + math.log(f)) * self.idf.get(t, 0.0) for t, f in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[dict, float]]:
        q = self._vec(_tokenize(query))
        scored = []
        for c, v in zip(self.chunks, self.vecs):
            # cosine = dot product (both already L2-normalized)
            s = sum(qw * v.get(t, 0.0) for t, qw in q.items())
            if s > 0:
                scored.append((c, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# --------------------------------------------------------------------------- #
# Answer: LLM narrates over retrieved passages, cites pages, or refuses
# --------------------------------------------------------------------------- #
@dataclass
class ProseAnswer:
    text: str
    status: str                 # answered | refused
    confidence: float
    section_hits: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)  # page/section/score/snippet
    passages: list[dict] = field(default_factory=list)    # retrieved source chunks


_NARRATE_SYS = (
    "You answer a question about a company's SEC filing using ONLY the numbered "
    "excerpts provided, in a natural, professional tone. Rules: (1) no outside "
    "knowledge; (2) stay close to the source wording; (3) cite the page(s) you used "
    "in square brackets like [p.23]; (4) any figure you mention (a dollar amount or "
    "a percentage) MUST appear inside double quotes, copied word-for-word from the "
    "excerpt together with the words that say what it measures — e.g. \"revenue "
    "increased $1.2 billion\" — never re-label or restate a figure in your own words; "
    "(5) if the excerpts do not answer the question, say so in one sentence and name "
    "the closest thing they do cover; (6) no greeting or filler, no markdown, no "
    "exclamation marks. Two to five sentences."
)


def _narrate(question: str, hits: list[tuple[dict, float]]) -> str:
    excerpts = "\n\n".join(
        f"[{i+1}] (p.{c['page_start']}, {c['section_name']})\n{c['text']}"
        for i, (c, _s) in enumerate(hits)
    )
    user = f"Question: {question}\n\nExcerpts from the filing:\n{excerpts}"
    return llm.complete(_NARRATE_SYS, user, max_tokens=400)


def answer_prose(question: str, cik: int, k: int = 4,
                 min_score: float = 0.05) -> ProseAnswer:
    """Retrieve relevant passages for a prose question and narrate over them."""
    chunks = [c for c in bank.load_prose() if c.get("cik") == cik]
    if not chunks:
        return ProseAnswer(
            "I have no narrative sections banked for this company, so I can't "
            "answer a prose question about it yet.", "refused", 0.0)

    # Section scoping: if the question clearly targets a section, answer only from
    # it — and refuse honestly if that section isn't banked (don't silently
    # substitute Risk Factors for a missing MD&A).
    ql = question.lower()
    want = None
    if any(w in ql for w in ["md&a", "mdna", "management discussion",
                             "discussion and analysis", "liquidity"]):
        want = "mdna"
    elif "risk" in ql:
        want = "risk_factors"
    if want:
        avail = {c.get("section") for c in chunks}
        if want not in avail:
            human = {"mdna": "MD&A (Item 7)",
                     "risk_factors": "Risk Factors (Item 1A)"}[want]
            return ProseAnswer(
                f"I don't have the {human} section banked for this company, so I "
                "won't answer that from a different section.", "refused", 0.0)
        chunks = [c for c in chunks if c.get("section") == want]

    hits = ProseIndex(chunks).search(question, k=k)
    # A weak top match additionally needs the question's content terms to be
    # reasonably COVERED by the retrieved passages — a single incidental token
    # ("strategy") clearing a low score isn't enough. Strong matches skip this
    # (a high cosine is its own evidence). NB: lexical cosine can't fully separate
    # a high-IDF off-topic token from an on-topic one (see design_doc); the LLM
    # narrator is the semantic backstop and refuses when excerpts don't answer.
    if hits and min_score <= hits[0][1] < 0.10:
        q_terms = set(_tokenize(question))
        covered = set()
        for c, _s in hits:
            covered |= (q_terms & set(_tokenize(c["text"])))
        if q_terms and len(covered) < max(2, len(q_terms) // 2):
            hits = []
    if not hits or hits[0][1] < min_score:
        return ProseAnswer(
            "I couldn't find a passage in the filing relevant to that question, "
            "so I won't answer from guesswork.", "refused", 0.0)

    citations = [{
        "section_name": c["section_name"], "page_start": c["page_start"],
        "page_end": c["page_end"], "score": round(s, 3),
        "snippet": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
    } for c, s in hits]
    section_hits = sorted({c["section"] for c, _ in hits})
    passages = [c for c, _ in hits]
    confidence = round(min(0.9, hits[0][1] * 2), 2)  # retrieval-strength proxy

    if llm.available():
        try:
            text = _narrate(question, hits)
        except Exception:
            text = _extractive(hits)
    else:
        text = _extractive(hits)

    return ProseAnswer(text, "answered", confidence, section_hits,
                       citations, passages)


def _extractive(hits: list[tuple[dict, float]]) -> str:
    """Offline fallback: return the top passages verbatim (no LLM narration).
    Honest about what it is — the raw grounded source, not a synthesized answer."""
    top = hits[0][0]
    body = top["text"]
    body = body[:700] + ("..." if len(body) > 700 else "")
    return (f"[no LLM configured — showing the top retrieved passage verbatim] "
            f"(p.{top['page_start']}, {top['section_name']}):\n{body}")


if __name__ == "__main__":
    import config
    import edgar

    ent = edgar.resolve_entity("Tesla")[0][0]
    filing = {"form": "10-K", "accession_no": "0001628280-25-000000"}
    pdf = config.FILINGS_DIR / "TSLA_10K_2025-12-31.pdf"

    chunks = extract_prose(pdf, ent, filing)
    print(f"extracted {len(chunks)} prose chunks")
    for key in SECTIONS:
        sub = [c for c in chunks if c["section"] == key]
        if sub:
            print(f"  {key}: {len(sub)} chunks, "
                  f"pages {sub[0]['page_start']}-{sub[-1]['page_end']}")
    bank.write_prose(chunks)

    idx = ProseIndex(chunks)
    for q in ["What are Tesla's main risks related to competition?",
              "What does management say about automotive gross margin?"]:
        hits = idx.search(q, k=3)
        print(f"\nQ: {q}")
        for c, s in hits:
            print(f"   {s:.3f}  p{c['page_start']} [{c['section']}]  "
                  f"{c['text'][:90]}...")
