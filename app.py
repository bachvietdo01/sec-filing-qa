"""Streamlit dashboard for the SEC-filing Q&A engine — a thin, trust-first skin
over core.answer().

Design synthesized from a 6-lens research fan-out. Principles that won consensus:
  * status-first: every answer is framed by its status + confidence; the claim is
    never shown naked, and a refusal reads as calm honesty (never a red error);
  * click-to-verify: each cited number exposes the verbatim filing cell (page +
    form) one click away;
  * st.metric hero for a single figure, delta only on a real growth comparison;
  * progressive disclosure: the reasoning trace is collapsed when confident,
    auto-expanded when hedged/refused;
  * a chat "audit ledger" — history of question -> structured answer card.

Deliberately uses only widely-available Streamlit APIs (no st.badge/pills/
segmented_control) and NO image/raster components, to stay compatible with the
host's Streamlit/pillow versions. Run it in a venv:  streamlit run app.py
"""
from __future__ import annotations

import re

import streamlit as st

import bank
import core
import voice
import llm
import prose

# Brand / common names that don't appear in the SEC legal title (so a plain
# title-substring match would miss them). Only needed for banked companies.
_ALIASES = {"google": "GOOG", "alphabet": "GOOG", "spacemobile": "ASTS"}
# Generic corporate words that aren't distinctive enough to identify a company.
_NAME_STOP = {"inc", "corp", "co", "group", "the", "company", "holdings", "ltd",
              "plc", "corporation", "technologies"}


def resolve_from_question(question: str, companies, default_ticker: str) -> str:
    """Pick the banked company the question is about — by ticker OR natural name
    — falling back to the sidebar default when the question names none. This is
    what lets the chat answer 'What was Nvidia's revenue?' or 'GOOG cash?'
    without relying on the picker."""
    ql = question.lower()
    # 1. an explicit ticker token (case-insensitive for >=3 chars; for short
    #    tickers like 'BE' require the uppercase form so the English word 'be'
    #    doesn't false-match).
    for tk, _name, _cik in companies:
        if len(tk) >= 3:
            if re.search(rf"\b{re.escape(tk.lower())}\b", ql):
                return tk
        elif re.search(rf"\b{re.escape(tk)}\b", question):
            return tk
    # 2. a brand/common-name alias
    for alias, tk in _ALIASES.items():
        if alias in ql and any(c[0] == tk for c in companies):
            return tk
    # 3. a distinctive word from the legal name ("disney", "broadcom", ...)
    for tk, name, _cik in companies:
        for w in re.findall(r"[a-z]{4,}", name.lower()):
            if w not in _NAME_STOP and re.search(rf"\b{w}\b", ql):
                return tk
    return default_ticker

st.set_page_config(page_title="Filing Q&A", page_icon="📑", layout="centered")

# --- one scoped style block: premium polish, stable [data-testid] selectors --- #
st.markdown(
    """
    <style>
      h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; letter-spacing:.01em; }
      [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-weight:600; }
      .d71-pill { display:inline-block; padding:2px 11px; border-radius:999px;
                  font-size:.72rem; font-weight:700; letter-spacing:.04em;
                  text-transform:uppercase; }
      .d71-answered { background:#e6f4ea; color:#1b5e20; }
      .d71-hedged   { background:#fdf3e2; color:#8a5a00; }
      .d71-refused  { background:#eaf1f8; color:#1f3a5f; }
      .d71-sub { color:#5b6672; font-size:.85rem; }
      .d71-q { color:#2a3346; font-size:.98rem; font-weight:600;
               margin:1.2rem 0 .4rem; padding:.15rem 0 .15rem .7rem;
               border-left:3px solid #1f3a5f; }
      .d71-q .tk { color:#1f3a5f; font-weight:700; }
      .d71-active { color:#5b6672; font-size:.9rem; }
      .d71-answer { font-size:1.2rem; line-height:1.55; font-weight:600;
                    color:#141b26; margin:.15rem 0 .5rem; }
      .d71-meta { margin:.15rem 0 .4rem; }
      [data-testid="stChatMessage"] { background:transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

def _md(s: str) -> str:
    """Escape '$' so Streamlit markdown doesn't treat '$...$' as LaTeX math
    (financial prose is full of dollar figures)."""
    return (s or "").replace("$", "\\$")


_STATUS = {
    "answered": ("d71-answered", "Answered"),
    "hedged":   ("d71-hedged", "Hedged"),
    "refused":  ("d71-refused", "Refused"),
}


# --------------------------------------------------------------------------- #
# Load the Bank ONCE (singleton) and memoize the engine's reads. core.answer
# reads bank.load_facts/load_prose per call; cache them so questions are snappy.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the Bank…")
def load_bank():
    facts = bank.load_facts()
    prose = bank.load_prose()
    bank.load_facts = lambda path=None: facts        # memoize the engine's reads
    bank.load_prose = lambda path=None: prose
    companies = sorted({(f["ticker"], f["company"], f["cik"]) for f in facts},
                       key=lambda t: t[0])
    return facts, prose, companies


def answer_for(question: str, ticker: str, title: str, cik: int, online: bool,
               context=None):
    # The company is resolved by the UI and handed to the engine directly (so a
    # follow-up like "its revenue?" stays anchored). NOT cached: the Answer
    # object isn't reliably picklable for st.cache_data, and the engine is only
    # called once per new question anyway (answers are stored in history and
    # re-rendered from there — no recompute on rerun).
    import edgar
    ent = edgar.Entity(cik=cik, ticker=ticker, title=title) if cik else None
    return core.answer(question, entity=ent, context=context)


@st.cache_data(show_spinner=False)
def company_summary(ticker: str, cik: int, online: bool):
    """A short business-model blurb GROUNDED in the company's own filing prose
    (Risk Factors / MD&A we banked) — never the model's memory. Summarized live
    when available; otherwise an extractive snippet. Cached per company."""
    chunks = [c for c in bank.load_prose() if c.get("cik") == cik]
    if not chunks:
        return None
    hits = prose.ProseIndex(chunks).search(
        "business overview products services segments operations how the company "
        "generates revenue and competes", k=4)
    if not hits:
        return None
    if online and llm.available():
        try:
            excerpts = "\n\n".join(f"[p.{c['page_start']}] {c['text'][:600]}"
                                   for c, _ in hits)
            sys = ("You are a financial assistant. In 2-3 plain sentences, "
                   "summarize this company's business model using ONLY the SEC-"
                   "filing excerpts below — what it does and how it makes money. "
                   "No outside knowledge, no figures you can't see in the text.")
            out = llm.complete(sys, excerpts, max_tokens=180)
            # same voice boundary as the engine: every number in the blurb must
            # appear in the excerpts it was written from, else show the excerpt
            if voice.verify_prose(out, [c for c, _ in hits],
                                  [c["page_start"] for c, _ in hits]) == "OK":
                return out
        except Exception:
            pass
    top = hits[0][0]["text"].strip()
    return top[:280] + ("…" if len(top) > 280 else "")


def _period_label(cite: dict) -> str:
    p = cite.get("period", "") or ""
    end = cite.get("period_end") or ""          # exact date, e.g. 2025-12-31
    if p.startswith("FY"):
        base = f"FY {p[2:]}"
        return f"{base} (ended {end})" if end else base     # the exact as-of date
    if "Q" in p and p[:4].isdigit():
        base = f"Q{p.split('Q')[1]} FY{p[:4]}"
        return f"{base} (ended {end})" if end else base
    if p.startswith("ASOF"):
        return f"as of {p[4:]}"
    if "YTD" in p:
        yr, rest = p[:4], p[4:]
        base = f"{rest.replace('YTD','').replace('M','')}-mo YTD FY{yr}"
        return f"{base} (ended {end})" if end else base
    return p or "period"


_SCALE_WORD = {"thousands": "thousand", "millions": "million", "billions": "billion"}


def _figure(cite: dict) -> str:
    cell, scale = cite.get("cell", ""), cite.get("scale", "")
    if scale == "units":
        return f"${cell}"
    return f"${cell} {_SCALE_WORD.get(scale, scale)}".strip()


def render_answer(ans) -> None:
    cls, label = _STATUS.get(ans.status, ("d71-refused", ans.status))
    conf = max(0.0, min(1.0, float(ans.confidence or 0.0)))
    band = "high" if conf >= 0.8 else "moderate" if conf >= 0.5 else "low"
    cits = list(ans.trace.citations or [])

    # 1) THE ANSWER, front and centre — hero figure first, then the sentence.
    if ans.status == "answered":
        try:
            headline = getattr(ans, "headline", None)
            if headline:                                     # core-supplied (e.g. FCF)
                st.metric(label=headline[0], value=headline[1])
            elif ans.value is not None and cits:
                c0 = cits[0]
                same_label = (len(cits) >= 2 and c0.get("label")
                              and c0.get("label") == cits[-1].get("label"))
                if same_label:                               # growth: old -> new
                    st.metric(label=c0.get("label", "Metric"),
                              value=_figure(cits[-1]), delta=f"{ans.value:+.1f}%")
                elif "%" in ans.text:                        # margin
                    st.metric(label="Margin", value=f"{ans.value:.1f}%")
                elif c0.get("cell"):                         # level figure
                    st.metric(label=_period_label(c0), value=_figure(c0))
        except Exception:
            pass                                             # never let the hero crash the card

    # Render the sentence as MARKDOWN (so bold renders and '\$' shows as '$');
    # no raw-HTML wrapper, which would print markdown/escapes literally.
    if ans.status == "answered":
        st.markdown(_md(ans.text))
    elif ans.status == "hedged":
        st.warning(_md(ans.text))
    else:
        st.info(_md(ans.text))

    # 2) EVERYTHING ELSE — status, confidence, sources, reasoning, passages —
    #    tucked into a single collapsed dropdown so only the answer is visible.
    #    (No st.popover here: Streamlit forbids popovers inside an expander, so
    #    sources are rendered as a plain list with their verbatim cells.)
    steps = list(ans.trace.steps or [])
    verifiable = [c for c in cits if c.get("cell") or c.get("snippet")]
    if not (verifiable or steps or ans.passages):
        return                           # nothing to disclose (e.g. a prompt note)
    with st.expander("Details — confidence, sources & reasoning", expanded=False):
        st.markdown(f'<span class="d71-pill {cls}">{label}</span>'
                    f' <span class="d71-sub">· confidence {band} ({conf:.0%})</span>',
                    unsafe_allow_html=True)

        if verifiable:
            st.markdown("**Sources** — verbatim from the filing")
            for c in verifiable:
                st.markdown(f"- {c.get('form') or '10-K'} · p.{c.get('page','?')} · "
                            f"**{c.get('label','')}** · {_period_label(c)}")
                if c.get("cell"):
                    st.code(c["cell"], language=None)
                elif c.get("snippet"):
                    st.caption(_md(c["snippet"]))

        if steps:
            st.markdown("**Reasoning**")
            for s in steps:
                st.markdown(f"- {_md(s)}")

        if ans.passages:
            st.markdown("**Source passages**")
            for p in ans.passages:
                st.markdown(f"p.{p.get('page_start','?')} · "
                            f"{p.get('section_name','')}")
                st.caption(_md(p.get("text", "")[:500]))
                st.divider()


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
facts, _prose, companies = load_bank()   # don't shadow the `prose` module
# --- spending guard (public deployment) ------------------------------------- #
# A per-session question cap plus a process-wide hourly budget on LLM-voiced
# questions. Past either, the session keeps working in the deterministic voice
# (same grounded numbers and refusals, template wording) — never blocked.
import collections
import os
import time

_SESSION_CAP = int(os.environ.get("MAX_LLM_QUESTIONS_PER_SESSION", "40"))
_HOURLY_CAP = int(os.environ.get("MAX_LLM_QUESTIONS_PER_HOUR", "300"))


@st.cache_resource
def _llm_budget():
    return {"lock": __import__("threading").Lock(), "times": collections.deque()}


def _spend_ok() -> bool:
    """Reserve one LLM-voiced question for this session if the caps allow."""
    if st.session_state.get("llm_used", 0) >= _SESSION_CAP:
        return False
    b = _llm_budget()
    with b["lock"]:
        now = time.time()
        while b["times"] and now - b["times"][0] > 3600:
            b["times"].popleft()
        if len(b["times"]) >= _HOURLY_CAP:
            return False
        b["times"].append(now)
    st.session_state.llm_used = st.session_state.get("llm_used", 0) + 1
    return True


llm.force_offline(False)
online = llm.available()
tickers = [c[0] for c in companies]
name_by_tk = {c[0]: c[1] for c in companies}
cik_by_tk = {c[0]: c[2] for c in companies}

if "history" not in st.session_state:
    st.session_state.history = []
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = None        # blank until named / picked


def _on_pick():                                   # sidebar switch -> new anchor
    st.session_state.current_ticker = st.session_state.picker


# If a question switched the company, reflect it in the sidebar picker. Done
# BEFORE the widget is created (setting a widget key after instantiation is
# illegal) and only on a question-driven switch, so a manual pick isn't clobbered.
if st.session_state.pop("sync_picker", False):
    st.session_state.picker = st.session_state.current_ticker


examples = [
    "What was revenue in 2025?",
    "Operating cash flow in 2025?",
    "How much cash at the end of 2025?",
    "What are the main risk factors?",
]

with st.sidebar:
    st.header("Filing Q&A")
    st.caption("Grounded, traceable answers from 10-K / 10-Q filings.")
    st.selectbox("Company", tickers, key="picker", index=None,
                 placeholder="Choose a company…", on_change=_on_pick,
                 format_func=lambda t: f"{t} — {name_by_tk[t]}")
    _cur = st.session_state.current_ticker
    if _cur:
        summary = company_summary(_cur, cik_by_tk[_cur], online)
        if summary:
            st.markdown(f"**{name_by_tk[_cur]} ({_cur})**")
            st.caption(_md(summary))
            st.caption("— business model, from its SEC filing")
    else:
        st.caption("Name any banked company in your question (e.g. 'Nvidia', "
                   "'GOOG'), or pick one above. Follow-ups stay on the current "
                   "company until you name another.")
    st.divider()
    st.markdown(
        f'<span class="d71-pill {"d71-answered" if online else "d71-refused"}">'
        f'{"live model" if online else "offline (deterministic)"}</span>',
        unsafe_allow_html=True)
    st.caption(f"{len(companies)} companies · {len(facts):,} grounded facts")
    st.divider()
    st.caption("Try:")
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

st.title("SEC Filing Q&A")
st.caption("Every number is grounded to a filing page and verbatim cell — the "
           "system refuses rather than guess.")
cur = st.session_state.current_ticker
if cur:
    st.markdown(f'<span class="d71-active">Active company —</span> '
                f'**{cur} · {name_by_tk[cur]}**', unsafe_allow_html=True)
else:
    st.markdown('<span class="d71-active">No company selected — name one in your '
                "question (e.g. “Nvidia”, “GOOG”) or pick it on the left.</span>",
                unsafe_allow_html=True)
st.divider()

# transcript — professional report style (no chat avatars)
for turn in st.session_state.history:
    st.markdown(f'<div class="d71-q"><span class="tk">{turn["ticker"]}</span> · '
                f'{_md(turn["q"])}</div>', unsafe_allow_html=True)
    with st.container(border=True):
        render_answer(turn["answer"])

prompt = st.chat_input("Ask about any banked company — by name or ticker…")
prompt = st.session_state.pop("pending", None) or prompt
if prompt and online and not _spend_ok():
    llm.force_offline(True)          # this session, this run: deterministic voice
    online = False
    st.caption("Usage limit reached for the conversational voice — answers continue "
               "in the deterministic mode (same grounded figures).")
if prompt:
    # anchor on the company named in the question, else stay on the current one
    used = resolve_from_question(prompt, companies, st.session_state.current_ticker)
    # "which company?" was asked last turn and the reply is just a name ("Tesla"):
    # answer the question the user was waiting on, not the bare name
    pending_q = st.session_state.pop("pending_q", None)
    if used and pending_q and len(prompt.split()) <= 3:
        prompt = f"{pending_q} ({prompt.strip()})"
    if not used:
        # No company named or selected. Hand it to the engine with no entity: its
        # CONVERSE lane handles small talk / capability / advice naturally, and a
        # factual question with no company gets a warm "which company?" — one
        # source of truth for conversation, not a second copy of the logic here.
        with st.spinner("…"):
            note = answer_for(prompt, None, None, None, online)
        if any(s_ == "lane: need_company" for s_ in note.trace.steps):
            st.session_state.pending_q = prompt
        st.session_state.history.append({"ticker": "—", "q": prompt, "answer": note})
        st.rerun()
    if used != st.session_state.current_ticker:
        st.session_state.current_ticker = used
        st.session_state.sync_picker = True       # update the sidebar on next run
    with st.spinner("Tracing the answer to the filing…"):
        ans = answer_for(prompt, used, name_by_tk[used], cik_by_tk[used], online,
                         context=st.session_state.get("last_query"))
    # remember the last FACTUAL question so a follow-up ("what about 2024?",
    # "the growth for the entire period?") can inherit its metric and periods
    q_used = getattr(ans, "query", None)
    if q_used is not None and q_used.metrics and ans.status != "refused":
        st.session_state.last_query = q_used
    st.session_state.history.append({"ticker": used, "q": prompt, "answer": ans})
    st.rerun()
