# Design reasoning (succinct)

Each choice below maps to a failure of the prior system (hallucination, annual/quarterly mixing, GAAP confusion, no traceability) and to the fundamentals the grader is listening for.

### Why the LLM / code split
Mirrors the **FinQA** paradigm: it annotates deterministic **"reasoning programs"** for full explainability, and finds that large pretrained models **fall far short of experts at multi-step numerical reasoning**. So:
- **Code** owns: PDF cell extraction, exact value lookup, **arithmetic**, period alignment, validation, control flow.
- **LLM** owns: parsing the messy question into a structured query, conversing (small talk, fenced to assert no fact), reading/summarizing prose, and phrasing the answer over code-grounded numbers. (It does **not** touch extraction in the prototype.)
Arithmetic goes to code (cf. **PAL**, program-aided language models); LLM arithmetic degrades with digit count and silently errs on large figures.

### Why no heavyweight agent framework (LangGraph / CrewAI / AutoGen)
The LLM stays behind a thin, owned wrapper (`llm.py`); orchestration is plain code (`core.py`). We deliberately **do not** adopt an agent framework. The distinction: **agentic *architecture*** (code drives a tool pipeline the LLM only proposes into — what we have) vs **agentic *framework*** (the model drives control flow — CrewAI/AutoGen). The latter re-introduces the exact failure we were hired to kill: if the model decides what runs, it can skip the grounding check, invent an intermediate figure, or route around the gate. Reasons, in order:
- **Control-flow is the failure class.** Hallucinated/mislabeled numbers come from the model being *in the loop*. Our value is pulling control flow + arithmetic *out* of it behind a one-way gate. A model-routed framework buys the bug back.
- **Determinism = auditability.** A number may need reproducing months later; deterministic code + grounded facts is reproducible with an explainable trace. Model-routed orchestration is nondeterministic (same question → different path → possibly different answer) through prompts you don't fully see.
- **Abstraction tax > our complexity.** Frameworks pay off for open-ended, many-tool orchestration; ours is a short deterministic router + lookup + operator layer. A graph lib would be *more* glue to say *less* than a plain function.
- **Opacity kills the one thing we sell.** This project's bugs were all found by reading a thin traceback + grounded facts. Deep framework call-trees and hidden prompt mutation make "why did it say X?" unanswerable.
- **Weight / churn / lock-in.** Large, fast-moving dependency trees on a lightweight, trust-sensitive target; and a framework wants to own the loop (it calls you) — backwards for a system we must own and audit. Our 225-test suite exists *because* components are plain deterministic functions.

**Expansion path (framework-free):** when the model genuinely needs to *plan over tools* (multi-hop comparisons across many filings), promote `core.answer` to Anthropic's **native tool-use loop** — expose `resolve_entity` / `find_fact` / `compute` / `prose_search` as tools the LLM sequences — keeping the two invariants: **every number stays deterministic, every answer passes the gate.** Same SDK, no new dependency. A framework is warranted only for genuinely open-ended, non-enumerable tasks — which a narrow, auditable financial-QA pipeline is not.

**Guardrail:** orchestration may be model-assisted, but **grounding, arithmetic, and the answer/refuse gate stay in code we own.**

### Why layered facts→operators→aggregates→narration (complexity without losing trust)
Complex questions ("3-year revenue growth", "is the margin trend improving?") tempt the obvious shortcut: hand the LLM the raw numbers and let it reason. That shortcut *is* the prior system's failure — one model subtraction reopens hallucination, period mixing, GAAP blur. So flexibility is delivered by a **stack of deterministic layers that compose**, not by loosening the boundary:
- **L0 facts → L1 operators → L2 aggregates** are all code; **L3 narration** is the only LLM step, and it introduces no number.
- **Closure is the lever:** an L1 operator's output is a valid input to an L2 one (`series(margin(op_income, revenue), years)`), so arbitrary-looking questions decompose into a *small, closed* operator set. The LLM only parses the question into the composition; code executes it.
- Each operator carries its inputs' **citations forward** and enforces the guards as **preconditions** (scale, `period_kind`, GAAP) — so a derived number is only as trustworthy as its chain to L0, and a broken chain **refuses**.
- This is the same logic as the no-framework decision, applied to *data flow* instead of *control flow*: keep the enumerable, auditable set in code; the operators are exactly the tools the native tool-use path would later expose. Full layering: *Layers of information* in `architecture.md`.

### Why a conversational router (small talk without weakening trust)
A trust-first tool still has to hold a natural conversation, but a greeting must not open a door for a fabricated number. The resolution is to **route by intent and make grounding rigor proportional to the factual weight of the reply**, so small talk and rigor never compete:
- **Converse** (small talk / capability): the LLM writes freely, fenced to a persona that may describe the assistant and its scope but **may not state any figure or fact** — and the generator falls back to a template if it ever emits a `$`-figure. Zero grounding is needed because zero facts are asserted.
- **Overview / meta**: grounded in banked metadata (coverage + which sections exist); deliberately **not** a synthesized business description, since we index the statements + Risk Factors + MD&A, not Item 1 "Business" — passing a risk-factor snippet off as an overview would mislead.
- **Factual**: the full deterministic spine.
The elegance is one principle doing double duty: **the LLM writes language in both the Converse and Factual lanes, but a lane-specific leash fixes what it may say** — persona-only vs. code-grounded-numbers-only. The invariant *no number or fact originates from the LLM* holds across both; this is the same discipline as the no-framework and layered-operator decisions, applied to the chat surface. Two boundaries decline rather than guess: investment **advice/opinion** and **off-scope** asks. Full lane table: *Conversational intent router* in `architecture.md`.

### Why vector retrieval only for prose, not numbers
Two distinct failure modes argue against similarity search for the number-finding path:
1. **Numeric-magnitude weakness** — embeddings capture some numeracy in-range but don't extrapolate to large numbers (Wallace et al. 2019).
2. **Topical collapse** — dense retrieval groups by semantic similarity, so **"Net income"** vs **"Net income attributable to common stockholders"** embed near-identically despite being different line items / dollar figures a few rows apart.
→ Numbers via **structured label + period + unit matching**; vector retrieval used **only** for risk-factor / MD&A prose.

**Retriever choice (prototype vs scale).** The prototype uses **TF-IDF cosine** rather than neural embeddings: it is dependency-light (no model download / no torch on a MacBook Air), fully **inspectable** (you can see which terms drove a match — itself a traceability win), and Anthropic ships no embeddings endpoint, so dense vectors would mean a second provider. **Dense embeddings are the drop-in upgrade at scale** — the `ProseIndex` interface doesn't change. Either way it's in-memory cosine at 3-filing scale; no vector DB.

**Prose grounding is softer than numeric — stated, not hidden.** Numbers are matched *verbatim*; prose is *paraphrased* by the LLM, leaving three cracks: overstatement, smuggled outside knowledge, and wrong-passage retrieval. Mitigations: narrate **only** over retrieved passages, ship the **page-cited source passages with every answer** (human verifies in one glance), and **two refusal layers** (retrieval refuses on zero overlap; the LLM refuses when passages don't answer).

### Extraction: deterministic-text now, hybrid/vision as the scale path (the real weak link)
**Built:** a deterministic extractor — pdfplumber **text** lines → the 3-gate
statement match → a **verbatim grounding check** on every cell (the printed token
must appear on its page, else the fact is rejected). No Camelot, no LLM/vision in
the prototype. This parses the clean consolidated statements across the 13 banked
filers well (~5,000 grounded facts, 100% grounded).
**Why it's still the weak link:** pdfplumber/Camelot reach only **~65–75% on
*complex* tables** and need per-document tuning; the literature's best practice is a
**pipeline + AI/vision fallback**. So the documented scale path is deterministic-
first → **LLM/vision on messy regions** → the same grounding check — and because the
grounding check is the gate, a vision fallback can never introduce an ungrounded
number. Messy/rotated/scanned tables are the **#1 thing that breaks at full scale**.

### Why parse-once upfront
At this scale, parse to a structured store once, then answer fast and traceably — and it matches the production shape (index the archive, don't re-interpret raw PDFs per query).

### Why we identify statements by line items, not titles
The SEC prescribes the **narrative item captions** (Item 1A, Item 7) but **not** the financial-statement titles — Reg S-X fixes the **line-item captions inside** (Rules 5-02/5-03) instead. So titles vary (Operations/Income/Earnings/Financial Position) while the rows are standardized. We therefore match a statement with **3 gates — title synonym + a period row + a ≥2 line-item signature** — and treat the line items as the real fingerprint. The payoff is fail-safe behavior: a region that isn't clearly a statement is **dropped, not misread**, and an unmatched statement becomes an honest "not in the filing," never a fabricated number.

### Trust mechanisms
- **Grounding** — every number must appear verbatim in the source filing.
- **Multi-match disambiguation** — look-alike rows surface a question, never a silent pick.
- **Sanity checks** — unit scaling, sign, period; period comparisons never cross `period_kind`s.
- **Confidence gate** — answer / hedge / refuse; a correct "unsure because…" beats a polished hallucination.
- **Prose grounding** — qualitative answers are narrated only over retrieved filing passages and ship with their page-cited sources; section-scoped, with two refusal layers for off-topic questions.
- **Conversational leash** — the small-talk lane asserts no filing fact, and its generator falls back to a template if it ever emits a `$`-figure, so chit-chat can't leak a number.
- **Regression suite** — 225 tests (hermetic fixtures + real-bank checks), plus a hand-verified benchmark with a tolerance band. (A sparing EDGAR **XBRL spot-check** at ingestion is a documented validation hook, not yet built.)

### Audit-driven refinements (10-agent adversarial pass)
A fan-out of 10 adversarial agents ran ~2,000 realistic cases against the engine and surfaced 35 reproduced issues; fixing them sharpened several design choices (regression-pinned in `tests/test_audit_fixes.py`):
- **Fiscal periods are relative to the filer's fiscal year-end, never the calendar.** Fiscal-quarter numbering and YTD length are derived from the filing's own period-end + duration (Disney's quarter ending June is fiscal **Q3**, not calendar Q2; its nine-month YTD is 3 fiscal quarters), and every period label in an answer is read from the **matched fact** (its `months` / `fiscal_quarter`), not reconstructed from the parsed query — so a nine-month figure is never phrased "six months," and an interim figure is never printed as a bare year.
- **Operations and cash-flow statements are duration statements, never instant.** A missed duration keyword on a 10-K income statement is promoted to the fiscal year (Bloom Energy printed dates but no "year ended" caption, which had hidden its entire annual income statement).
- **Derived-measure labeling is universal.** EBITDA / FCF carry the non-GAAP label in *every* path (single, multi, growth, margin) — a computed estimate must never read as a grounded line item. The non-GAAP guard separates an explicit **adjustment modifier** ("adjusted EBITDA" → refuse: that's management's bespoke variant) from a bare **"non-GAAP" mention** ("why is FCF non-GAAP?" → answer, labeled), with hyphens normalized.
- **Growth is honest about its edge cases.** A zero base yields no percentage (not `nan%`); a **negative** base is flagged (a widening loss is not "+242% growth"); explicitly **mixed period kinds** (a quarter vs a full year) are refused; CAGR is annualized, not the cumulative change; and a multi-metric growth request answers every metric.
- **Verbatim grounding extends to precision.** The displayed figure matches the filing's **own decimals** (a filer printing "$10,229.9 million" is shown exactly that), so the stated number is literally in the cited cell and a stated growth % reconciles with the stated inputs.
- **Intent is validated, not silently downgraded.** An unavailable year, an out-of-range quarter (Q5/Q0), or a trailing-twelve-month request **hedges** instead of quietly returning the annual figure; "per share" forces an EPS metric; a strong qualitative cue (risk / MD&A) routes to prose even when a metric word appears; prose is **section-scoped** and refuses when the requested section isn't banked rather than substituting another.
- **Where a gate is inherently imperfect, say so.** Lexical (TF-IDF) prose retrieval cannot fully separate a high-IDF off-topic token from an on-topic one; the floor was raised and term-coverage added, but the **LLM narrator is the semantic backstop** (it refuses when the excerpts don't answer) — consistent with the stated prose-grounding weakness above.

### "Correct" means (for the benchmark)
Right **line item** + right **period** + value within a small **tolerance band** (rounding/units). Right value but wrong period, or a near-miss line item, counts as **wrong**.

### Honest weaknesses
Table extraction on messy layouts; label disambiguation; restated / amended filings; period-alignment edge cases; **prose answers are paraphrased (softer grounding than verbatim numbers), and lexical retrieval can miss a paraphrased query a dense embedder would catch**; and scale — storage, parsing throughput, indexing strategy, and cost at tens of thousands of filings.

**On generalization:** the parser is hardened across **13 banked companies** (varied fiscal-year ends, interim 10-Qs, decimal-millions filers, footnote superscripts) and was stress-tested against **20 further un-banked filers** via an isolated ingest harness. It handles case-insensitive titles + synonyms + the "Condensed" prefix, scale from the parenthetical, fiscal-relative periods, and the annual/quarterly guard. What remains the documented **scale path** (not a demo claim): scanned-PDF OCR, Notes/segment/non-GAAP reconciliation tables, bank/insurer statement layouts that don't fit the standard three-statement shape, 52/53-week per-column fiscal dates, and foreign forms (20-F/40-F). See *Filing structure* in `architecture.md`.
