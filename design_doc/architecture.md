# Architecture — SEC-filing NL Q&A prototype

## Core principle
**The LLM handles language; deterministic code handles numbers; every number is grounded to the source filing before it can appear in an answer.**

We never poll the LLM directly with the PDF to "get the number" — that is exactly the prior system's failure mode (hallucinated values, period mixing, GAAP confusion, no traceability).

## Diagram

```
  LEGEND:   LLM = language only        CODE = all numbers


  INGEST  (once per filing → writes to THE BANK)

    company not in the Bank?
         │
         ▼
    ┌──────────────────────┐   ┌───────────────────┐   ┌───────────────────┐
    │ CODE  EDGAR discover │──▶│ fetch primary doc │──▶│ render HTML → PDF │
    │ (sparing: which      │   │ (the filing's own │   │ (headless Chrome) │
    │  filings exist)      │   │  .htm, NOT XBRL)  │   │                   │
    └──────────────────────┘   └───────────────────┘   └─────────┬─────────┘
                                                                  ▼
                            ┌───────────────────────────────┐
                            │ CODE  EXTRACT (deterministic) │
                            │  pdfplumber TEXT + 3-gate      │
                            │  statement match               │
                            │  → grounding: num in source    │
                            │  (LLM/vision fallback = scale   │
                            │   path, not built)             │
                            └───────────────┬───────────────┘
                                            ▼
          ╔════════════════════════════════════════════════════════╗
          ║            THE BANK   (system of record)                ║
          ║  FACT STORE   value · period · unit · GAAP-flag         ║
          ║               + PROVENANCE (filing · page · raw cell)   ║
          ║  PROSE STORE  chunks + retrieval index (risk / MD&A)    ║
          ╚════════════════════════════════════════════════════════╝


  ANSWER  (per question — reads ONLY from THE BANK)

        ┌───────────────────────────────┐
        │            question           │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ LLM   UNDERSTAND              │
        │ Q → {company, metric, periods}│
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────────────────────┐
        │ CODE  ROUTE (conversational intent)            │
        │  small talk / capability → CONVERSE (LLM, no   │
        │                            fact asserted)      │
        │  advice / off-scope       → DECLINE            │
        │  overview / coverage      → grounded meta      │
        │  a figure / claim         → the spine ↓        │
        └───────────────┬───────────────────────────────┘
                        ▼  (factual lane only)
        ┌───────────────────────────────┐
        │ CODE  RESOLVE ENTITY          │
        │ company → CIK (static SEC map)│
        │  • ambiguous? → disambiguate  │
        │  • not in Bank? → honest miss │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ CODE  FIND THE NUMBER         │◀──┐
        │ query THE BANK (fact store)   │   │ retry: next label
        │  • must be in the filing      │   │ synonym (deterministic)
        │  • 2 look-alikes? → ask        │───┘
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ CODE  COMPUTE (operator layer)│
        │ growth · series · margin ·    │
        │ EBITDA · FCF · ROE · CAGR …   │
        │  (never the LLM)              │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ CODE  CONFIDENCE GATE         │
        │ answer │ hedge │ refuse        │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ LLM   WRITE THE ANSWER        │
        │ phrase over the found numbers │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ ANSWER + TRACE                │
        │ filing · page · raw numbers   │
        │ + confidence ("unsure if…")   │
        └───────────────────────────────┘


  One glance:

   ┌────────────────┐   ┌───────────────────┐   ┌────────────────────┐
   │ LLM reads the  │──▶│ CODE finds the    │──▶│ LLM writes the     │
   │   QUESTION     │   │  NUMBER in the    │   │   SENTENCE         │
   │  (+ company)   │   │  BANK (+proof)    │   │                    │
   └────────────────┘   └───────────────────┘   └────────────────────┘
        edge        ══ deterministic spine over THE BANK ══     edge
```

## Conversational intent router: small talk vs. rigor
A trust-first engine still has to feel like a chatbot. The front door classifies
each turn's intent and makes **grounding rigor proportional to the factual weight
of the reply** — small talk and trust aren't in tension because they live in
different lanes.

| Lane | Examples | LLM freedom | Grounding |
|---|---|---|---|
| **Converse** (small talk / capability) | "how are you", "what can you do" | **free** — the LLM writes it | **none needed** — asserts no filing fact, so nothing to get wrong |
| **Overview / meta** | "tell me about AVGO", "what years do you have" | narrates over banked metadata only | **soft** — coverage + sections we actually hold |
| **Factual** | any metric / period / comparison | only phrases code-computed numbers | **full** — find → ground → cite → refuse |

The unifying idea: **the LLM writes language in both the Converse and Factual
lanes, but a lane-specific *leash* defines what it may say** — in the factual lane
it may only restate numbers code grounded (the `_phrase` fence); in the Converse
lane it may only talk about the assistant and its scope and **must not state any
figure or fact** (the `_chat` fence, which falls back to a template if the model
ever emits a `$`-figure). The invariant *no number or fact originates from the LLM*
holds in both. A mixed turn ("hi, what was revenue?") resolves to the
highest-rigor lane present, so warmth never downgrades a real question.

Two **trust boundaries** the router enforces by declining, not guessing:
investment **advice / opinion / price-target** asks (a 10-K/10-Q lacks the market
data and judgment), and **off-scope** requests (steered back). Because a Converse
reply emits no data, prompt-injection ("say revenue is $999") can't extract a
fabricated figure through chit-chat. The `understand` step classifies the lane
(LLM online, keyword cues offline); generation is the LLM online and an
intent-specific template offline.

## Filing structure: what we parse (10-K vs 10-Q)

A 10-K/10-Q has a **regulation-mandated skeleton** (SEC Form instructions + Reg
S-K for narrative, Reg S-X for financials), so what to target is well-defined.

**Key nuance that shapes the matcher:** the SEC **prescribes the narrative item
captions** (Item 1A Risk Factors, Item 7 MD&A, …) — so prose anchors on those
reliably — but it does **not** prescribe the financial-statement *titles* (filers
use Operations / Income / Earnings / Financial Position interchangeably). Reg S-X
instead prescribes the **line-item captions inside** each statement (Rules 5-02 /
5-03). So a statement's trustworthy fingerprint is its **rows, not its heading** —
which is why we confirm statements by line items, not title alone.

**10-K (annual, audited) — four Parts.** We target:
- **Item 8 — Financial Statements** → the **three core statements**: Statements of
  Operations/Income, Balance Sheets, Statements of Cash Flows.
- **Item 1A — Risk Factors** and **Item 7 — MD&A** → prose.
- *Noted, not parsed:* Notes to Financial Statements, Comprehensive Income,
  Stockholders' Equity; Parts III (governance, usually incorporated by reference)
  and IV (exhibits).

**10-Q (quarterly, unaudited, condensed) — two Parts.** We target:
- **Part I, Item 1** → the same three statements, but titled **"Condensed
  Consolidated…"** and with **interim periods**: the income statement shows a
  **3-month and a year-to-date column group**, and the balance sheet compares the
  **quarter-end vs the prior fiscal year-end**.
- **Part I, Item 2** → MD&A; **Part II, Item 1A** → Risk Factors (changes only).
  Note the item numbering differs from the 10-K.

**How we identify each of the 3 statements — 3 gates, all required (else refuse):**
1. **Title synonym** — case-insensitive; optional `Condensed` / `Consolidated`;
   Operations | Income | Earnings; Balance Sheets | Financial Position / Condition;
   Cash Flows. (A fast candidate, not proof.)
2. **Period row nearby** — a years/dates line must follow, proving it's the real
   statement and not a table-of-contents / index entry.
3. **Line-item signature** — the region must carry ≥2 of that statement's Reg S-X
   captions (cash flow → operating / investing / financing activities; income →
   revenues / cost of revenues; balance → Total assets / Total liabilities).

A *false* region would have to pass all three — near-impossible. A statement we
**can't** match yields **no facts and is recorded missing**, so a question about it
is **refused, never guessed**: the worst case is a safe miss, not a fabricated
value. (The LLM only phrases numbers code already found; nothing found → nothing
phrased.)

Two formatting reads sit on top:
- **Scale** — from the `(in thousands | millions | billions)` parenthetical, never
  assumed (a "(in thousands)" filer would otherwise be 1000× off).
- **Periods** — every fact tagged `period_kind ∈ {annual, quarter, ytd, instant}`
  with its own `period_end`; **comparisons across `period_kind`s are refused** —
  the concrete mechanism against the prior system's annual/quarterly mixing.
  **Fiscal, not calendar:** quarter numbering and YTD length are computed relative
  to the filer's own fiscal year-end (Disney ends in September, so its June quarter
  is fiscal **Q3**), and the period label shown in an answer is read from the
  **matched fact**, never reconstructed from the parsed question — so a nine-month
  figure is never phrased "six months," and an interim figure is never printed as a
  bare year. Operations/cash-flow statements are duration statements and are never
  stored as `instant`.

**Known limits (explicitly out of scope, documented not built):** scanned / image
PDFs (need OCR); Notes / segment / non-GAAP reconciliation tables; 52/53-week
per-column fiscal dates beyond the demo filer; foreign forms (20-F / 40-F);
amendments (10-K/A); full 10-Q prose. These are the generalization/scale path, not
the demo's claim.

## Entity resolution (company → CIK)
The question names a company; the system must map it to an exact SEC entity. On-principle split:
- **LLM extracts the company *mention*** ("Tesla", "the EV maker", "TSLA") — language/NER.
- **Code resolves mention → CIK** via SEC's official **`company_tickers.json`** (ticker/name → CIK), a **static file downloaded once and cached** (≈0 cost against the EDGAR budget). **The LLM never emits a CIK** — it would hallucinate one or hit the wrong entity (subsidiaries, former names, share classes like GOOG/GOOGL).
- **Ambiguity → flag/disambiguate**, never a silent pick. Resolved CIK → check the Bank; on a miss it drives the ingestion event.

## The Bank (Fact Store + Prose Store) — the system of record
The Bank is the **hinge** between *expensive, one-time, messy-PDF ingestion* and *fast, deterministic query*. Ingestion writes grounded, provenance-tagged facts into it **once**; the agent only ever **reads** from it. This is what makes answers fast, traceable, and scalable — and why a cache miss is an **ingestion event, not a live scrape**.

**Fact record (one row = one reported value) — actual fields:**
```
  company, cik, ticker
  filing:      filing_stem, form(10-K/10-Q), accession_no
  statement:   operations | balance_sheet | cash_flows
  label:       raw_label (verbatim)              # matched by synonym at query time;
                                                 # no canonical key is stored
  period:      period (FY2025 | 2025Q3 | 2025YTD9M | ASOF…),
               period_kind(annual|quarter|ytd|instant), period_end,
               fiscal_year, fiscal_quarter, months
  value:       value, unit(USD|shares), scale(thousands|millions|billions|units),
               is_negative, is_dash, gaap_flag
  PROVENANCE:  page_no, raw_cell_text, section, source_line   ← verifiability
  EXTRACTION:  extraction_method(pdfplumber_text), grounding_ok(bool), confidence
```
`raw_label` + `page_no` + `raw_cell_text` + `grounding_ok` + `period` are the fields
that let a skeptic verify a number and that remove the four historical failures at
the data layer. (Restatement handling — add a new row, mark the old superseded —
is a documented scale-path field, not exercised by the demo corpus.)

**Prose record (one row = one narrative passage):** `company, cik, filing, form, section(risk_factors|mdna), section_name, chunk_text, page_start, page_end`. The retrieval index (TF-IDF vectors here; dense embeddings at scale) is built over these chunks — in the prototype it is rebuilt in memory on load; at scale it is persisted alongside the chunks.

**Ingestion contract:**
- A filing enters the Bank only after **extract → grounding check → confidence score**.
- Low-confidence rows are **flagged, not stored as fact**.
- Filings are **immutable** (keyed by `accession_no` / `source_doc_hash`); amendments/restatements **add a new row and mark the old `superseded_by`** — never overwrite.

**Prototype vs scale (same schema, different engine):**
- **Prototype:** plain JSON on disk (`store/facts.json`, `store/prose.json`) read into memory — ~5,000 facts across 13 companies, trivial on a laptop and diff-able by eye.
- **At tens of thousands of filings:** a columnar store (DuckDB/Parquet) partitioned by company/period + an index; ingestion becomes a **batch pipeline**; the **query layer is unchanged** because it only reads the Bank.

## Prose answers (the prose twin of the numeric spine)
Qualitative questions ("main risks?", "what does management say about margins?") are answered from the narrative sections — **Item 1A Risk Factors** and **Item 7 MD&A** — parsed from the *same rendered PDF*, not from tables and not from the model's memory.

```
  question → CODE RETRIEVE (rank filing passages) → LLM NARRATE (only over the
  retrieved passages, cite the page) → ANSWER + its source passages, or REFUSE
```

- **Retrieve (code):** vector-space similarity over prose chunks — **TF-IDF cosine** in the prototype (dependency-light and inspectable: you can see which terms matched), with **dense embeddings** as the drop-in upgrade at scale (same interface). We deliberately **never embed numbers** — embeddings are magnitude-blind and topically collapse ("Net income" vs "Net income attributable to common stockholders" sit almost on top of each other). Vectors for prose only.
- **Narrate (LLM, fenced):** the model writes the answer **only over the retrieved passages** and cites the page; if nothing relevant is retrieved it **refuses** rather than drawing on training knowledge.
- **Grounding is softer here, and we say so.** Numbers are matched *verbatim*; prose is *paraphrased*, so three cracks remain — overstatement, smuggled outside knowledge, and wrong-passage retrieval. Mitigation: every prose answer ships **with its page-cited source passages** so a human verifies in one glance — never narration without its sources. Two refusal layers back this up: retrieval refuses on zero term-overlap, and the LLM refuses when the retrieved passages don't actually answer the question.

## The boundary (the key design choice)
- **LLM at the input/output edges only** — understand the question, extract the company mention, converse (small talk), phrase the answer, summarize prose. Extraction in the prototype is **fully deterministic** (pdfplumber text + grounding); an LLM/vision step *into* extraction is the documented scale path for messy tables, not built.
- **Deterministic numeric spine** — resolve company→CIK, parse PDF cells, match label+period+unit, arithmetic, validation, and the control flow.
- **Validation gate = a one-way valve:** fuzzy in, verified out. Nothing the LLM produces (company, label, number, sentence) becomes an answer until it passes the gate (structured-schema constraint + grounding + CIK-from-static-map).
- **Control flow:** the LLM *proposes* the plan (decompose the question into sub-lookups); **code executes** routing / retry / the answer-or-refuse gate — so the loop is predictable and debuggable.

## Layers of information: facts → operators → aggregates → narration
The boundary above isn't a single wall — it's a **stack of layers**, and this is
the deliberate answer to "how do we handle *complex* questions without loosening
trust?" Flexibility comes from **operators composing**, never from handing the raw
numbers to the LLM to reason over.

```
  L3  NARRATION   LLM   phrase / summarize the finished figures   ── language edge
      ▲  carries labels (non-GAAP, period) through; descriptive, not causal
  ─── trust boundary (one-way: nothing below is ever the LLM's to compute) ───
  L2  AGGREGATES  CODE  series / trend / min·max·avg / CAGR
      ▲  output of L1 is a valid input here — this closure buys the flexibility
  L1  OPERATORS   CODE  growth(Δ,%) · ratio · margin · sum · diff · EBITDA · FCF
      ▲  each returns a derived figure that carries its inputs' citations forward
  L0  FACTS       CODE  THE BANK — one grounded value + provenance per row
```

- **L0 — grounded facts.** The Bank. Each number verbatim from the filing with
  citation, `period_kind`, `scale`, GAAP flag. The floor everything traces to.
- **L1 — operators over facts.** Pure deterministic functions (`COMPUTED`,
  `_answer_growth`): grounded inputs → a derived figure that **carries its inputs'
  citations forward**. Every operator enforces the guards as *preconditions* —
  same scale, same `period_kind` (the annual/quarterly guard), GAAP/non-GAAP not
  silently mixed.
- **L2 — aggregates / compositions.** The output of an L1 operator is a valid
  *input* to an L2 one — that **closure** is what lets complex questions decompose.
  A three-year growth question is `series(revenue, [2023,24,25])` → each YoY step;
  "operating-margin trend" is `series(margin(op_income, revenue), …)` → sign of the
  Δs. The LLM parses the question into this composition; **code executes it.**
- **L3 — narration.** The LLM phrases the finished figures (the `_phrase` step).
  It introduces **no number** not handed up from below, and stays **descriptive**
  (what the numbers show), never causal or predictive (which needs outside
  knowledge we don't have).

**Three invariants make the layers trustworthy** — the existing guards, promoted to
properties *every* operator must honor:
1. **Provenance closure.** A derived number is valid only if it can emit the full
   citation chain down to L0. A composition that can't → **refuse**.
2. **Type / period safety.** Operators refuse incompatible inputs (annual vs
   quarter, mismatched scale, GAAP vs non-GAAP) — a precondition, not a special case.
3. **Partiality propagates.** Any missing input makes the whole expression refuse,
   never fabricate.

**The anti-pattern we explicitly reject:** reading "flexible enough for complex
questions" as "give the LLM the raw layer and let it reason." The moment the model
does one subtraction, the trust model is gone (invented values, period mixing,
GAAP blur — the four historical failures). Flexibility is bounded by the
**operator set**, not by open-ended model reasoning. We therefore keep that set
**small and closed** (level, growth, series/trend, ratio, margin, sum, diff,
aggregate, CAGR), composable ~1–2 levels deep — you can only guard what you can
enumerate.

**How this meets the no-framework decision:** these deterministic operators *are*
the tools. Today a code router maps question → composition. When the question space
outgrows the router (multi-hop comparisons across many filings), the **same**
operators are exposed as Anthropic tools the LLM *sequences* — but the tools still
do the arithmetic. Same architecture, two scales. (See *Agentic workflow* and
`design_reasoning.md` → *Why no heavyweight agent framework*.)

## Control flow (deterministic router, LLM fenced to language)
A single pass, with **code** owning every decision and the LLM confined to the two
language steps:

```
understand (LLM) → ROUTE (code) → [converse | resolve-entity → find → compute] → gate → phrase (LLM)
```

- The **LLM** does two things only: parse the question into a structured query
  (`understand`), and phrase the finished reply (`_phrase` over grounded numbers,
  or `_chat` within the no-facts persona). Both have deterministic offline fallbacks.
- **Code** owns all control flow: the intent router, entity resolution, the
  label/period/scale lookup (with a deterministic alt-label retry), the operator
  layer, and the answer/hedge/refuse gate.

What makes it *agentic in architecture* (not a fixed script): it decomposes the
question, routes by intent, and **verifies-then-retries-or-refuses**. What it is
**not** is an agentic *framework* where the model drives control flow — that would
re-introduce the hallucination/traceability failure we exist to prevent.

**No heavyweight agent framework (LangGraph / CrewAI / AutoGen).** The LLM stays
behind a thin owned wrapper (`llm.py`); orchestration is plain code (`core.py`). The
framework-free expansion path is Anthropic's native tool-use loop over the existing
deterministic functions (`resolve_entity` / `find_fact` / `compute` / `prose_search`)
— the LLM would *sequence* them for multi-hop questions, but the functions still do
the arithmetic and every answer still passes the gate. Full rationale:
*"Why no heavyweight agent framework"* in `design_reasoning.md`.

## Ingestion & EDGAR usage
- The Bank is populated by an **ingestion helper**, not by query-time scraping.
- EDGAR is used **sparingly** at ingestion: the **static `company_tickers.json`** (name/ticker → CIK, downloaded once), discover *which* filings exist for a CIK, fetch the **primary filing document** (then parse it as a messy PDF — **not** the clean XBRL facts, which would dodge the hard problem), and a handful of **XBRL spot-checks** for validation.
- Query-time cache miss → surface honestly ("not in the Bank yet — ingest it?"), never an inline live fetch.
- On-demand live fetch **does not scale** (EDGAR rate limits) — which is the whole reason the bulk archive exists; production ingestion is a batch pipeline.
