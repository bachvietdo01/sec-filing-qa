# SEC-Filing Q&A — grounded, traceable, honest

A lightweight **agentic** prototype that answers natural-language questions about
SEC filings (10-K / 10-Q) sourced from **PDFs**, with answers that are grounded
in the source, traceable to a page, and honestly hedged when the system isn't
sure.

It is built for one job: **be trustworthy about financial fundamentals.** The
prior assistant at the client lost the finance team's trust by hallucinating
values, mixing annual and quarterly figures, confusing GAAP vs non-GAAP, and
offering no way to check an answer. This prototype is organized around a single
principle that removes all four:

> **The LLM handles language. Deterministic code handles every number. Every
> number must appear verbatim in the source filing before it can enter an
> answer.**

```
  LLM reads the QUESTION  ->  CODE finds the NUMBER (+proof)  ->  LLM writes the SENTENCE
                           == deterministic numeric spine over THE BANK ==
```

The LLM is never handed the PDF and asked for the number — that is precisely the
failure mode we're replacing.

**Live demo:** [huggingface.co/spaces/bachdo01/sec-filing-qa](https://huggingface.co/spaces/bachdo01/sec-filing-qa)
— 14 banked companies (TSLA GOOG NVDA AVGO DIS AAL HSY VRT SOFI ISRG BE ASTS CRWV KTOS).

### The voice boundary (`voice.py`)

Code decides every fact and **drafts every reply** — answers, refusals, hedges and
small talk. The LLM may only *rewrite* a draft for tone; one verifier checks every
rewrite and falls back to the draft on any failure:

- **numbers, both directions** — every figure in the rewrite must match the draft
  (same kind, value within the rounding of the digits shown) and none may be dropped;
- **direction** — "grew"/"fell" must agree with the signs in the draft;
- **no opinion or forecast** words the draft doesn't use;
- **protected caveats** (couldn't-ground lists, non-GAAP labels) are re-attached verbatim;
- **refusals still decline**, and converse/refusal rewrites may not agree with a
  user's premise or add new names, periods, products or places;
- **prose narration** must quote every figure verbatim from the retrieved excerpt.

The LLM's parse of the question is a suggestion checked against the text; when it
proposes a metric the user didn't name, the assistant asks instead of answering.
Without an API key the app runs the same deterministic drafts, with the same numbers.

---

## Quickstart

```bash
pip install -r requirements.txt

# Optional: enable the live LLM language steps (otherwise a deterministic
# fallback is used — the prototype runs fully offline without a key).
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env       # already gitignored

# 1. The walkthrough (primary artifact): build the Bank, ask questions,
#    see the trace behind every answer, run the benchmark.
jupyter notebook demo.ipynb

# 2. The objective trust check: hand-verified Q/A within a tolerance band.
python3 benchmark.py             # 8/8

# 3. The hermetic unit-test suite (no network, no live API).
python3 -m pytest tests/ -q      # 185 tests

# 4. The dashboard — a posh, trust-first Streamlit skin over core.answer.
streamlit run app.py
```

> **Running the dashboard:** use a clean virtualenv — `python3 -m venv .venv &&
> source .venv/bin/activate && pip install -r requirements.txt`. The repo's base
> conda env has a `pillow`/`streamlit` version conflict; a venv sidesteps it, and
> the app itself uses no image components so nothing triggers `pillow` at runtime.

The demo is scoped to **Tesla, FY2025 10-K** (≈160 facts + ≈190 prose passages).
The architecture is general; the scope is deliberately small so the trust
reasoning — not the data volume — is what's on display.

### A note on the LLM
The language steps default to Anthropic **Claude Haiku 4.5** (the cheapest model;
the design offloads all hard numeric work to code, so a light model suffices).
If no key/credits are configured, `understand` falls back to keyword parsing and
prose answers show the top retrieved passage verbatim — so nothing in this repo
requires a key to run or to evaluate.

---

## Why it's trustworthy (failure → mechanism)

Each historical failure is addressed at the data layer, not by prompting:

| Prior failure | Mechanism here |
|---|---|
| **Hallucinated values** | **Grounding gate** — a number is banked only if its printed cell appears verbatim on its source page; the Bank *refuses* to store an ungrounded fact. The LLM never emits a number. |
| **Mixing annual / quarterly** | Every fact carries an explicit **period** (`FY2025`, `ASOF2025-12-31`) and `period_type`; matching is period-exact, and growth/compare refuses across mismatched scales. |
| **GAAP vs non-GAAP / wrong subtotal** | Line items keyed by **exact label + section**; look-alikes are surfaced, not silently merged. "Net income" returns the figure *attributable to common stockholders* **and flags** the total "Net income" line. |
| **No traceability** | Every answer ships a **trace**: the reasoning steps, and a citation to `filing · page · raw cell` (prose answers ship their page-cited source passages). |

Two hazards worth calling out, both handled:

- **`Basic` / `Diluted` collision** — these labels appear twice in the income
  statement: once as EPS ($/share) and once as weighted-average share counts
  (millions). The extractor keeps them apart via the section header, so an EPS
  question can never return a share count.
- **Honest refusal** — asking for a value not in the Bank (e.g. net income in
  2019) or a question with no identifiable company returns a refusal, never a
  fabricated figure.

---

## How it works

Full diagram and record schemas: **[`design_doc/architecture.md`](design_doc/architecture.md)**.
The *why* behind each choice (with literature): **[`design_doc/design_reasoning.md`](design_doc/design_reasoning.md)**.

**The Bank** is the system of record and the hinge between one-time ingestion and
fast query:

- **Fact Store** — structured statement values, each with provenance (filing,
  page, raw cell), unit/scale, GAAP flag, and `grounding_ok`.
- **Prose Store** — Risk Factors (Item 1A) and MD&A (Item 7) passages with page
  provenance.

The answer path only ever **reads** the Bank:

- **Numeric spine** — `understand` (LLM → structured query) → `resolve_entity`
  (company → CIK via SEC's static `company_tickers.json`; the LLM never emits a
  CIK) → `find_number` (match label + period in the Bank, disambiguate
  look-alikes) → `compute` (growth/margin, in code) → **confidence gate**
  (answer / hedge / refuse) → phrase.
- **Prose twin** — retrieve relevant passages (TF-IDF cosine), narrate **only
  over them**, cite the page, refuse when nothing relevant is retrieved. Prose
  grounding is softer than numeric (paraphrase vs verbatim), so every prose
  answer ships with its source passages for one-glance verification.

---

## Agentic design

A single tool-using controller runs:

```
plan → resolve-entity → retrieve → extract → validate → (retry / fallback) → compute → answer
```

with a confidence gate that can **refuse**. The tools (`resolve-entity`,
`table-lookup`, `prose-search`, `compute`, `xbrl-spot-check`) are deterministic;
the controller decides *which and when*. What makes it agentic rather than a
fixed script: it **decomposes** the question, **chooses** tools, and
**verifies-then-retries-or-refuses**. It is intentionally **one controller, not a
multi-agent swarm** — the loop stays predictable and debuggable, which is itself
a trust property.

**The key design choice is the boundary:** the LLM sits at the input/output edges
(understand the question, phrase the answer, narrate prose) plus one fenced step
into extraction (interpret a messy label / structure a broken table region). The
numeric spine — parse, match, arithmetic, validation, control flow — is
deterministic. The validation gate is a one-way valve: fuzzy in, verified out.

---

## Data environment & EDGAR stance

Per the brief, **PDF is the primary source**; the live EDGAR API is a
**constrained, low-budget** resource, not the pipeline. Accordingly:

- Filings are treated as a **local archive** of PDFs. We render the EDGAR HTML to
  PDF on purpose — the hard problem is messy PDF tables, not HTML `<table>` tags;
  parsing a rendered PDF confronts the real layout problem.
- EDGAR is touched only at **ingestion**, sparingly: the **static**
  `company_tickers.json` (name/ticker → CIK, downloaded once and cached),
  one submissions lookup to discover which filings exist, the primary document
  fetch, and a handful of optional XBRL spot-checks for validation.
- A query-time cache miss is an **ingestion event**, surfaced honestly — never an
  inline live scrape (which wouldn't survive rate limits or scale).

---

## Scale: what changes, what doesn't

| Layer | This prototype | At tens of thousands of filings |
|---|---|---|
| Fact Store | JSON + in-memory filter | columnar store (DuckDB/Parquet), partitioned by company/period + index |
| Ingestion | parse one PDF inline | batch pipeline; messy tables escalate to an LLM/vision fallback, still grounding-checked |
| Prose retrieval | TF-IDF cosine, in-memory | dense embeddings + vector index — same `ProseIndex` interface |
| EDGAR | sparing per-filing fetch | bulk-archive ingestion; EDGAR stays constrained, never per-query |

**Unchanged at scale:** the query layer and the trust invariants (grounding,
disambiguation, the confidence gate, traceability) — because the answer path only
reads the Bank.

---

## Honest weaknesses

- **Table extraction on messy layouts** is the real weak link: pdfplumber/Camelot
  reach only ~65–75% on *complex* financial tables and need per-document tuning.
  The demo uses clean consolidated statements; the ~65–75% reality is the #1
  thing that breaks at full scale, which is why extraction is deterministic-first
  with an LLM/vision fallback and a grounding check on every number.
- **Label disambiguation**, **restated / amended filings**, and
  **period-alignment edge cases** remain hard.
- **Prose answers are paraphrased** (softer grounding than verbatim numbers), and
  lexical retrieval can miss a paraphrased query that a dense embedder would
  catch — which is why sources are always shown.

---

## Repo map

```
config.py        paths, model, EDGAR access policy
edgar.py         sparing EDGAR front-end: resolve_entity (CIK), discover, fetch
ingest.py        render filing HTML -> PDF (headless Chrome)
extract.py       rendered PDF -> grounded Facts (the trust-critical spine)
bank.py          THE BANK: Fact Store + Prose Store (refuses ungrounded facts)
prose.py         Risk/MD&A extraction + TF-IDF retrieval + grounded narration
llm.py           isolated Anthropic wrapper (the only module that calls the LLM)
core.py          the engine: understand -> resolve -> find -> compute -> gate -> answer
pipeline.py      ingest pipeline (PDF -> facts) with a CSV processing/error log
benchmark.py     hand-verified Q/A with a tolerance band
demo.ipynb       the walkthrough
app.py           Streamlit dashboard (trust-first skin over core.answer)
.streamlit/      native theme (config.toml)
tests/           185 hermetic unit tests
design_doc/      architecture diagram, design reasoning, references
filings/ store/  raw PDFs / the built Bank + ingest_log.csv (gitignored)
```
