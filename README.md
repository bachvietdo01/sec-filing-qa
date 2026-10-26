# SEC Filing Q&A

Ask plain-English questions about 10-K / 10-Q filings and get answers you can check:
every figure is copied from a filing cell, cited to its page, and refused rather than
guessed when the filing doesn't support it.

**[Live demo →](https://huggingface.co/spaces/bachdo01/sec-filing-qa)** · 14 companies ·
5,676 grounded facts · 490 tests

---

## The idea

Financial assistants lose trust in four ways: they invent numbers, answer a
look-alike metric, mix a quarter with a year, and give no way to check. A better
prompt doesn't fix any of those. A different split of responsibilities fixes all four:

> **Code owns every fact. The LLM only owns the wording, and every word it writes
> is checked against what the code decided.**

```
question ─► router (code) ─► the Bank (code) ─► draft reply (code) ─► LLM rewrite ─► verifier ─► user
               lanes,          verbatim cells,      answer / refusal /     tone only      any doubt →
               periods         page provenance      hedge, with figures                   show the draft
```

The LLM never reads a financial table and never produces a figure. Numbers come
from code; in a Risk Factors / MD&A summary, they can only appear as verbatim quotes
from the retrieved passage. If the LLM fails a check, or no key is configured, the
user sees the code's own draft: same numbers, plainer wording.

## What it looks like

```
> What was Kratos's revenue growth from 2023 to 2025?
  Total revenue changed +29.9% over the whole period, from $1,037.1 million (FY2023)
  to $1,346.8 million (FY2025) — a +14.0% compound annual growth rate.
  Year over year: +9.6%; +18.5%.            [10-K p.94 · cells 1,037.1 / 1,136.3 / 1,346.8]

> Okay, for each year?            → carries the metric and periods from the last turn
> What about net income?          → same periods; flags that a % change off a loss isn't meaningful
> Tesla automotive revenue 2025   → refused: a segment, not the consolidated total it would have used
> Should I buy NVDA?              → declines advice; offers the fundamentals instead
```

## Why you can trust an answer

| Failure mode | Mechanism |
|---|---|
| **Invented numbers** | A fact enters the Bank only if its printed cell appears verbatim on its page. Answers compose only Bank facts; arithmetic (growth, margins, CAGR, FCF, EBITDA) runs in code from cited cells. |
| **Look-alike metrics** | A **default-refuse allowlist**: every content word in a question must be accounted for. "Energy revenue", "revenue in the US", "consolidated net income" and "revenue per employee" are refused instead of answered with total revenue or net income. |
| **Period mixing** | Each requested period is its own **leg** with its own kind (annual, quarter, year-to-date). "Q2 2025 and FY2025" answers both; "growth from Q2 to the full year" is refused. Fiscal quarters follow each company's fiscal calendar, so Disney's June quarter is fiscal Q3. |
| **LLM drift** | `voice.py` checks every rewrite: numbers must match in both directions (within rounding), direction words must agree with signs, and no opinions or forecasts. Caveats are re-attached verbatim, and refusals must still decline. Chat rewrites may not agree with a user's premise or add new names, periods or products. Prose summaries must quote their figures verbatim. |
| **Misread questions** | The LLM's parse is only a suggestion and is checked against the text. If it guesses a metric the user never named, the assistant asks instead of answering. Misspellings ("operrating", "Telsa") are corrected only toward the app's own vocabulary, or by an LLM fix accepted only word-for-word, and the reply says what it read. Injected role tags and hidden markup are stripped. |
| **No way to verify** | Every answer carries a trace: filing, page, verbatim cell, period-end date (shown only when known, never guessed), and the reasoning steps. |

It's built to say **no** well: a missing period names the periods that exist, a
component refusal offers the company-wide figure, and "is that good?" declines to
grade but offers the year-over-year change.

## How it works

- **The Bank** (`store/`): facts from the income statement, balance sheet and cash
  flows of each 10-K/10-Q, each with page, verbatim cell, scale, period kind and
  fiscal period. Risk Factors and MD&A passages are kept alongside for retrieval.
- **The router** (`core.py`): one fixed order of lanes, decided in code. First the
  conversation lanes (advice, market data, definitions, small talk), then company
  resolution against the banked set, then the company lanes in order: coverage,
  overview, non-GAAP, prose, unsupported period, untracked metric, and finally the
  factual answer.
  Follow-ups work by **filling slots**: a question inherits only the metric, periods
  or calculation it leaves blank, and the reply says so.
- **The voice** (`voice.py`): one render-and-verify function for every LLM output.
  Offline it returns the draft, so behaviour without an API key is identical.

Design rationale and diagrams: [`design_doc/`](design_doc/).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env   # optional; without it the app runs offline-deterministic

streamlit run app.py                         # the chat app
python3 -m pytest tests/ -q                  # 490 hermetic tests, no network
python3 benchmark.py                         # hand-verified Q/A spot checks (8/8)
jupyter notebook demo.ipynb                  # walkthrough with traces (Tesla)
```

**Add a company** (needs headless Chrome to render the filing; EDGAR is touched only at ingestion):

```bash
EDGAR_CONTACT=you@example.com python3 pipeline.py KTOS    # latest 10-K + 10-Q → grounded facts
```

The pipeline banks the financial-statement facts. Risk Factors / MD&A passages are
extracted separately with `prose.extract_prose` on the rendered 10-K.

Covered today: TSLA · GOOG · NVDA · AVGO · DIS · AAL · HSY · VRT · SOFI · ISRG · BE · ASTS · CRWV · KTOS.

## Limits, honestly

- **Scope is the three core statements plus Risk Factors and MD&A.** There are no
  segments, notes, Item 1 business descriptions or market data, so questions about
  them are refused by design.
- **Table extraction is the weak link at scale.** Messy layouts need per-filer label
  synonyms. Some filers split depreciation and amortization into separate lines,
  and EBITDA is refused for them.
- **Prose summaries can fall back to the verbatim passage** when the narrator can't
  quote its figures exactly.
- **English only.** Follow-up memory spans one factual turn.

## Repo map

```
core.py       router, lanes, follow-up frames, answerers (the engine)
voice.py      the verifier between code drafts and LLM wording
extract.py    rendered PDF → grounded facts (periods, scale, grounding check)
prose.py      Risk Factors / MD&A extraction, retrieval, quoted narration
bank.py       the Bank: fact and prose stores
edgar.py      sparing EDGAR access: ticker → CIK, filing discovery, fetch
ingest.py     filing HTML → PDF (headless Chrome)
pipeline.py   ingestion with a processing log
llm.py        the only module that calls the LLM (+ per-session spending switch)
app.py        Streamlit chat UI with citations, reasoning and usage caps
benchmark.py  hand-verified Q/A spot checks with a tolerance band
stress_harness.py   isolated temp bank for stress-testing a new ticker
demo.ipynb    walkthrough with traces
tests/        490 tests: extraction, router, verifier, adversarial regressions
design_doc/   architecture and design reasoning
store/        the built Bank        filings/  source PDFs and HTML
```
