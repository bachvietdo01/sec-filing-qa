"""The engine: question in -> grounded, traceable, hedged answer out.

The agentic flow, with the LLM fenced to language only:

    understand (LLM*)  question -> {metric, years, calc, kind, company}
    resolve_entity     company mention -> CIK (static SEC map; code)
    find_number        metric+period -> the banked fact(s)   (code)
                         * exact label via an alias table
                         * ambiguous label -> disambiguate, never silent-pick
    compute            growth / margin / delta                (code)
    validate + gate    grounding, period & scale sanity -> answer|hedge|refuse
    write_answer (LLM*) phrase over the verified numbers       (language only)

(* LLM steps fall back to deterministic logic when no API key is set, so the
pipeline runs offline. No number ever originates from the LLM.)

Every numeric answer carries a trace: filing, page, the raw printed cell(s), and
the arithmetic steps. That is the antidote to the prior system's failure modes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import bank
import edgar
import llm
import prose
import voice

# --------------------------------------------------------------------------- #
# Metric dictionary: canonical key -> where the number lives in the Bank.
# This table is where label disambiguation is encoded deterministically.
# `scale` filters rows that share a label (e.g. "Basic" is both EPS and shares).
# --------------------------------------------------------------------------- #
@dataclass
class MetricSpec:
    name: str                 # human-readable
    statement: str
    labels: tuple[str, ...]   # candidate raw_labels, tried in PRIORITY order
    scale: str | None = None  # filter when a label is reused (EPS vs shares)
    margin_eligible: bool = False
    alt_note: str = ""        # surfaced caveat (e.g. a look-alike line item)
    # labels that only identify the metric under a given SECTION header — Broadcom
    # prints "Net income per share" twice, under "Basic income per share:" and
    # "Diluted income per share:". (label, section-substring) pairs, tried last.
    section_labels: tuple = ()
    # last-resort caption PATTERN (matched on the normalized label) for filers
    # whose caption appends their own name: "Diluted income (loss) per common
    # share attributable to Kratos:". Anchored, so a share COUNT never matches.
    label_pattern: str = ""


METRICS: dict[str, MetricSpec] = {
    # Label synonyms cover filer variation (SEC fixes line-item *captions* loosely,
    # not statement titles): Hershey "Net sales", Broadcom "Net revenue", etc.
    # Tried in order; the first exact match in the right statement wins.
    "revenue": MetricSpec("Total revenue", "operations",
                          ("Total revenues", "Total revenue", "Net revenues",
                           "Net revenue", "Net sales", "Total net revenue",
                           "Total net sales", "Total operating revenues",
                           "Operating revenues", "Revenues", "Revenue")),
    "gross_profit": MetricSpec("Gross profit", "operations",
                               ("Gross profit", "Gross margin"), margin_eligible=True),
    "operating_income": MetricSpec(
        "Operating income", "operations",
        ("Income from operations", "Operating income", "Operating profit",
         "Operating income (loss)", "Income (loss) from operations",
         "Loss from operations", "Operating loss"), margin_eligible=True),
    "operating_expenses": MetricSpec("Total operating expenses", "operations",
                                     ("Total operating expenses",)),
    "rnd": MetricSpec("Research and development", "operations",
                      ("Research and development", "Research and development costs",
                       "Research and development expenses",
                       "Research and development expense")),
    "net_income": MetricSpec(
        "Net income", "operations",
        ("Net income attributable to common stockholders",
         "Net loss attributable to common stockholders",
         "Net income (loss) attributable to common stockholders",
         # NOT "...attributable to common stockholders, basic/diluted": those are
         # EPS numerators (they disagree with the consolidated bottom line, e.g.
         # CoreWeave −1,196 basic vs −1,167 consolidated). Prefer the real line.
         "Net income", "Net income (loss)", "Net loss",
         "Net loss and comprehensive loss",
         "Net loss before allocation to noncontrolling interest",
         # "Net earnings" is Procter & Gamble's (and other consumer filers')
         # caption for net income; tried AFTER the "Net income" variants so a
         # filer that reports both is unaffected (first match in order wins).
         "Net earnings", "Net earnings (loss)",
         "Net earnings attributable to common stockholders"),
        margin_eligible=True,
        alt_note='when a filing reports both total "Net income" and the amount '
                 '"attributable to common stockholders", this prefers the latter '
                 "(the EPS numerator)"),
    "eps_basic": MetricSpec(
        "Basic EPS", "operations",
        ("Basic", "Basic net income per share", "Basic earnings per share",
         "Basic net income (loss) per share", "Basic earnings (loss) per share",
         "Basic net income per share of common stock", "Basic net loss per share",
         "Net income per share, basic", "Net income per share - basic",
         "Earnings (loss) per share - basic", "Earnings per share - basic",
         "Net loss per share, basic and diluted",
         "Net loss per share available to common stockholders, basic and diluted",
         "Basic and diluted",
         "Net loss per share attributable to common stockholders, basic"),
        scale="units",
        section_labels=(("Net income per share", "basic"), ("Net loss per share", "basic"),
                        ("Net income (loss) per share", "basic")),
        label_pattern=r"^basic (?:net )?(?:income|earnings|loss)(?: \(loss\))? per (?:common )?"
                      r"share(?: attributable to [^,]+)?:?$"),
    "eps_diluted": MetricSpec(
        "Diluted EPS", "operations",
        ("Diluted", "Diluted net income per share", "Diluted earnings per share",
         "Diluted net income (loss) per share", "Diluted earnings (loss) per share",
         "Diluted net income per share of common stock", "Diluted net loss per share",
         "Net income per share, diluted", "Net income per share - diluted",
         "Earnings (loss) per share - diluted", "Earnings per share - diluted",
         "Net loss per share, basic and diluted",
         "Net loss per share available to common stockholders, basic and diluted",
         "Basic and diluted",
         "Net loss per share attributable to common stockholders, diluted"),
        scale="units",
        section_labels=(("Net income per share", "diluted"), ("Net loss per share", "diluted"),
                        ("Net income (loss) per share", "diluted")),
        label_pattern=r"^diluted (?:net )?(?:income|earnings|loss)(?: \(loss\))? per (?:common )?"
                      r"share(?: attributable to [^,]+)?:?$"),
    "cash": MetricSpec("Cash and cash equivalents", "balance_sheet",
                       ("Cash and cash equivalents", "Cash and equivalents",
                        "Cash")),
    "total_assets": MetricSpec("Total assets", "balance_sheet", ("Total assets",)),
    "total_liabilities": MetricSpec("Total liabilities", "balance_sheet",
                                    ("Total liabilities",)),
    "total_current_assets": MetricSpec("Total current assets", "balance_sheet",
                                       ("Total current assets",)),
    "total_current_liabilities": MetricSpec(
        "Total current liabilities", "balance_sheet",
        ("Total current liabilities",)),
    "total_equity": MetricSpec(
        "Total equity", "balance_sheet",
        ("Total stockholders' equity", "Total shareholders' equity",
         "Total equity", "Total stockholders' equity (deficit)",
         "Total shareholders' investment", "Total members' equity",
         "Stockholders' equity", "Total stockholders' equity attributable to "
         "common stockholders", "Total permanent equity")),
    "dep_amort": MetricSpec(
        "Depreciation & amortization", "cash_flows",
        ("Depreciation, amortization and impairment",
         "Depreciation and amortization", "Depreciation, depletion and amortization",
         "Depreciation, amortization and accretion, net",
         "Depreciation and amortization expense",
         "Depreciation, amortization, and impairment")),  # NOT bare "Depreciation":
                                                          # split-D&A filers refuse
    "operating_cash_flow": MetricSpec(
        "Net cash from operating activities", "cash_flows",
        ("Net cash provided by operating activities",
         "Net cash provided by (used in) operating activities",
         "Net cash provided by (used for) operating activities",
         "Net cash used in operating activities",
         "Net cash and cash equivalents provided by operating activities",
         "Cash provided by operations", "Cash used in operations",
         "Cash flows from operations", "Net cash from operations",
         "Net cash flows provided by operating activities",
         # Procter & Gamble labels the operating-activities subtotal
         # "Total operating activities" (the indirect-method net cash figure).
         "Total operating activities")),
    # Capital expenditures — the investing-activities outflow. Labels vary a lot
    # across filers, so this is a curated EXACT-synonym list (no fuzzy matching:
    # a wrong grab like "Depreciation of property..." would corrupt FCF). Used to
    # compute free cash flow; refuses cleanly when a filer's label isn't covered.
    "capex": MetricSpec(
        "Capital expenditures", "cash_flows",
        ("Capital expenditures",
         "Capital expenditures and aircraft purchase deposits",
         "Purchases of property and equipment",
         "Purchases of property, plant and equipment",
         "Purchases of property and equipment, net",
         "Purchases of property and equipment excluding finance leases, net of sales",
         "Purchases related to property and equipment and intangible assets",
         "Investments in parks, resorts and other property",
         "Additions to property, plant and equipment",
         "Additions to property and equipment",
         "Payments for property and equipment",
         "Payments to acquire property and equipment",
         "Capital expenditures, net", "Purchase of property and equipment",
         "Expenditures for property and equipment",
         "Purchases of property, plant, and equipment",
         "Purchase of property, plant and equipment",
         "Purchases of property, equipment and software",
         "Purchase of property and equipment, including capitalized internal-use software",
         "Purchase of property and equipment, including capitalized internal-use",
         "Capital additions (including software)")),
}

# --------------------------------------------------------------------------- #
# Computed metrics — derived in CODE from grounded line items (each input is
# cited). Scope is deliberately filing-only: these need nothing but the three
# statements. Valuation ratios (P/E, EV/EBITDA, market cap, yield) are NOT here:
# they require a market share price, which a 10-K/10-Q does not contain.
#   op:  pct   -> a / b * 100  (%)        ratio -> a / b  (x)
#        sum   -> a + b  (money)          fcf   -> a - |b|  (money)
# --------------------------------------------------------------------------- #
COMPUTED = {
    "ebitda": ("EBITDA", "sum", "operating_income", "dep_amort"),
    "roa": ("Return on assets", "pct", "net_income", "total_assets"),
    "roe": ("Return on equity", "pct", "net_income", "total_equity"),
    "current_ratio": ("Current ratio", "ratio", "total_current_assets",
                      "total_current_liabilities"),
    "free_cash_flow": ("Free cash flow", "fcf", "operating_cash_flow", "capex"),
}
_SCALE_MULT = {"thousands": 1e3, "millions": 1e6, "billions": 1e9, "units": 1.0}

# cheap keyword cues for the offline (no-LLM) understand() fallback
_METRIC_CUES = {
    "revenue": ["revenue", "revenues", "sales", "top line"],
    "gross_profit": ["gross profit", "gross margin"],
    "operating_income": ["operating income", "income from operations",
                         "operating profit", "operating margin", "operating loss",
                         "loss from operations"],
    "operating_expenses": ["total operating expenses", "operating expenses",
                           "opex"],
    "net_income": ["net income", "net profit", "net loss", "earnings",
                   "bottom line", "profit", "net margin", "profit margin",
                   "profitable", "profitability", "make money", "made money",
                   "lose money", "lost money"],
    "eps_diluted": ["diluted eps", "diluted earnings per share"],
    "eps_basic": ["eps", "earnings per share"],
    "cash": ["cash", "cash and cash equivalents", "cash on hand"],
    "total_assets": ["total assets"],
    "total_liabilities": ["total liabilities"],
    "total_current_assets": ["total current assets", "current assets"],
    "total_current_liabilities": ["total current liabilities",
                                  "current liabilities"],
    "dep_amort": ["depreciation and amortization", "depreciation & amortization",
                  "d&a", "depreciation"],
    "rnd": ["r&d", "research and development", "research & development", "research"],
    "operating_cash_flow": ["operating cash flow", "cash from operations",
                            "cash flow from operations", "operating activities",
                            "cash provided by operating", "cash flow", "cash flows",
                            "net cash from operating activities",
                            "cash from operating activities",
                            "net cash provided by operating activities",
                            "operating cashflow", "cashflow from operations"],
    "capex": ["capex", "capital expenditure", "capital expenditures",
              "capital spending", "purchases of property and equipment",
              "purchases of property, plant and equipment", "capital additions"],
    "free_cash_flow": ["free cash flow", "fcf", "operating cash flow minus capex",
                       "operating cash flow less capex", "ocf minus capex",
                       "cash flow minus capex"],
    "ebitda": ["ebitda", "earnings before interest, taxes, depreciation and amortization",
               "earnings before interest taxes depreciation and amortization"],
    "roa": ["return on assets", "roa"],
    "roe": ["return on equity", "roe"],
    "current_ratio": ["current ratio"],
    "total_equity": ["total equity", "shareholders equity", "stockholders equity",
                     "stockholders' equity", "shareholders' equity",
                     "total stockholders' equity", "permanent equity",
                     "book value", "net worth"],
}
# the phrases that ask for a metric AS A MARGIN (vs. its dollar level)
_MARGIN_CUES = {
    "operating_income": ["operating margin", "operating income margin",
                         "operating profit margin"],
    "gross_profit": ["gross margin", "gross profit margin"],
    "net_income": ["net margin", "profit margin", "net income margin", "net profit margin"],
    "ebitda": ["ebitda margin"],
    "free_cash_flow": ["fcf margin", "free cash flow margin"],
}
_CALC_CUES = {
    "growth": ["growth", "grow", "increase", "change", "yoy", "year over year",
               "year-over-year", "vs", "versus", "compared", "cagr",
               "compound annual", "annualized", "annualised", "qoq", "sequential",
               "sequentially", "quarter over quarter", "quarter-over-quarter"],
    "margin": ["margin"],
}


@dataclass
class Query:
    kind: str                 # numeric | prose
    company_mention: str
    metric: str | None
    years: list[int]
    calc: str                 # none | growth | margin
    raw_question: str
    period_kind: str = "annual"   # annual | quarter | ytd
    quarter: int | None = None    # 1..4 when period_kind is quarter/ytd
    metrics: list = field(default_factory=list)  # all metrics asked (multi)
    instant_end: str | None = None  # explicit as-of date "MM-DD" (balance sheet)
    legs: list = field(default_factory=list)       # per-period Legs (router)
    untracked: list = field(default_factory=list)  # [(start, name)] asked, not banked
    pct_of_revenue: bool = False                   # "R&D as a % of revenue"
    followup: bool = False                         # metric/period carried over
    suggested_metric: str | None = None            # LLM guess when text names none
    prose_hint: str = ""                           # carried metric for "why...?"
    margin_metrics: list = field(default_factory=list)  # metrics "margin" modifies
    items: list = field(default_factory=list)      # [(metric, calc)] when one question
                                                   # asks a metric as a level AND a margin


@dataclass(frozen=True)
class Leg:
    """One requested period, carrying its OWN kind — so "Q2 2025 and fiscal 2025"
    is two legs (a quarter and a year), never one global period_kind that would
    silently coerce one of them."""
    kind: str                 # annual | quarter | ytd
    year: int | None
    quarter: int | None = None
    end_month: int | None = None  # "quarter ended March" -> fiscal quarter per company
    explicit: bool = False        # False only for a bare year defaulted to annual


@dataclass
class Trace:
    steps: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)  # filing/page/cell

    def cite(self, fact: dict):
        self.citations.append({
            "filing": fact["filing_stem"], "form": fact["form"],
            "page": fact["page_no"], "label": fact["raw_label"],
            "period": fact["period"], "cell": fact["raw_cell_text"],
            "unit": fact["unit"], "scale": fact["scale"],
            # the exact as-of/period-end date — None when the filing never printed
            # the day (it was assumed), so a guess is never shown as exact
            "period_end": (None if fact.get("period_end_inferred")
                           else fact.get("period_end")),
            "period_kind": fact.get("period_kind"),
        })


@dataclass
class Answer:
    text: str
    confidence: float
    status: str               # answered | hedged | refused
    trace: Trace
    value: float | None = None
    passages: list | None = None   # retrieved source chunks (prose answers)
    headline: tuple | None = None  # (label, value_str) for the metric hero
    query: object = None           # the resolved Query (context for a follow-up)


# --------------------------------------------------------------------------- #
# 1. understand  (LLM -> structured query; deterministic fallback)
# --------------------------------------------------------------------------- #
_UNDERSTAND_SYS = (
    "You convert a finance question about SEC filings into a strict JSON query. "
    "Fields: kind ('numeric' or 'prose'); company (the company name/ticker "
    "mentioned); metric (one of: " + ", ".join(list(METRICS) + list(COMPUTED))
    + ", or null for "
    "prose); metrics (a LIST of every metric key the question asks for — one or "
    "more, in order); years (list of 4-digit integers the question refers to; [] if "
    "unspecified); calc ('none', 'growth', or 'margin'); period ('annual', "
    "'quarter', or 'ytd', default 'annual') and quarter (1-4 or null). Examples: "
    "'Q2 2026' or 'second quarter' -> period 'quarter', quarter 2; 'first half' / "
    "'six months' / 'year to date' -> period 'ytd'. You never compute or output "
    "any financial number."
)


def _expand_year_range(question: str, years: list[int]) -> list[int]:
    """Expand an inclusive year RANGE ("2023-2025", "from 2023 to 2025", "each
    year 2023 through 2025") to every year in it, so a multi-period request isn't
    collapsed to its two endpoints. Only fires when the phrasing signals a range."""
    if len(years) < 2:
        return years
    ql = (question or "").lower()
    ranged = (_re.search(r"(?:19|20)\d{2}\s*(?:-|–|—|to|through|thru)\s*(?:19|20)\d{2}", ql)
              or _re.search(r"\b(?:each|every|all)\b[^.]*\byears?\b", ql)
              or (_re.search(r"\bfrom\b", ql) and _re.search(r"\bto\b", ql))
              or (_re.search(r"\bbetween\b", ql) and _re.search(r"\band\b", ql)))
    if not ranged:
        return years
    lo, hi = min(years), max(years)
    return list(range(lo, hi + 1)) if 0 < hi - lo <= 15 else years


def understand(question: str) -> Query:
    question = question or ""                        # never crash on None
    if llm.available():
        try:
            d = llm.complete_json(_UNDERSTAND_SYS, question, max_tokens=256)
            qn = d.get("quarter")
            pk = (d.get("period") or "annual").lower()
            if pk not in ("annual", "quarter", "ytd"):   # clamp unknown values
                pk = "quarter" if "q" in pk else "annual"
            valid = set(METRICS) | set(COMPUTED)
            mlist = [m for m in (d.get("metrics") or []) if m in valid]
            if not mlist and d.get("metric") in valid:
                mlist = [d["metric"]]
            return _reconcile(_understand_fallback(question), mlist,
                              d.get("company", "") or "")
        except Exception:
            pass  # fall through to the deterministic parser
    return _understand_fallback(question)


def _reconcile(fb, llm_metrics, llm_company) -> Query:
    """The LLM parse is a SUGGESTION, checked against the text. Metric, years and
    calc come from the deterministic parse of what the user actually wrote — an
    LLM that swaps revenue for net income, invents a year, or adds a growth calc
    can no longer turn a correctly cited figure into an answer to a question
    nobody asked. What the LLM still contributes:
      * a company name the offline parser missed;
      * a metric GUESS when the wording names none ("how much did they make?") —
        kept as `suggested_metric`, which makes the router ASK, never answer."""
    if not fb.company_mention and llm_company:
        fb.company_mention = llm_company
    if not fb.metrics and llm_metrics:
        fb.suggested_metric = llm_metrics[0]
    return fb


def _has_prose_cue(text: str, has_metric: bool) -> bool:
    """Does the question actually ask for the filing's NARRATIVE (Risk Factors /
    MD&A)? One deterministic test for both the offline parser and the LLM path —
    the LLM's own 'prose' label is not trusted on its own (a follow-up like "what
    is the growth rate for the entire period?" is not a narrative question)."""
    q = (text or "").lower()
    strong = any(w in q for w in ["risk", "md&a", "management discussion",
                                  "discussion and analysis"])
    # "why did margin fall / what drove revenue" asks for management's narrative
    # about a METRIC — prose, but only when a metric is actually named.
    why_cue = any(w in q for w in ["management explain", "explain the", "why did",
                                   "why was", "why were", "what drove", "drivers of",
                                   "reasons for", "according to management"])
    # filing-specific narrative cues only — NOT generic words like "strategy /
    # litigation / outlook / competition / discuss" that drag off-topic questions
    # into prose retrieval.
    weak = any(w in q for w in ["liquidity", "management discuss",
                                "management's discussion", "going concern",
                                "say about", "company say", "disclose about",
                                "material weakness", "capital resources",
                                "risk factor"])
    return strong or (has_metric and why_cue) or (not has_metric and weak)


def _requested_items(q) -> list:
    """Split a question into (metric, calc) items when "margin" doesn't apply to
    every metric: "revenue, operating income and operating margin" is revenue
    (level), operating income (level) AND operating income (margin) — never a
    "total revenue margin". Empty when one calc covers the whole question."""
    t = _norm_text(q.raw_question)
    mks = [m for m in (q.metrics or []) if m in METRICS or m in COMPUTED]
    if not mks:
        return []
    items, any_margin = [], False
    for m in mks:
        mspans = [(x.start(), x.end()) for c in _MARGIN_CUES.get(m, [])
                  for x in _re.finditer(r"\b" + _re.escape(c) + r"\b", t)]
        level = any(not any(s0 <= x.start() < e0 for s0, e0 in mspans)
                    for c in _METRIC_CUES.get(m, []) if "margin" not in c
                    for x in _re.finditer(r"\b" + _re.escape(c) + r"\b", t))
        if mspans and m != "revenue":
            any_margin = True
            if level:
                items.append((m, "none"))
            items.append((m, "margin"))
        else:
            items.append((m, "none"))
    if not any_margin or len(items) < 2:
        return []
    return items


def _cue_spans(q: str) -> dict:
    """{metric_key: (start, end, cue_len)} for every metric cue in lowercase `q`.
    Each metric's best match is its longest cue; a metric whose span is CONTAINED
    in a longer metric's span is dropped, so the more specific phrase wins
    ("earnings per share" beats "earnings"; "free cash flow" absorbs "cash")."""
    spans = {}
    for key, cues in _METRIC_CUES.items():
        best = None
        for cue in cues:
            m = _re.search(r"\b" + _re.escape(cue) + r"\b", q)   # word-boundary:
            if not m:                                            # 'roa' != 'bROAdcom'
                continue
            p = m.start()
            if best is None or len(cue) > best[2] or (len(cue) == best[2] and p < best[0]):
                best = (p, p + len(cue), len(cue))
        if best:
            spans[key] = best
    drop = {k for k in spans for j in spans
            if k != j and spans[j][0] <= spans[k][0]
            and spans[k][1] <= spans[j][1] and spans[j][2] > spans[k][2]}
    return {k: v for k, v in spans.items() if k not in drop}


def _understand_fallback(question: str) -> Query:
    import re
    question = question or ""                       # never crash on None
    q = question.lower()
    # Years: accept 19xx/20xx and tolerate thousands commas ("year 2,024"). A
    # named-but-unavailable year (e.g. 1999) then flows to find_fact and refuses,
    # rather than silently defaulting to the latest banked year.
    years = [int(y) for y in re.findall(r"\b((?:1[89]|20)\d{2})\b",
                                        question.replace(",", ""))]
    # Collect EVERY metric whose cue appears (supports multi-metric questions
    # like "revenue and operating income"). Each metric's best match is its
    # longest cue; then a metric whose matched span is CONTAINED in a longer
    # metric's span is dropped, so the more specific phrase wins ("earnings per
    # share" beats "earnings"; "free cash flow" absorbs a stray "cash").
    spans = _cue_spans(q)
    metrics = sorted(spans, key=lambda k: spans[k][0])

    # An EARNINGS "per share" caption ("earnings (loss) per share – diluted",
    # "net income per share") breaks the contiguous "earnings per share" cue and
    # matches net_income's greedy "earnings" cue — which would answer the dollar
    # TOTAL. Force EPS, but ONLY for an earnings/income/loss per-share phrase —
    # NOT "dividend / book value / revenue / cash flow per share" (those are
    # different, unbanked ratios; the unsupported-metric guard refuses them rather
    # than letting them be answered with EPS).
    if (re.search(r"(earnings|income|loss|profit|eps)\b[^.]*\bper\s+(?:diluted\s+|basic\s+)?share", q)
            and not (set(metrics) & {"eps_basic", "eps_diluted"})):
        eps = "eps_basic" if "basic" in q else "eps_diluted"
        metrics = [eps] + [m for m in metrics if m != "net_income"]
    metric = metrics[0] if metrics else None
    calc = "none"
    for c, cues in _CALC_CUES.items():
        if any(cue in q for cue in cues):
            calc = c
            break
    # Prose routing: a STRONG qualitative cue (risk / MD&A) makes the question
    # prose even when a metric word appears incidentally ("main risks to revenue"
    # is a narrative question, not a request for the revenue number). Weaker cues
    # only route to prose when no metric was named.
    # Only FILING-SPECIFIC narrative cues route to prose. Generic words ("strategy",
    # "litigation", "outlook", "competition", "discuss", "narrative") were too
    # eager — they pulled off-topic questions ("explain litigation in a divorce",
    # "outlook for the football game") into prose retrieval, which then leaked or
    # dead-ended. Those now fall through to the conversational steer-back instead.
    kind = "prose" if _has_prose_cue(q, bool(metrics)) else "numeric"
    # company mention: capitalized words (original casing), minus question
    # stopwords; or an all-caps ticker. The LLM path does this far better —
    # this only needs to be serviceable offline.
    stop = _MENTION_STOP
    # grab CONSECUTIVE capitalized tokens so a multi-word name survives ("American
    # Airlines", "Walt Disney") instead of collapsing to its ambiguous first word.
    mention = ""
    for seq in re.findall(r"[A-Z][a-zA-Z.&]*(?:\s+[A-Z][a-zA-Z.&]*)*", question):
        words = [w for w in seq.split() if w not in stop]
        if words:
            mention = " ".join(words)
            break
    if not mention:
        tk = [t for t in re.findall(r"(?<![&/])\b[A-Z]{2,5}\b(?![&/])", question)
              if t not in _ACRONYM_STOP]
        mention = tk[0] if tk else ""

    # period intent: quarter / ytd / annual
    period_kind, quarter = "annual", None
    _WORD_Q = {"first": 1, "second": 2, "third": 3, "fourth": 4}
    mq = re.search(r"\bq([1-4])\b", q) or re.search(
        r"\b(first|second|third|fourth)\s+quarter", q)
    if "year to date" in q or "ytd" in q or "first half" in q \
            or "six months" in q or "nine months" in q or "first nine" in q:
        period_kind = "ytd"
        # a bare "YTD" / "year to date" names no length: leave quarter None and let
        # the answerer pick the latest banked YTD (a 9-month YTD must not be asked
        # for as 6 months just because "half" was the old default).
        quarter = (3 if ("nine months" in q or "first nine" in q) else
                   2 if ("six months" in q or "first half" in q) else None)
    elif mq:
        period_kind = "quarter"
        g = mq.group(1)
        quarter = int(g) if g.isdigit() else _WORD_Q[g]
    elif "quarter" in q or "quarterly" in q:
        period_kind = "quarter"
    # An explicit as-of date with a DAY ("as of March 31, 2026", "March 31,
    # 2026") pins a balance-sheet instant to that exact date, so an interim
    # as-of is never silently answered with the fiscal year-end balance. A
    # month without a day ("nine months ended March 2026") is left to the
    # duration logic above.
    instant_end = None
    _MN = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
           "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
           "november": 11, "december": 12, "jan": 1, "feb": 2, "mar": 3,
           "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
           "oct": 10, "nov": 11, "dec": 12}
    md = re.search(r"\b(" + "|".join(_MN) + r")\.?\s+(\d{1,2})\b", q)
    if md:
        mon, day = _MN[md.group(1)], int(md.group(2))
        if 1 <= day <= 31:
            instant_end = f"{mon:02d}-{day:02d}"
    return Query(kind, mention, metric, _expand_year_range(question, years), calc,
                 question, period_kind=period_kind, quarter=quarter,
                 metrics=metrics, instant_end=instant_end)


# --------------------------------------------------------------------------- #
# 2. find_number  (deterministic match + disambiguation)
# --------------------------------------------------------------------------- #
_Q_END_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}

import re as _re
_FOOTNOTE_RE = _re.compile(r"[\s\d*†‡]+$")   # trailing footnote markers


def _norm_label(s: str) -> str:
    """Normalize a line-item label for matching: unify curly apostrophes and
    en/em dashes, lowercase, strip trailing footnote markers (Bloom "Total
    revenue1"), and drop a trailing "(loss)"/"(income)" qualifier (Vertiv
    "Operating profit (loss)"). raw_label stays verbatim in the Bank; only the
    comparison is normalized."""
    s = (s or "").replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    s = _FOOTNOTE_RE.sub("", s.strip()).strip().lower()
    s = _re.sub(r"\s*\((loss|income|loss/income|income/loss)\)\s*$", "", s)
    return s.strip()


def _fiscal_quarter_of(month: int, fy_end: int) -> int:
    """Fiscal quarter (1..4) of a period ENDING in calendar `month`, given the
    company's fiscal year-end month. A Dec filer (fy_end=12) yields the calendar
    quarters (Mar->Q1, Jun->Q2, …); a June filer (MSFT) maps Sep->Q1, Dec->Q2,
    Mar->Q3, Jun->Q4."""
    return ((month - fy_end - 1) % 12) // 3 + 1


def _period_matches(f: dict, spec: MetricSpec, year: int,
                    period_kind: str, quarter: int | None,
                    fy_end_months: set[int] | None = None,
                    instant_end: str | None = None) -> bool:
    """Does a banked fact match the requested period? Matched on period_kind +
    fiscal_year (+ quarter), NOT a reconstructed label — so a quarterly request
    NEVER silently accepts an annual figure (and vice versa). This is the
    annual/quarterly guard at the lookup layer."""
    if f.get("fiscal_year") != year:
        return False
    pk = f.get("period_kind")
    if spec.statement == "balance_sheet":
        if pk != "instant":
            return False
        # An explicitly named as-of date ("as of March 31, 2026") must match that
        # exact date — never fall through to the fiscal year-end instant, which
        # would silently answer a DIFFERENT period.
        if instant_end:
            return f["period_end"][5:10] == instant_end
        if period_kind in ("quarter", "ytd") and quarter:
            m = int(f["period_end"][5:7])
            # FISCAL-quarter match: a balance-sheet instant carries no fiscal
            # quarter, so derive it from its month and the filer's year-end (a
            # June filer's Q3 ends in March, not September). Falls back to the
            # calendar mapping only when the year-end can't be determined.
            if fy_end_months:
                return _fiscal_quarter_of(m, max(fy_end_months)) == quarter
            return m == _Q_END_MONTH[quarter]
        # annual/default -> the FISCAL year-end instant. The year-end month is
        # read from the company's own annual duration statements (`fy_end_months`,
        # e.g. Disney ends in Sept, Broadcom in Nov), not assumed to be December.
        m = int(f["period_end"][5:7])
        if fy_end_months:
            return m in fy_end_months
        return m not in (3, 6, 9)            # fallback when we can't tell
    if period_kind == "quarter":
        return pk == "quarter" and (quarter is None
                                    or f.get("fiscal_quarter") == quarter)
    if period_kind == "ytd":
        return pk == "ytd" and (quarter is None
                                or f.get("fiscal_quarter") == quarter)
    return pk == "annual"


def find_fact(cik: int, metric_key: str, year: int, period_kind: str = "annual",
              quarter: int | None = None,
              instant_end: str | None = None) -> tuple[dict | None, str, list[str]]:
    """Return (fact, status, candidates). status: 'found' | 'ambiguous' |
    'not_found'. Numbers come only from the Bank, matched by exact label +
    period_kind/period (+ scale filter for reused labels)."""
    spec = METRICS[metric_key]
    cik_facts = bank.facts_for(cik)
    # the company's own fiscal year-end month(s), read from its annual duration
    # statements — lets a balance-sheet "annual" lookup find a Sept/Nov year-end.
    fy_end_months = {int(f["period_end"][5:7]) for f in cik_facts
                     if f.get("period_kind") == "annual"
                     and f["statement"] in ("operations", "cash_flows")
                     and f.get("period_end")}
    # an explicit as-of date only constrains a balance-sheet (instant) lookup
    ie = instant_end if spec.statement == "balance_sheet" else None
    rows = [f for f in cik_facts
            if f["statement"] == spec.statement
            and _period_matches(f, spec, year, period_kind, quarter, fy_end_months, ie)
            and (spec.scale is None or f["scale"] == spec.scale)]

    # EPS sanity: a per-share figure is small — reject an implausible magnitude
    # (guards against a mis-scaled weighted-average SHARE COUNT, e.g. AVGO 4,712).
    # EPS magnitude guard: no banked filer reports EPS over ~$100, while the rows
    # that share the generic "Basic"/"Diluted" label and get mis-scaled as per-share
    # — weighted-average SHARE COUNTS (hundreds of millions) and mislabeled dollar
    # numerators (e.g. CoreWeave's "(290)") — are always far larger. A $100 ceiling
    # separates them. (A filer with genuine EPS > $100 would need a smarter rule.)
    if metric_key in ("eps_basic", "eps_diluted"):
        rows = [f for f in rows if abs(f.get("value") or 0) <= 100]

    # Net income: prefer the amount attributable to the PARENT / common holders
    # over a bare consolidated "Net income" (which includes noncontrolling
    # interests) — e.g. Disney's "attributable to The Walt Disney Company".
    if metric_key == "net_income":
        def _is_attrib(f):
            nl = _norm_label(f["raw_label"])
            # the amount attributable to the parent/common holders, but NOT a
            # per-share figure nor a basic/diluted EPS-NUMERATOR variant (CoreWeave
            # reports "…attributable to common stockholders, basic" and "…, diluted"
            # at different values — those are EPS inputs, not the net-income line).
            return (_re.search(r"net (income|loss).*attributable to", nl)
                    and "per share" not in nl
                    and not nl.endswith(", basic") and not nl.endswith(", diluted")
                    and "noncontrolling" not in nl and "redeemable" not in nl)
        attrib = [f for f in rows if _is_attrib(f)]
        if attrib:
            common = [f for f in attrib
                      if "common" in _norm_label(f["raw_label"])]
            pick = common or attrib
            if len({round(h["value"], 4) for h in pick}) == 1:
                tenk = [h for h in pick if h.get("form") == "10-K"]
                return (tenk or pick)[0], "found", []
            return None, "ambiguous", [f"{h['raw_label']}={h['raw_cell_text']}"
                                       for h in pick]

    # Try the candidate labels in PRIORITY order; the first label that matches
    # wins. One hit -> found. Multiple hits: if they AGREE on the value they are
    # just the same figure reported in two filings (a 10-K and a 10-Q's prior-
    # year comparative) -> found (prefer the 10-K). If they DISAGREE it is a real
    # conflict (e.g. a restatement) -> ambiguous, never a silent pick.
    for label in spec.labels:
        want = _norm_label(label)
        hits = [f for f in rows if _norm_label(f["raw_label"]) == want]
        if not hits:
            continue
        if len(hits) == 1:
            return hits[0], "found", []
        if len({round(h["value"], 4) for h in hits}) == 1:   # consistent dup
            tenk = [h for h in hits if h.get("form") == "10-K"]
            return (tenk or hits)[0], "found", []
        return None, "ambiguous", [f"{h['raw_label']} {h['period']}="
                                   f"{h['raw_cell_text']} ({h['form']})"
                                   for h in hits]
    if spec.label_pattern:
        rx = _re.compile(spec.label_pattern)
        hits = [f for f in rows if rx.match(_norm_label(f["raw_label"]))]
        if hits and len({round(h["value"], 4) for h in hits}) == 1:
            tenk = [h for h in hits if h.get("form") == "10-K"]
            return (tenk or hits)[0], "found", []
        if hits:
            return None, "ambiguous", [f"{h['raw_label']} {h['period']}="
                                       f"{h['raw_cell_text']}" for h in hits]
    for label, sec in spec.section_labels:
        want = _norm_label(label)
        hits = [f for f in rows if _norm_label(f["raw_label"]) == want
                and sec in (f.get("section") or "").lower()]
        if len(hits) == 1 or (hits and len({round(h["value"], 4) for h in hits}) == 1):
            tenk = [h for h in hits if h.get("form") == "10-K"]
            return (tenk or hits)[0], "found", []
        if hits:
            return None, "ambiguous", [f"{h['raw_label']} {h['period']}="
                                       f"{h['raw_cell_text']}" for h in hits]
    return None, "not_found", []


# --------------------------------------------------------------------------- #
# 3. compute  (all arithmetic in code, never the LLM)
# --------------------------------------------------------------------------- #
def _fmt_money(fact: dict) -> str:
    if fact.get("is_dash"):             # a dash cell is "not reported", not $0
        return "—"
    if fact["scale"] == "units":        # EPS: dollars per share
        return f"${fact['value']:,.2f}"
    suffix = {"thousands": " thousand", "millions": " million",
              "billions": " billion"}.get(fact.get("scale"), "")
    # Match the filing's OWN precision: a filer that prints "$10,229.9 million"
    # (Vertiv, Intuitive) must have the displayed figure verbatim in the cited
    # cell — rounding to whole units would break grounding and make a stated
    # growth % irreconcilable with the stated numbers.
    raw = str(fact.get("raw_cell_text", ""))
    dec = 0
    if "." in raw:
        dec = min(len(raw.split(".")[-1].strip(" )%")), 4)
    return f"${fact['value']:,.{dec}f}{suffix}"


# --------------------------------------------------------------------------- #
# 4. answer  (orchestrate + gate + phrase)
# --------------------------------------------------------------------------- #
# A follow-up signal: something factual-shaped for the frame to complete.
_FOLLOWUP_RE = _re.compile(
    r"\b(?:this|that|it|its|those|these|them|same|entire|whole|overall|cumulative|"
    r"instead|also|again|too|each|every|per\s+year|by\s+year|year\s+by\s+year|"
    r"breakdown|break\s+it\s+down|what\s+about|how\s+about|and\s+for|and\s+in|"
    r"over\s+that|the\s+period|that\s+period|in\s+total|all|trend|history|over\s+time|"
    r"last\s+quarter|prior\s+year|last\s+year|year\s+before|year\s+earlier)\b|"
    r"^\W*(?:ok(?:ay)?[,\s]+)?and\b")


def _frame_desc(q) -> str:
    suffix = " margin" if q.calc == "margin" else ""
    names = ", ".join((METRICS[m].name if m in METRICS else COMPUTED[m][0]) + suffix
                      for m in q.metrics if m in METRICS or m in COMPUTED)
    ys = sorted(set(q.years))
    pre = ("FY" if q.period_kind == "annual" else
           f"Q{q.quarter} " if q.period_kind == "quarter" and q.quarter else
           f"{q.quarter * 3}-month YTD " if q.period_kind == "ytd" and q.quarter else "")
    span = (f"{pre}{ys[0]}–{pre}{ys[-1]}" if len(ys) > 1 else f"{pre}{ys[0]}") if ys else ""
    return names + (f", {span}" if span else "")


def _lc(name: str) -> str:
    """Lower-case a metric name for mid-sentence use, keeping acronyms (EBITDA)."""
    return name if name[:2].isupper() else name[:1].lower() + name[1:]


_WHY_RE = _re.compile(r"\b(?:why|what\s+drove|what\s+caused|reasons?\s+for|drivers?\s+of|"
                      r"how\s+come|explain)\b")


def _carry_why(q, ctx, tr) -> None:
    """"why did it fall?" after a figure: the question is about the PREVIOUS
    metric's narrative — send it to MD&A retrieval with that metric named."""
    if ctx is None or not getattr(ctx, "metrics", None) or q.metrics or q.untracked:
        return
    rl = _norm_text(q.raw_question)
    if not _WHY_RE.search(rl) or len(rl.split()) > 15:
        return
    m = ctx.metrics[0]
    nm = METRICS[m].name if m in METRICS else COMPUTED[m][0]
    q.metrics, q.metric, q.kind = list(ctx.metrics), m, "prose"
    q.prose_hint, q.followup = nm.lower(), True
    if not q.years and not q.legs:
        q.years, q.legs = list(ctx.years), list(ctx.legs)
        q.period_kind, q.quarter = ctx.period_kind, ctx.quarter
    tr.steps.append(f"follow-up: 'why' about {nm} -> management's narrative (MD&A)")


def _carry_periods(q, ctx, tr) -> None:
    """New metric, no period named ("what about net income?", "and EPS?"): keep
    the previous question's periods — and its calculation when the phrasing is
    an explicit follow-up — instead of silently jumping to the latest year."""
    if ctx is None or not getattr(ctx, "metrics", None) or q.untracked:
        return
    if not q.metrics or q.years or q.legs or q.kind == "prose":
        return                      # a carried "why" stays a narrative question
    rl = _norm_text(q.raw_question)
    cue = _re.search(r"^\W*(?:ok(?:ay)?[,\s]+)?(?:what|how)\s+about\b|^\W*(?:ok(?:ay)?[,\s]+)?and\b|"
                     r"\bsame\b|\binstead\b|\btoo\b|\balso\b", rl)
    if not cue:
        return                      # a fresh question: no stale periods
    q.years, q.legs = list(ctx.years), list(ctx.legs)
    q.period_kind, q.quarter = ctx.period_kind, ctx.quarter
    if cue and q.calc == "none" and ctx.calc == "growth":
        q.calc = ctx.calc           # "and net income?" after a growth question
        q._calc_carried = True
    q.kind, q.followup = "numeric", True
    tr.steps.append(f"follow-up: kept the previous periods {q.years} (calc={q.calc})")


_ALL_YEARS_RE = _re.compile(r"\b(?:each|every|all)\s+(?:the\s+)?(?:fiscal\s+)?years?\b|"
                            r"\ball\s+(?:the\s+)?years\b|\bentire\s+period\b|\bwhole\s+period\b|"
                            r"\btrend\w*\b|\bover\s+time\b|\bhistory\b|\bhistorical\w*\b|"
                            r"\byear\s+by\s+year\b|\bby\s+year\b|\bper\s+year\b")
_LAST_Q_RE = _re.compile(r"\b(?:last|latest|most\s+recent|recent|current)\s+quarter\b")
_PRIOR_RE = _re.compile(r"\b(?:same\s+(?:quarter|period)\s+(?:last|a|the\s+prior|the\s+previous)\s+year|"
                        r"(?:prior|previous|last)\s+year|a\s+year\s+(?:earlier|before|ago)|"
                        r"year[\s-]+ago)\b")
_YEAR_BEFORE_RE = _re.compile(r"\b(?:the\s+)?year\s+before(?:\s+that)?\b|\bprevious\s+one\b")
_COMPARE_TO_RE = _re.compile(r"\b(?:compare[ds]?|vs\.?|versus|against|relative\s+to)\b")


def _relative_periods(q, ctx, ent, tr) -> None:
    """Relative period phrases resolved deterministically against the bank and
    the previous frame — never left to silently keep a stale period:
      * "each year / all years / entire period / trend"  -> every banked fiscal year
      * "last / most recent quarter"                     -> the latest banked quarter
      * "same quarter last year / prior year"            -> the frame's period, one year back
        (with a comparison word: both periods, as growth)
      * "the year before that"                           -> one year before the frame
      * "compare that to 2023"                           -> frame years + the new year, growth"""
    if not q.metrics:
        return
    rl = _norm_text(q.raw_question)
    mk = q.metrics[0]
    stmt_kind = "annual"
    explicit_years = sorted({y for y in _re.findall(r"\b((?:19|20)\d{2})\b", rl)} | set())
    explicit_years = [int(y) for y in explicit_years]
    frame_years = sorted(set(getattr(ctx, "years", []) or [])) if ctx is not None else []
    if _ALL_YEARS_RE.search(rl) and not explicit_years:
        ys = sorted(_default_year(ent, stmt_kind, 20))
        if ys:
            q.years, q.legs, q.period_kind, q.quarter = ys, [], "annual", None
            if _re.search(r"\b(?:trend\w*|entire|whole|growth|grow\w*|change\w*)\b", rl):
                q.calc = "growth" if q.calc in ("none", "growth") else q.calc
            elif q.calc == "growth" and not _EXPLICIT_GROWTH_RE.search(rl):
                q.calc = "none" if not getattr(q, "_calc_carried", False) else q.calc
            tr.steps.append(f"relative period: all banked years {ys}")
        return
    if _LAST_Q_RE.search(rl) and not explicit_years:
        qs = sorted({(f["fiscal_year"], f.get("fiscal_quarter")) for f in bank.facts_for(ent.cik)
                     if f.get("period_kind") == "quarter" and f.get("fiscal_quarter")})
        if not qs:
            q.years, q.legs, q.period_kind, q.quarter = [], [], "quarter", None
            q._no_quarters = True
            tr.steps.append("relative period: latest quarter — none banked")
            return
        y, fq = qs[-1]
        if _re.search(r"\b(?:latest|last|most\s+recent|full)\s+(?:fiscal\s+)?year\b|\bannual\b", rl):
            ya = (_default_year(ent, "annual") or [None])[0]
            if ya:                     # "latest quarter AND latest fiscal year": two legs
                q.legs = [Leg("quarter", y, fq, explicit=True), Leg("annual", ya, explicit=True)]
                q.years = sorted({y, ya})
                tr.steps.append(f"relative period: Q{fq} {y} and FY{ya}")
                return
        q.years, q.legs, q.period_kind, q.quarter = [y], [], "quarter", fq
        tr.steps.append(f"relative period: latest banked quarter Q{fq} {y}")
        return
    if ctx is not None and frame_years and not explicit_years:
        if _YEAR_BEFORE_RE.search(rl) and (_EXPLICIT_GROWTH_RE.search(rl) or _COMPARE_TO_RE.search(rl)
                                           or _re.search(r"\bfrom\s+the\s+year\s+before\b", rl)):
            last = frame_years[-1]
            q.years, q.legs, q.calc = [last - 1, last], [], "growth"
            q.period_kind, q.quarter = ctx.period_kind, ctx.quarter
            q._calc_carried = True
            tr.steps.append(f"relative period: {last} compared with the year before")
            return
        if _YEAR_BEFORE_RE.search(rl):
            q.years, q.legs = [frame_years[0] - 1], []
            q.period_kind, q.quarter = ctx.period_kind, ctx.quarter
            if q.calc == "growth" and not _EXPLICIT_GROWTH_RE.search(rl):
                q.calc = "none"
            tr.steps.append(f"relative period: the year before {frame_years[0]}")
            return
        if _PRIOR_RE.search(rl):
            prior = [y - 1 for y in frame_years]
            q.period_kind, q.quarter, q.legs = ctx.period_kind, ctx.quarter, []
            if _COMPARE_TO_RE.search(rl) or _EXPLICIT_GROWTH_RE.search(rl):
                q.years, q.calc = sorted(set(prior) | set(frame_years)), "growth"
            else:
                q.years = prior
                if q.calc == "growth" and not _EXPLICIT_GROWTH_RE.search(rl):
                    q.calc = "none"
            tr.steps.append(f"relative period: prior year {q.years}")
            return
    if ctx is not None and frame_years and explicit_years and _COMPARE_TO_RE.search(rl) \
            and q.followup and not set(explicit_years) <= set(frame_years):
        if q.legs and any(lg.explicit and lg.kind != ctx.period_kind for lg in q.legs):
            # "compare that (6-month YTD) to the 2025 full year": both legs keep
            # their own kind, so the mixed comparison is refused, not coerced
            q.legs = [Leg(ctx.period_kind, y, ctx.quarter, explicit=True)
                      for y in frame_years] + list(q.legs)
            q.years = sorted(set(frame_years) | set(explicit_years))
            q.calc = "growth"
            tr.steps.append("relative period: compared across period kinds")
            return
        q.years = sorted(set(frame_years) | set(explicit_years))
        q.legs, q.period_kind, q.quarter = [], ctx.period_kind, ctx.quarter
        q.calc = "growth" if q.calc == "none" else q.calc
        tr.steps.append(f"relative period: compared with the previous frame -> {q.years}")
        return
    # fresh "X in 2026 vs the prior year": both periods, as growth
    if len(explicit_years) == 1 and _PRIOR_RE.search(rl) and _COMPARE_TO_RE.search(rl + " vs"):
        y = explicit_years[0]
        q.years, q.calc = [y - 1, y], "growth"
        q._calc_carried = True
        tr.steps.append(f"relative period: {y} vs the prior year")


def _merge_frame(q, ctx, tr) -> bool:
    """Slot-filling, not keyword-matching: a question that names NO metric — and
    would otherwise get "which metric?" — inherits the previous factual frame
    (metric, periods, calc) for every slot it leaves empty. Requires a factual
    signal (a calc, a period, or a follow-up word) so "hmm, interesting" doesn't
    replay the last figure. Returns True when something was carried."""
    if ctx is None or not getattr(ctx, "metrics", None) or q.metrics or q.untracked:
        return False
    rl = _norm_text(q.raw_question)
    if len(rl.split()) > 25:
        return False
    if not (q.calc != "none" or q.years or q.legs or _FOLLOWUP_RE.search(rl)):
        return False
    q.metrics = list(ctx.metrics)
    q.metric = q.metrics[0]
    if not q.years and not q.legs:
        q.years, q.legs = list(ctx.years), list(ctx.legs)
        q.period_kind, q.quarter = ctx.period_kind, ctx.quarter
    if q.calc == "none":
        q.calc = ctx.calc
        q._calc_carried = ctx.calc != "none"
    q.kind, q.followup = "numeric", True
    tr.steps.append(f"follow-up: carried metric {q.metrics}, periods {q.years}, "
                    f"calc={q.calc} from the previous question")
    return True


_ROLE_TAG = _re.compile(r"(?i)(?:^|(?<=[.!?\n]))\s*(?:system|assistant|user|human|ai)\s*(?:override)?\s*:\s*")
_HIDDEN = _re.compile(r"<!--.*?-->|<[^>]{1,200}>", _re.S)


def _sanitize(question: str) -> str:
    """Drop markup a user can't see and forged role labels ("system: …",
    "Assistant: …") so they can neither steer the lane nor pose as a prior turn.
    What remains is still treated as data, never as instructions."""
    q = _HIDDEN.sub(" ", question or "")
    q = _ROLE_TAG.sub(" ", q)
    q = _re.sub(r"(?i)\bsystem\s+override\b|\bverification\s+disabled\b", " ", q)
    return _re.sub(r"\s+", " ", q).strip()


_TYPO_JOINS = {"cashflow": "cash flow", "cashflows": "cash flows", "netincome": "net income",
               "freecashflow": "free cash flow", "grossprofit": "gross profit",
               "opex": "opex", "capex": "capex"}


def _cue_vocab() -> set:
    words = set()
    for cues in list(_METRIC_CUES.values()) + list(_MARGIN_CUES.values()) + list(_CALC_CUES.values()):
        for c in cues:
            words |= {w for w in _re.findall(r"[a-z]+", c) if len(w) >= 6}
    # only DISTINCTIVE metric words (6+ letters) are correction targets, so an
    # ordinary word ("cover", "year") is never pulled toward a short cue ("over")
    return words | {"revenues", "quarterly", "earnings", "expenses", "dividend"}


_REAL_WORDS = None


def _real_words() -> set:
    """Every word that occurs in the banked filings' own prose (~4 MB of English)
    plus their line-item labels: a word found there is a real word, never a typo."""
    global _REAL_WORDS
    if _REAL_WORDS is None:
        words = set()
        try:
            for c in bank.load_prose():
                words |= set(_re.findall(r"[a-z]{4,}", c.get("text", "").lower()))
            for f in bank.load_facts():
                words |= set(_re.findall(r"[a-z]{4,}", f.get("raw_label", "").lower()))
        except Exception:
            pass
        _REAL_WORDS = words
    return _REAL_WORDS


def _fix_typos(question: str):
    """Correct a misspelled METRIC word ("operrating income", "reveune", "net
    incme") toward the app's own cue vocabulary — only for a close, unique match,
    and never silently: returns (fixed question, [(typed, read_as), ...]) so the
    answer can say what it assumed."""
    import difflib
    vocab = _cue_vocab()
    keep = vocab | _FUNCTION_WORDS | _alias_words() | {
        "incoming", "operator", "operators", "marginal", "profile", "profiles",
        "covered", "current", "currency", "capital", "income", "equity"}
    fixes = []

    def sub(m):
        w = m.group(0)
        lw = w.lower()
        if lw in _TYPO_JOINS and _TYPO_JOINS[lw] != lw:
            fixes.append((w, _TYPO_JOINS[lw]))
            return _TYPO_JOINS[lw]
        if len(lw) < 5 or lw in keep or not lw.isalpha() or w[:1].isupper():
            return w                        # proper nouns (Apple, Delta) are names, not typos
        if lw in _real_words():
            return w                        # a real word ("internet", "repurchases")
        cands = difflib.get_close_matches(lw, sorted(vocab), n=2, cutoff=0.84)
        if not cands:
            return w
        best = difflib.SequenceMatcher(None, lw, cands[0]).ratio()
        if len(cands) > 1 and difflib.SequenceMatcher(None, lw, cands[1]).ratio() >= best - 0.03:
            return w                        # two near-equal readings: don't guess
        fixes.append((w, cands[0]))
        return cands[0]
    fixed = _re.sub(r"[A-Za-z]+", sub, question or "")
    return fixed, fixes


_MISS_LANES = {"capability", "unexplained", "need_metric", "clarify", "need_company",
               "unbanked_company", "untracked"}


def answer(question: str, entity=None, context=None, _spell_retry: bool = True) -> Answer:
    """Answer one question. `context` is the previous answer's `.query` (the UI
    passes it) so a follow-up can inherit its metric and periods.

    If the question wasn't understood, an LLM spell-check may propose a corrected
    question; voice.spellfix accepts it only as a word-for-word spelling fix, the
    corrected question is answered by the same code path, and the reply says what
    it read."""
    first = _answer_once(question, entity, context)
    lanes = [s_[6:] for s_ in first.trace.steps if s_.startswith("lane: ")]
    if not (_spell_retry and llm.available() and lanes and lanes[-1] in _MISS_LANES):
        return first
    prop = voice.spellfix(_sanitize(question))
    if not prop:
        return first
    fixed, changes = prop
    second = _answer_once(fixed, entity, context)
    lanes2 = [s_[6:] for s_ in second.trace.steps if s_.startswith("lane: ")]
    if lanes2 and lanes2[-1] in _MISS_LANES and second.status == first.status:
        return first                  # the spelling fix didn't help: keep the original
    shown = ", ".join(f"“{t}” as “{r}”" for t, r in changes)
    second.text += f" (I read {shown}.)"
    second.trace.steps.append(f"spelling (LLM, verified word-for-word): {changes}")
    return second


def _answer_once(question: str, entity=None, context=None) -> Answer:
    question = _sanitize(question)
    question, typo_fixes = _fix_typos(question)
    holder = {}
    a = _answer(question, entity, context, holder)
    a.query = holder.get("q")
    lanes = [s_[6:] for s_ in a.trace.steps if s_.startswith("lane: ")]
    converse = any(s_.startswith("converse lane") for s_ in a.trace.steps)
    if (a.status in ("refused", "hedged") and not a.trace.citations and not a.passages
            and not converse and llm.available()):
        years = " ".join(_re.findall(r"\b(?:19|20)\d{2}\b", question or ""))
        a.text = voice.render("refusal" if a.status == "refused" else "hedge",
                              a.text, question, trace=a.trace, allowed_text=years)
    if holder.get("typed_company"):
        typo_fixes = list(typo_fixes) + [holder["typed_company"]]
    if typo_fixes and a.status != "refused" and not converse:
        shown = ", ".join(f"“{t}” as “{r}”" for t, r in typo_fixes[:3])
        a.text += f" (I read {shown}.)"
        a.trace.steps.append(f"spelling: read {typo_fixes}")
    return a


def _answer(question: str, entity, context, holder) -> Answer:
    """`entity`, when supplied by the caller (the UI resolves the company
    deterministically from the question or its picker), is used directly — the
    engine does NOT re-identify the company from the text. This keeps follow-ups
    like "what is its revenue?" anchored to the right company instead of relying
    on the LLM to re-extract it. When entity is None, fall back to resolving the
    company from the question's mention.

    Control flow is a fixed precedence of LANES (see _pre_lane / _post_lane):
    converse lanes that need no company, then company resolution, then the
    company lanes, then the deterministic factual spine."""
    q = understand(question)
    holder["q"] = q
    tr = Trace()
    rawq = q.raw_question or ""
    _resolve_metrics(q, entity)                      # allowlist: default-refuse
    if q.kind == "prose" and not _has_prose_cue(rawq, bool(q.metrics)):
        q.kind = "numeric"                           # don't trust an LLM 'prose' label
    q.legs = _parse_legs(rawq)
    if q.legs and not _legs_hetero(q.legs):
        lg = q.legs[0]
        if lg.kind != "annual" or lg.explicit:     # a bare year keeps q's own kind
            q.period_kind, q.quarter = lg.kind, lg.quarter
        q._end_month = lg.end_month
    leg_years = [lg.year for lg in q.legs if lg.year]
    if leg_years and not set(leg_years) <= set(q.years):
        q.years = _expand_year_range(rawq, sorted(set(q.years) | set(leg_years)))
    _rl = rawq.lower()
    q.items = _requested_items(q)
    if q.items:
        tr.steps.append(f"asked items: {q.items}")
    if q.calc == "margin" and len(q.metrics or []) > 1:
        # which metrics does "margin" actually modify? the cue followed by "margin"
        sp = _cue_spans(_norm_text(rawq))
        tg = {m_ for m_ in q.metrics if m_ in sp and
              _re.match(r"\s*(?:\w+\s+)?margins?\b", _norm_text(rawq)[sp[m_][1]:])
              or (m_ in sp and "margin" in _norm_text(rawq)[sp[m_][0]:sp[m_][1]])}
        if tg:
            q.margin_metrics = sorted(tg)
    if _re.search(r"(?:\b(?:as\s+(?:a\s+)?)?(?:percent(?:age)?|share)|%)\s+of\s+(?:total\s+)?"
                  r"(?:revenues?|sales)\b", _rl) and q.metrics:
        num = [m for m in q.metrics if m != "revenue"]
        if num:
            q.metrics, q.metric, q.calc = num, num[0], "margin"
            q.pct_of_revenue = True
    # "EPS ..., diluted": a trailing qualifier still selects diluted
    if ("diluted" in _rl and "basic" not in _rl and "eps_basic" in (q.metrics or [])):
        q.metrics = ["eps_diluted" if m == "eps_basic" else m for m in q.metrics]
        q.metric = q.metrics[0]
    if ("basic" in _rl and "diluted" in _rl and _re.search(r"\beps\b|per\s+share", _rl)
            and q.metrics and set(q.metrics) & {"eps_basic", "eps_diluted"}):
        q.metrics = ["eps_basic", "eps_diluted"] + [
            m for m in q.metrics if m not in ("eps_basic", "eps_diluted")]
        q.metric = q.metrics[0]
    tr.steps.append(f"understood: metric={q.metric} years={q.years} "
                    f"calc={q.calc} kind={q.kind} company='{q.company_mention}'"
                    + (f" untracked={[n for _, n in q.untracked]}" if q.untracked else "")
                    + (f" legs={[(lg.kind, lg.year, lg.quarter) for lg in q.legs]}"
                       if _legs_hetero(q.legs) else ""))
    named = _companies_named(rawq)
    if named and getattr(named[0], "_typed", None):
        holder["typed_company"] = (named[0]._typed, named[0].title)
    if (context is not None and getattr(context, "_cik", None) and named
            and named[0].cik != context._cik and _COMPARE_TO_RE.search(_norm_text(rawq))):
        tr.steps.append("lane: multi_company")
        return Answer(f"That compares two companies. I answer one company at a time, "
                      f"each from its own filing — ask the same question for "
                      f"{named[0].title} on its own and I'll pull it.", 0.2, "hedged", tr)
    _carry_why(q, context, tr)
    _carry_periods(q, context, tr)

    # --- lanes that need no company ------------------------------------------
    pre = _pre_lane(q, entity, named, context)
    if pre:
        lane, payload = pre
        tr.steps.append(f"lane: {lane}")
        if lane == "companies":
            return _answer_companies(tr)
        if lane == "explain":
            return _answer_definition(payload, entity, tr)
        if lane == "judgment":
            return _answer_judgment(q, entity, context, tr)
        return _chat(q, entity, lane, tr)

    # --- a company is REQUIRED from here on ----------------------------------
    ent, early = _resolve_company(q, entity, named, tr)
    if early is not None:
        return early
    tr.steps.append(f"company: {ent.title} (CIK {ent.cik})")
    q._cik = ent.cik

    lane = _post_lane(q, ent)
    # the router would ask "which metric?" / fall back to a generic reply: let the
    # conversation frame fill the empty slots instead
    if lane in ("need_metric", "capability") and _merge_frame(q, context, tr):
        lane = _post_lane(q, ent)
    if lane in ("need_metric", "capability") and q.suggested_metric:
        lane = "clarify"
    tr.steps.append(f"lane: {lane}")
    if lane == "coverage":
        return _answer_coverage(ent, tr)
    if lane == "overview":
        return _answer_overview(ent, tr)
    if lane == "nongaap":
        return Answer(
            f"That's a non-GAAP / adjusted measure. I read only {ent.title}'s "
            "audited GAAP financial statements — company-reported non-GAAP figures "
            "(adjusted EBITDA, non-GAAP EPS, adjusted net income, etc.) live in "
            "MD&A or press-release reconciliations I don't parse, so I won't "
            "estimate one. I can give a GAAP figure instead — e.g. GAAP net "
            "income, GAAP EPS, operating margin — or a measure I derive from GAAP "
            "inputs and label as such: EBITDA (operating income + D&A) or free "
            "cash flow (operating cash flow − capex).",
            0.2, "hedged", tr)
    if lane == "prose":
        return _answer_prose(ent, q, tr)
    if lane == "untracked":
        # Something we deliberately don't bank as a line item. REFUSE — never let a
        # look-alike stand in (revenue for a cost line or a revenue component, EPS
        # for dividend-per-share, net income for retained earnings): silent-wrong-
        # metric is the failure class we exist to prevent.
        names = _untracked_names(q)
        near = (q.metrics or getattr(q, "_absorbed", []) or [None])[0]
        if any(n_.startswith("“") for _s, n_ in q.untracked):
            near = None                # another company's figure: no Tesla stand-in
            nxt = (" If you meant another company, I don't have it banked — name one "
                   "I cover and I'll pull its figures.")
        elif near:               # the nearest TRACKED figure, offered — never substituted
            nm = METRICS[near].name if near in METRICS else COMPUTED[near][0]
            nxt = f" If the company-wide {_lc(nm)} would help, just ask for it."
        else:
            nxt = (" I can give revenue, net income, cash flow, margins, EPS or growth "
                   "over time instead.")
        return Answer(
            f"I don't track {names} for {ent.title}, and I won't substitute a "
            f"different line item for it.{nxt}", 0.0, "refused", tr)
    if lane == "clarify":
        mk = q.suggested_metric
        nm = METRICS[mk].name if mk in METRICS else COMPUTED[mk][0]
        return Answer(
            f"Just to be sure before I pull a number — do you mean {ent.title}'s "
            f"{_lc(nm)}? Ask for it by name (and a year, if you have one in mind) and "
            "I'll get it straight from the filing.", 0.3, "hedged", tr)
    if lane == "need_metric":
        return Answer(
            f"The {q.calc} of which metric for {ent.title}? Name one — e.g. "
            "revenue, net income, operating income, EPS, or free cash flow — "
            "and I'll compute it from the filings.", 0.2, "hedged", tr)
    if lane == "capability":
        return _chat(q, ent, "capability", tr)
    if lane == "badperiod":
        # a period we don't carry (TTM, Q5/Q0) must hedge, not fall back to annual
        return Answer(
            f"I report periods exactly as {ent.title} files them — full fiscal "
            "years, plus the quarters and year-to-date in the 10-Q. I don't carry "
            "a trailing-twelve-month figure or an out-of-range quarter, so I won't "
            "guess one. Ask for a fiscal year or a quarter (Q1–Q4).",
            0.2, "hedged", tr)
    if lane == "unexplained":
        left = _unexplained_tokens(q, ent, named)
        shown = ", ".join(f"“{w}”" for w in left[:3])
        tr.steps.append(f"unexplained words: {left}")
        nm = (METRICS[q.metrics[0]].name if q.metrics and q.metrics[0] in METRICS
              else "figure")
        return Answer(
            f"I'm not sure what {shown} refers to there, so I won't guess. I report "
            f"{ent.title}'s company-wide figures as filed — not segments, products, "
            f"regions or other companies. If you meant the company-wide {_lc(nm)}, "
            "ask for it directly.", 0.0, "refused", tr)
    _relative_periods(q, context, ent, tr)
    if getattr(q, "_no_quarters", False):
        return Answer(f"I don't have quarterly figures banked for {ent.title} — only "
                      f"its annual filings." + _offer(ent, q.metrics[0]), 0.0, "refused", tr)
    a = _answer_factual(ent, q, tr)
    if q.followup and a.status != "refused" and not a.text.startswith("Yes"):
        a.text = f"Following up on your last question ({_frame_desc(q)}): " + a.text
    if q.untracked and a.status != "refused":
        a.text += (f" (I don't track {_untracked_names(q)} as a separate figure, so "
                   "I left that out.)")
        a.status = "hedged"              # partial: not everything asked was answered
    return a


_COMPARE_RE = _re.compile(r"\b(?:compare[ds]?|comparison|vs\.?|versus|against|"
                          r"relative\s+to|than|compared)\b", _re.I)


def _unnamed_rival(rawq: str, ent):
    """"Compare Tesla and Ford revenue": a comparison naming a second, unbanked
    company we can't resolve exactly. Returns that name, or None."""
    if not _COMPARE_RE.search(rawq or ""):
        return None
    own = {w.lower() for w in _re.findall(r"[A-Za-z]+", ent.title)} | {ent.ticker.lower()}
    own |= {a for a, tk in _ENT_ALIASES.items() if tk == ent.ticker}
    for m in _re.finditer(r"\b(?:vs\.?|versus|against|than|with|and|to)\s+"
                          r"([A-Z][A-Za-z&.]+)(?:'s)?\b", rawq or ""):
        tok = m.group(1)
        if tok in _MENTION_STOP or (tok.upper() in _ACRONYM_STOP and tok != "GM"):
            continue
        if tok.lower() in own or tok.lower() in _alias_words():
            continue
        return tok
    return None


def _resolve_company(q, entity, named, tr):
    """(entity, early_answer). Banked-aware: a company the question names wins
    over a stray capitalized word; two companies, or a company that conflicts
    with the one in focus, hedge rather than answer the wrong company."""
    rawq = q.raw_question or ""
    vague = _is_vague(q)
    if len({e.cik for e in named}) > 1:
        names = " and ".join(f"{e.title} ({e.ticker})" for e in named[:3])
        tr.steps.append("lane: multi_company")
        return None, Answer(
            f"That names more than one company ({names}). I answer one company at a "
            "time, each grounded in its own filing — ask about each separately and "
            "I'll pull both.", 0.2, "hedged", tr)
    if entity is not None:
        if named and named[0].cik != entity.cik:
            # the question names a different banked company: follow the question
            # (as the UI does), and say so in the trace — never answer with the
            # figures of the company that merely happened to be in focus.
            tr.steps.append(f"question names {named[0].ticker}; switching from "
                            f"{entity.ticker}")
            return named[0], None
        rival = _unnamed_rival(rawq, entity)
        if rival:
            tr.steps.append("lane: multi_company")
            return None, Answer(
                f"That compares {entity.title} with {rival}. I answer one company at a "
                f"time from its own filings, and I don't have {rival} banked — ask "
                f"about {entity.ticker} on its own and I'll pull it.", 0.2, "hedged", tr)
        if not named:
            other = _foreign_company(rawq, entity)
            if other is not None:
                tr.steps.append("lane: company_conflict")
                return None, Answer(
                    f"You asked about {other.title}, which I don't have in my banked "
                    f"filings — and I won't answer it with {entity.title}'s figures. "
                    "Name a company I cover (or pick it on the left).",
                    0.0, "refused", tr)
        return entity, None
    if named:
        rival = _unnamed_rival(rawq, named[0])
        if rival:
            tr.steps.append("lane: multi_company")
            return None, Answer(
                f"That compares {named[0].title} with {rival}. I answer one company "
                f"at a time from its own filings, and I don't have {rival} banked — "
                f"ask about {named[0].ticker} on its own and I'll pull it.",
                0.2, "hedged", tr)
        return named[0], None
    cands, status = edgar.resolve_entity(q.company_mention or "")
    # BANKED-ONLY anchoring: we never anchor to (or claim to cover) an arbitrary
    # SEC filer a stray word matched ("GM"/"Morning"/"are you an AI?").
    banked = [c for c in cands if bank.facts_for(c.cik)]
    m = (q.company_mention or "").strip()
    if status == "ambiguous" and not banked and m and (
            not vague or _re.search(r"\b(?:cover|have|support|track|include|know)\b", rawq.lower())):
        tr.steps.append("lane: unbanked_company")
        return None, Answer(f"I don't have {m} in my banked filings, so I can't answer "
                            "that. I cover a fixed set of US public companies — name one "
                            "of those (or pick it on the left).", 0.0, "refused", tr)
    if status == "ambiguous":                          # never silent-pick
        pool = banked or cands
        names = ", ".join(f"{c.title} ({c.ticker})" for c in pool[:5])
        tr.steps.append("lane: ambiguous_company")
        return None, Answer(f"That company name is ambiguous — did you mean one of: "
                            f"{names}? I won't pick silently.", 0.2, "hedged", tr)
    if len(banked) == 1:
        return banked[0], None
    if not vague and m:
        # a company was NAMED in a factual question but isn't banked -> honest miss
        nm = cands[0].title if cands else m
        tr.steps.append("lane: unbanked_company")
        return None, Answer(f"I don't have {nm} in my banked filings, so I can't "
                            "answer that. I cover a set of US public companies — name "
                            "one of those (or pick it on the left).", 0.0, "refused", tr)
    if _untracked_is_primary(q):
        tr.steps.append("lane: untracked")
        return None, Answer(
            f"I don't track {_untracked_names(q)} for any of the companies I cover, "
            "and I won't substitute a different line item for it. I can give revenue, "
            "net income, operating income, operating expenses, EPS, cash, cash flow, "
            "margins, EBITDA, or growth over time — name a company.",
            0.0, "refused", tr)
    tr.steps.append("lane: need_company")
    return None, _chat(q, None, "need_company", tr)


def _nongaap_blocked(q) -> bool:
    """Two senses of non-GAAP must not be conflated:
     (a) a COMPANY-REPORTED adjusted figure (adjusted EBITDA, non-GAAP EPS) —
         from MD&A / press-release reconciliations we don't parse;
     (b) a measure we DERIVE from GAAP inputs (FCF, EBITDA) — formally non-GAAP,
         but every input is an audited line we ground, cite and label.
    An explicit ADJUSTMENT modifier always means (a) and blocks; a bare
    "non-GAAP" mention blocks only when it targets a figure we can't derive."""
    q_low = (q.raw_question or "").lower().replace("-", " ")
    parsed_derivable = any(m in COMPUTED
                           for m in (q.metrics or ([q.metric] if q.metric else [])))
    derivable_targeted = parsed_derivable or _mentions_derivable(q.raw_question)
    wants_adjusted = any(c in q_low for c in _NONGAAP_ADJUST_CUES)
    mentions_nongaap = any(c in q_low for c in _NONGAAP_WORD_CUES)
    return wants_adjusted or (mentions_nongaap and not derivable_targeted)


def _latest_interim_quarter(ent, kind: str, year: int):
    qs = [f.get("fiscal_quarter") for f in bank.facts_for(ent.cik)
          if f.get("period_kind") == kind and f.get("fiscal_year") == year
          and f.get("fiscal_quarter")]
    return max(qs) if qs else None


def _default_year(ent, kind: str, n: int = 1) -> list:
    avail = sorted({f["fiscal_year"] for f in bank.facts_for(ent.cik)
                    if f.get("period_kind") == kind}, reverse=True)
    return avail[:n]


def _fy_end_month(ent):
    ends = {int(f["period_end"][5:7]) for f in bank.facts_for(ent.cik)
            if f.get("period_kind") == "annual" and f.get("period_end")
            and f.get("statement") in ("operations", "cash_flows")}
    return max(ends) if ends else 12


def _leg_query(ent, q, kind, quarter, year, end_month=None):
    """A per-period copy of the query: its own kind/quarter. A quarter named by
    its END MONTH ("quarter ended March 2026") becomes the company's FISCAL
    quarter (Disney's March quarter is fiscal Q2); an unspecified YTD/quarter
    resolves to the latest banked for that year — never a guessed one."""
    if kind == "quarter" and quarter is None and end_month:
        quarter = _fiscal_quarter_of(end_month, _fy_end_month(ent))
    if kind in ("ytd", "quarter") and quarter is None:
        quarter = _latest_interim_quarter(ent, kind, year)
    return replace(q, period_kind=kind, quarter=quarter)


def _fiscal_year_of(ent, end_month, cal_year):
    """Fiscal year containing a period that ends in (end_month, cal_year)."""
    fy_end = _fy_end_month(ent)
    return cal_year if end_month <= fy_end else cal_year + 1


def _period_key(p):
    qq, y = p
    return (y, {"annual": 2, "ytd": 1, "quarter": 0}.get(qq.period_kind, 3), qq.quarter or 0)


_MONTHS_FULL = ["", "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]
_LAST_N_RE = _re.compile(r"\b(?:last|past|previous|prior|recent)\s+(two|three|four|five|[2-9])"
                         r"\s+(?:fiscal\s+)?(?:years|quarters)\b")
_NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5}
_CALENDAR_RE = _re.compile(r"\bcalendar\b")
_YEAR_ENDED_RE = _re.compile(r"(?<!half\s)(?<!half-)\b(?:year|twelve\s+months|12\s+months)\s+end(?:ed|ing)\s+("
                             + "|".join(m.lower() for m in _MONTHS_FULL[1:]) + r")\b")


def _calendar_mismatch(ent, q, tr):
    """We report periods exactly as the company files them. A CALENDAR period, or
    a "year ended <month>" that isn't the company's fiscal year-end, is a period
    we'd have to re-cut — hedge instead of silently answering the fiscal one."""
    ql = (q.raw_question or "").lower()
    ends = {int(f["period_end"][5:7]) for f in bank.facts_for(ent.cik)
            if f.get("period_kind") == "annual" and f.get("period_end")}
    if not ends:
        return None
    fy_end = _MONTHS_FULL[max(ends)]
    m = _YEAR_ENDED_RE.search(ql)
    asked = None
    if _CALENDAR_RE.search(ql) and ends != {12}:
        asked = "a calendar period"
    elif m and _MONTHS_FULL.index(m.group(1).capitalize()) not in ends:
        asked = f"a year ended in {m.group(1).capitalize()}"
    if not asked:
        return None
    tr.steps.append(f"period mismatch: asked {asked}; fiscal year ends {fy_end}")
    return Answer(f"You asked for {asked}, but {ent.title}'s fiscal year ends in "
                  f"{fy_end}, and I report periods exactly as the company files them — "
                  "I won't re-cut figures into a different period. Ask for a fiscal "
                  "year or fiscal quarter and I'll pull it.", 0.2, "hedged", tr)


def _answer_factual(ent, q, tr) -> Answer:
    """The deterministic spine: periods -> per-period figures -> compose."""
    rawq = q.raw_question or ""
    mismatch = _calendar_mismatch(ent, q, tr)
    if mismatch is not None:
        return mismatch
    mks = [m for m in (q.metrics or ([q.metric] if q.metric else []))
           if m in METRICS or m in COMPUTED]
    hetero = _legs_hetero(q.legs)
    years = list(q.years)
    explicit_growth = bool(_EXPLICIT_GROWTH_RE.search(_norm_text(rawq)))
    # "basic vs diluted EPS" / "revenue versus net income in 2024": a comparison of
    # METRICS within one period is a side-by-side list, not growth.
    if (q.calc == "growth" and not explicit_growth and not hetero
            and not getattr(q, "_calc_carried", False)
            and (len(mks) > 1 or len(set(years)) <= 1)):
        q.calc = "margin" if _re.search(r"\bmargins?\b", rawq.lower()) else "none"
        tr.steps.append("metric-vs-metric in one period -> listed side by side")

    if hetero:
        # each leg answered in its OWN period kind — never one coerced kind
        periods = []
        for lg in q.legs:
            y = lg.year or (_default_year(ent, lg.kind) or [None])[0]
            if y is None:
                continue
            if lg.end_month and lg.year:
                y = _fiscal_year_of(ent, lg.end_month, lg.year)
            periods.append((_leg_query(ent, q, lg.kind, lg.quarter, y, lg.end_month), y))
        if not periods:
            return Answer(f"I don't have figures banked for {ent.title} for those "
                          "periods.", 0.0, "refused", tr)
        seen_p, uniq = set(), []
        for qq, y in periods:
            k = (qq.period_kind, qq.quarter, y)
            if k not in seen_p:
                seen_p.add(k)
                uniq.append((qq, y))
        periods = sorted(uniq, key=_period_key)
        tr.steps.append("per-leg periods: " + ", ".join(
            _period_tag(qq, y) for qq, y in periods))
        all_bs = all((m in METRICS and METRICS[m].statement == "balance_sheet")
                     or m == "current_ratio" for m in mks)
        mixed = len({qq.period_kind for qq, _ in periods}) > 1 and not all_bs
        if mixed and (q.calc == "growth" or _COMPARATIVE_RE.search(_norm_text(rawq))):
            # "did the margin IMPROVE from FY2025 to Q2 2026?" invites comparing a
            # quarter with a year — refuse rather than lay them side by side.
            return Answer(
                "That compares an interim period (a quarter / year-to-date) with a "
                "full fiscal year — mixing period kinds. I won't compare across them; "
                "ask for two comparable periods." + _offer(ent, mks[0]), 0.0, "refused", tr)
        if q.calc == "growth":
            if mixed:
                return Answer(
                    "That compares an interim period (a quarter / year-to-date) with "
                    "a full fiscal year — mixing period kinds. I won't compute growth "
                    "across them; ask for two comparable periods.", 0.0, "refused", tr)
            return _answer_growth(ent, q, periods, tr, mks)
        return _answer_multi(ent, q, periods, tr, mks)

    # one period kind: which years? default to the latest available
    m_last = _LAST_N_RE.search(_norm_text(rawq))
    if not years and m_last:
        n = _NUM_WORDS.get(m_last.group(1), None) or int(m_last.group(1))
        years = _default_year(ent, q.period_kind, n)
        tr.steps.append(f"'last {n}' -> {years}")
    if not years:
        years = _default_year(ent, q.period_kind, 2 if q.calc == "growth" else 1)
        tr.steps.append(f"no year given -> using {years} ({q.period_kind})")
    if not years:                       # nothing of this period kind is banked
        return Answer(f"I don't have {q.period_kind} figures banked for "
                      f"{ent.title} to answer that — try a specific year, or "
                      "ask about an annual period.", 0.0, "refused", tr)
    em = getattr(q, "_end_month", None)
    if em and q.period_kind == "quarter" and q.quarter is None:
        years = [_fiscal_year_of(ent, em, y) for y in years]
    periods = sorted({y: _leg_query(ent, q, q.period_kind, q.quarter, y, em)
                      for y in years}.items())
    periods = [(qq, y) for y, qq in periods]
    if not q.years:
        q.years = [y for _qq, y in periods]
        q.period_kind, q.quarter = periods[0][0].period_kind, periods[0][0].quarter
    if q.calc == "growth" and _re.search(r"\bmargins?\b", rawq.lower()) and len(periods) > 1:
        q.calc = "margin"                       # "gross margin 2024 vs 2025"
        periods = [(replace(qq, calc="margin"), y) for qq, y in periods]
        return _answer_multi(ent, q, periods, tr, mks, pp_change=True)
    if q.calc == "growth":
        if len(periods) == 1 and len(q.years) > 1:   # "from 2025 to 2025"
            return Answer("A growth question needs two distinct periods; you named "
                          "the same one twice.", 0.2, "hedged", tr)
        if len(periods) == 1:
            # single-period growth: infer the comparable PRIOR period (same quarter
            # a year earlier; the previous quarter when asked QoQ / sequentially)
            qq, y = periods[0]
            if qq.period_kind == "quarter" and qq.quarter and _QOQ_RE.search(_norm_text(rawq)):
                py, pq = (y, qq.quarter - 1) if qq.quarter > 1 else (y - 1, 4)
                prior = (replace(qq, quarter=pq), py)
            else:
                prior = (_leg_query(ent, q, qq.period_kind, q.quarter, y - 1), y - 1)
            periods = [prior, periods[0]]
            tr.steps.append(f"one period named -> compared with the prior period "
                            f"{_period_tag(*prior)}")
        return _answer_growth(ent, q, periods, tr, mks)
    # Several figures and/or several periods -> decompose into per-period atoms
    if q.items and q.calc != "growth":
        return _answer_multi(ent, q, periods, tr, mks)
    if mks and (len(mks) > 1 or len(periods) > 1):
        return _answer_multi(ent, q, periods, tr, mks)
    qq, y = periods[0]
    if q.calc == "margin":
        return _answer_margin(ent, qq, y, tr)
    if q.metric in COMPUTED:                  # derived ratios / EBITDA / FCF
        return _answer_computed(ent, qq, y, tr, q.metric)
    return _answer_level(ent, qq, y, tr)


def _period_desc(q, year: int) -> str:
    """Human phrase for the requested period (so the answer says *which* period
    the number is — never leaving annual/quarterly ambiguous)."""
    if q.period_kind == "quarter":
        return f"Q{q.quarter} {year}" if q.quarter else f"a {year} quarter"
    if q.period_kind == "ytd":
        if not q.quarter:
            return f"year-to-date {year}"
        return f"the {q.quarter * 3} months year-to-date in {year}"
    return f"{year}"


def _available_periods(ent, mk) -> str:
    """Which periods of metric `mk` are actually banked for this company — a
    deterministic next step for every "I don't have that period" reply (years and
    period names only; no figures)."""
    if mk in COMPUTED:
        mk = COMPUTED[mk][2]
    if mk not in METRICS:
        return ""
    spec, seen = METRICS[mk], []
    facts = [f for f in bank.facts_for(ent.cik) if f["statement"] == spec.statement]
    cands = sorted({(f.get("period_kind"), f.get("fiscal_year"), f.get("fiscal_quarter"))
                    for f in facts if f.get("fiscal_year")},
                   key=lambda t: (t[1], {"annual": 2, "ytd": 1, "quarter": 0, "instant": 2}
                                  .get(t[0], 3), t[2] or 0))
    annual, interim = [], []
    for pk, y, fq in cands:
        kind = "annual" if pk in ("annual", "instant") else pk
        if pk == "instant":
            ok = find_fact(ent.cik, mk, y, "annual", None)[1] == "found"
            if ok and y not in annual:
                annual.append(y)
            continue
        if find_fact(ent.cik, mk, y, kind, fq)[1] != "found":
            continue
        if kind == "annual":
            annual.append(y)
        elif kind == "quarter":
            interim.append(f"Q{fq} {y}")
        else:
            interim.append(f"{fq * 3}-month YTD {y}" if fq else f"YTD {y}")
    if annual:
        a = sorted(set(annual))
        seen.append(f"FY{a[0]}–FY{a[-1]}" if len(a) > 1 else f"FY{a[0]}")
    seen += list(dict.fromkeys(interim))
    return ", ".join(seen)


def _offer(ent, mk) -> str:
    av = _available_periods(ent, mk)
    if not av:
        return ""
    nm = METRICS[mk].name if mk in METRICS else COMPUTED[mk][0]
    return f" I do have {ent.title}'s {_lc(nm)} for {av} — want one of those?"


def _answer_level(ent, q, year, tr) -> Answer:
    fact, st, cand = find_fact(ent.cik, q.metric, year, q.period_kind, q.quarter, instant_end=q.instant_end)
    spec = METRICS[q.metric]
    desc = _period_desc(q, year)
    if st == "ambiguous":
        return Answer(f"The label for {spec.name} in {desc} matches more than "
                      f"one line item ({', '.join(cand)}). I won't pick one "
                      "silently — please disambiguate.", 0.3, "hedged", tr)
    if st == "not_found" or fact is None:
        return Answer(f"I don't have {spec.name} for {desc} in the banked "
                      f"filings for {ent.title}, so I won't fabricate it."
                      + _offer(ent, q.metric), 0.0, "refused", tr)
    if fact.get("is_dash"):
        # the filing prints a dash ("—") for this line — i.e. not reported / nil,
        # not a numeric zero. Say so rather than render a misleading "$0".
        tr.cite(fact)
        return Answer(f"{ent.title} reports a dash (“—”) for {spec.name} in {desc} "
                      "— i.e. no value, not zero — so there's nothing to report.",
                      0.3, "hedged", tr)
    tr.cite(fact)
    tr.steps.append(f"matched '{fact['raw_label']}' {fact['period']} = "
                    f"{fact['raw_cell_text']} (p{fact['page_no']})")
    val = _fmt_money(fact)
    if q.metric == "capex" and fact["value"] < 0:
        val = _fmt_money({**fact, "value": -fact["value"]}) + " (a cash outflow)"
    # the period label comes from the MATCHED FACT (so a nine-month YTD is never
    # phrased as "six months", and a non-December fiscal quarter is named right).
    fdesc = _fact_desc(fact)
    # the net-income caveat applies ONLY when the grounded line really is the
    # "attributable to common stockholders" amount — not a bare "Net income" nor a
    # parent-company ("attributable to <Company>") attribution.
    caveat = ""
    if (spec.alt_note and "attributable to common" in _norm_label(fact["raw_label"])
            and not q.followup):
        caveat = f" Note: {spec.alt_note}."
    text = _phrase(q.raw_question,
                   f"{ent.title}'s {spec.name} "
                   f"{fdesc if fdesc.startswith(('as of', 'at fiscal')) else 'for ' + fdesc} was {val}." + caveat,
                   tr)
    return Answer(text, fact["confidence"], "answered", tr, fact["value"])


def _fact_tag(f: dict) -> str:
    """Short period tag taken from the MATCHED FACT (never the parsed query), so
    an interim figure is never printed as a bare year: FY2025 / Q3 2025 / 9-mo
    2025 / as-of date."""
    if not f:
        return "?"
    pk, y = f.get("period_kind"), f.get("fiscal_year")
    if pk == "quarter":
        return f"Q{f.get('fiscal_quarter')} {y}" if f.get("fiscal_quarter") else f"{y} quarter"
    if pk == "ytd":
        return f"{f.get('months')}-mo {y}" if f.get("months") else f"YTD {y}"
    if pk == "instant":
        if f.get("period_end_inferred"):
            return f"FY{y} year-end"
        return f"as of {f.get('period_end')}"
    return f"FY{y}"


def _fact_desc(f: dict) -> str:
    """Longer human period phrase from the matched fact (for a single answer)."""
    if not f:
        return "?"
    pk, y = f.get("period_kind"), f.get("fiscal_year")
    if pk == "quarter":
        return f"Q{f.get('fiscal_quarter')} {y}" if f.get("fiscal_quarter") else f"a {y} quarter"
    if pk == "ytd":
        m = f.get("months")
        return f"the {m} months year-to-date in {y}" if m else f"year-to-date {y}"
    if pk == "instant":
        if f.get("period_end_inferred"):
            return f"at fiscal year-end {y}"
        return f"as of {f.get('period_end')}"
    return f"{y}"


def _pct_change(old: float, new: float):
    """(display, numeric_or_None) for a period-over-period change. A zero base has
    no meaningful %, and a % off a NEGATIVE base flips sign/meaning (a widening
    loss looks like growth) — so both are flagged rather than printed as a clean
    percentage."""
    if old == 0:
        return "n/a (from a zero base)", None
    pct = (new - old) / old * 100.0
    if old < 0:
        return ("n/a (off a negative base — a % change from a loss would mislead)", None)
    return f"{pct:+.1f}%", pct


def _answer_growth(ent, q, periods, tr, metric_keys=None) -> Answer:
    """Growth as a consecutive period-over-period series (the L2 aggregate op)
    over `periods` — a chronological list of (per-period query, year), so annual
    YoY, same-quarter YoY and QoQ all go through one path.

    Handles levels AND computed metrics, several metrics at once, and CAGR. Every
    period label comes from the MATCHED FACT; zero/negative bases and mixed
    period kinds are handled honestly, not as a misleading %."""
    mks = [m for m in (metric_keys or ([q.metric] if q.metric else []))
           if m in METRICS or m in COMPUTED] or ([q.metric] if q.metric else [])
    if not mks or mks[0] is None:
        return Answer("Growth of which metric? Name one and I'll compute it.",
                      0.2, "hedged", tr)
    if len(periods) < 2:
        return Answer("A growth question needs two distinct periods; I only have "
                      "one.", 0.2, "hedged", tr)
    all_bs = all((m in METRICS and METRICS[m].statement == "balance_sheet")
                 or m == "current_ratio" for m in mks)
    if len({qq.period_kind for qq, _ in periods}) > 1 and not all_bs:
        return Answer("That compares an interim period (a quarter / year-to-date) "
                      "with a full fiscal year — mixing period kinds. I won't "
                      "compute growth across them; ask for two comparable periods.",
                      0.0, "refused", tr)
    idx = list(range(len(periods)))
    _ql = (q.raw_question or "").lower()
    cagr = any(w in _ql for w in ["cagr", "compound annual", "annualized",
                                  "annualised"])
    blocks, confs, last_val, headline = [], [], None, None
    _qn = _norm_text(q.raw_question)
    wants_whole = bool(_re.search(r"\b(?:entire|whole|overall|cumulative|in\s+total|"
                                  r"total\s+(?:change|growth))\b", _qn))
    steps_first = (not wants_whole and bool(_re.search(
        r"\b(?:yoy|y/y|year over year|each\s+year|every\s+year|per\s+year|by\s+year|"
        r"year\s+by\s+year|annual\s+growth|breakdown|break\s+it\s+down)\b", _qn)))
    for mk in mks:
        figs = {i: _metric_figure(ent, periods[i][0], periods[i][1], mk) for i in idx}
        nm = next((figs[i]["name"] for i in idx if figs[i].get("name")), str(mk))
        miss = [i for i in idx if not figs[i]["ok"]]
        if miss:
            blocks.append(("miss", nm, miss)); continue
        facts = [f for i in idx for f in figs[i]["facts"]]
        if len({f.get("scale") for f in facts}) > 1:
            blocks.append(("scalemix", nm, None)); continue
        if len({f.get("period_kind") for f in facts}) > 1 and not all_bs:
            blocks.append(("kindmix", nm, None)); continue
        blocks.append(("ok", nm, (figs, mk in COMPUTED)))

    oks = [b for b in blocks if b[0] == "ok"]
    if not oks:                                   # surface the specific refusal
        st, nm, extra = blocks[0]
        if st == "miss":
            where = ", ".join(_period_desc(*periods[i]) for i in extra)
            return Answer(f"I'm missing a grounded {_lc(nm)} value for {where}, "
                          "so I won't compute a growth rate." + _offer(ent, mks[0]),
                          0.0, "refused", tr)
        if st == "scalemix":
            return Answer("The periods use different units/scales; I won't compute "
                          "growth across them.", 0.0, "refused", tr)
        return Answer("Those figures cover different period kinds (mixing annual "
                      "and quarterly data), so I won't compute growth across them.",
                      0.0, "refused", tr)

    lines = []
    first, last = idx[0], idx[-1]
    for _st, nm, (figs, nongaap) in oks:
        for i in idx:
            for f in figs[i]["facts"]:
                tr.cite(f); confs.append(f.get("confidence", 0.9))
        fl = {i: (figs[i]["facts"][0] if figs[i]["facts"] else None) for i in idx}
        pl = lambda i: _fact_tag(fl[i])
        tag = " [non-GAAP, computed from GAAP inputs]" if nongaap else ""
        o_num, n_num = figs[first]["num"], figs[last]["num"]
        span = periods[last][1] - periods[first][1]
        annual = periods[first][0].period_kind == "annual"
        if cagr and annual and span >= 1 and o_num > 0 and n_num > 0:
            rate = ((n_num / o_num) ** (1.0 / span) - 1) * 100.0
            tr.steps.append(f"{nm} CAGR {figs[first]['value_str']} ({pl(first)}) -> "
                            f"{figs[last]['value_str']} ({pl(last)}) / {span}y "
                            f"= {rate:+.1f}%/yr")
            lines.append(f"{nm}: a {rate:+.1f}% compound annual growth rate from "
                         f"{figs[first]['value_str']} ({pl(first)}) to "
                         f"{figs[last]['value_str']} ({pl(last)}) over {span} "
                         f"years{tag}")
            last_val = rate
            continue
        steps = []
        for p, c in zip(idx, idx[1:]):
            disp, num = _pct_change(figs[p]["num"], figs[c]["num"])
            steps.append((p, c, disp, num))
            tr.steps.append(f"{nm} {figs[p]['value_str']} ({pl(p)}) -> "
                            f"{figs[c]['value_str']} ({pl(c)}) = {disp}")
        last_val = steps[-1][3]
        if len(steps) == 1:
            p, c, disp, _ = steps[0]
            lines.append(f"{nm} went from {figs[p]['value_str']} ({pl(p)}) to "
                         f"{figs[c]['value_str']} ({pl(c)}), a change of {disp}{tag}")
        else:
            # LEAD with the whole period (what "growth from 2023 to 2025" asks),
            # then the year-by-year steps as supporting detail.
            tdisp, tnum = _pct_change(o_num, n_num)
            fromto = (f"from {figs[first]['value_str']} ({pl(first)}) to "
                      f"{figs[last]['value_str']} ({pl(last)})")
            if tnum is None:      # zero / negative start: no meaningful % at all
                core_txt = (f"{fromto} — a % change isn't meaningful from a "
                            f"{'loss' if o_num < 0 else 'zero base'}")
            else:
                core_txt = f"{tdisp} {fromto}"
            if annual and span >= 1 and o_num > 0 and n_num > 0:
                rate = ((n_num / o_num) ** (1.0 / span) - 1) * 100.0
                core_txt += f" — a {rate:+.1f}% compound annual growth rate over {span} years"
                tr.steps.append(f"{nm} CAGR over {span}y = {rate:+.1f}%/yr")
            cagr_txt = core_txt[core_txt.find(" — a "):] if " — a " in core_txt and tnum is not None else ""
            whole = (f"{nm} went {core_txt}" if tnum is None
                     else f"{nm} changed {tdisp} over the whole period, {fromto}{cagr_txt}")
            tr.steps.append(f"{nm} whole period {pl(first)}->{pl(last)} = {tdisp}")
            stept = "; ".join(f"{pl(p)}→{pl(c)} {disp}" for p, c, disp, _ in steps)
            step_word = "Year over year" if annual else "Period over period"
            if steps_first:
                # "YoY growth 2023-2025" / "each year's growth": the steps ARE the
                # answer; the whole-period change follows as context.
                detail = "; ".join(f"{pl(p)}→{pl(c)} {disp} ({figs[p]['value_str']} → "
                                   f"{figs[c]['value_str']})" for p, c, disp, _ in steps)
                lines.append(f"{nm}, {step_word.lower()}: {detail}{tag}. Over the whole "
                             f"period: {core_txt}")
            else:
                lines.append(f"{whole}{tag}. {step_word}: {stept}")
            last_val = tnum
            if len(oks) == 1 and tnum is not None:
                headline = (f"{nm} · {pl(first)} → {pl(last)}", tdisp.split(" ")[0])

    body = f"{ent.title}'s " + "; ".join(lines) + "."
    skipped = [nm for st, nm, _ in blocks if st != "ok"]
    if skipped:
        body += f" (I couldn't ground: {', '.join(skipped)}.)"
    if q.followup and _re.search(r"\b(?:yoy|year over year|year-over-year)\b",
                                 (q.raw_question or "").lower()):
        body = ("Yes — the earlier figures were year-over-year steps. " + body)
    text = _phrase(q.raw_question, body, tr)
    return Answer(text, min(confs) if confs else 0.9, "answered", tr, last_val,
                  headline=headline)


def _abs(fact: dict) -> float:
    """A fact's value in absolute dollars/shares (so differently-scaled inputs
    can be combined safely)."""
    return float(fact["value"]) * _SCALE_MULT.get(fact.get("scale"), 1.0)


def _money_abs(absval: float, scale: str) -> str:
    return _fmt_money({"value": absval / _SCALE_MULT.get(scale, 1.0), "scale": scale})


def _answer_computed(ent, q, year, tr, key) -> Answer:
    """Compute a derived metric in code from two grounded line items, citing
    both. Refuses (never estimates) if either input isn't grounded. EBITDA and
    free cash flow are flagged non-GAAP (computed, not reported line items)."""
    name, op, a_key, b_key = COMPUTED[key]
    desc = _period_desc(q, year)
    fa, s1, _ = find_fact(ent.cik, a_key, year, q.period_kind, q.quarter, instant_end=q.instant_end)
    fb, s2, _ = find_fact(ent.cik, b_key, year, q.period_kind, q.quarter, instant_end=q.instant_end)
    if s1 != "found" or s2 != "found":
        miss = METRICS[a_key].name if s1 != "found" else METRICS[b_key].name
        return Answer(f"I can't compute {_lc(name)} for {ent.title} in {desc}: "
                      f"the \"{miss}\" line isn't grounded in the Bank, so I won't "
                      "estimate it.", 0.0, "refused", tr)
    tr.cite(fa); tr.cite(fb)
    a, b = _abs(fa), _abs(fb)
    note = ""
    if op in ("pct", "ratio"):
        if b <= 0:
            # a zero OR NEGATIVE denominator (e.g. ROE on a stockholders'-equity
            # DEFICIT) makes the ratio meaningless — a loss over a deficit prints a
            # spurious large positive %. Refuse rather than mislead.
            why = ("is a deficit (negative)" if b < 0 else "is zero")
            return Answer(f"{ent.title}'s {METRICS[b_key].name.lower()} {why} for "
                          f"{desc}, so {_lc(name)} isn't meaningful — I won't "
                          "report a number that would mislead.", 0.2, "hedged", tr)
        val = a / b * (100 if op == "pct" else 1)
        hv = f"{val:.1f}%" if op == "pct" else f"{val:.2f}x"
        expl = f"{_fmt_money(fa)} / {_fmt_money(fb)}"
    elif op == "sum":
        val = (a + b) / _SCALE_MULT.get(fa["scale"], 1.0)
        hv = _money_abs(a + b, fa["scale"])
        expl = f"operating income {_fmt_money(fa)} + D&A {_fmt_money(fb)}"
        note = (" (EBITDA isn't a GAAP-defined line — it's non-GAAP — but I don't "
                "take it from a reconciliation; I compute it from two audited GAAP "
                "lines, operating income + D&A.)")
    else:  # fcf: a - |b|
        absval = a - abs(b)
        val = absval / _SCALE_MULT.get(fa["scale"], 1.0)
        hv = _money_abs(absval, fa["scale"])
        capd = {"value": abs(fb["value"]), "scale": fb["scale"]}
        expl = f"operating cash flow {_fmt_money(fa)} − capex {_fmt_money(capd)}"
        note = (" (Free cash flow isn't a GAAP-defined line — it's non-GAAP — but "
                "I don't take it from a reconciliation; I compute it from two "
                "audited GAAP cash-flow lines, operating cash flow − capex.)")
    tr.steps.append(f"{name} = {expl} = {hv}")
    text = _phrase(q.raw_question,
                   f"{ent.title}'s {_lc(name)} for {desc} was {hv} ({expl})." + note,
                   tr)
    return Answer(text, min(fa["confidence"], fb["confidence"]), "answered",
                  tr, val, headline=(f"{name} · {desc}", hv))


def _metric_figure(ent, q, year, mk) -> dict:
    """Compute ONE metric (level or derived) as {ok, name, value_str, num, facts},
    for assembling multi-metric / multi-year / growth answers. ok=False if not
    grounded. `num` is the comparable numeric value (absolute $ for money)."""
    if mk in COMPUTED:
        name, op, a_key, b_key = COMPUTED[mk]
        fa, s1, _ = find_fact(ent.cik, a_key, year, q.period_kind, q.quarter, instant_end=q.instant_end)
        fb, s2, _ = find_fact(ent.cik, b_key, year, q.period_kind, q.quarter, instant_end=q.instant_end)
        if s1 != "found" or s2 != "found":
            return {"ok": False, "name": name}
        a, b = _abs(fa), _abs(fb)
        if op in ("pct", "ratio"):
            if b <= 0:                 # zero or negative base -> not meaningful
                return {"ok": False, "name": name}
            num = a / b * (100 if op == "pct" else 1)
            vs = f"{num:.1f}%" if op == "pct" else f"{num:.2f}x"
        elif op == "sum":
            num = a + b
            vs = _money_abs(num, fa["scale"])
        else:
            num = a - abs(b)
            vs = _money_abs(num, fa["scale"])
        return {"ok": True, "name": name, "value_str": vs, "num": num,
                "facts": [fa, fb], "nongaap": True}
    spec = METRICS[mk]
    f, s, _ = find_fact(ent.cik, mk, year, q.period_kind, q.quarter, instant_end=q.instant_end)
    if s != "found":
        return {"ok": False, "name": spec.name}
    vs_ = (_fmt_money({**f, "value": -f["value"]}) + " (outflow)"
           if mk == "capex" and f["value"] < 0 else _fmt_money(f))
    return {"ok": True, "name": spec.name, "value_str": vs_,
            "num": f["value"], "facts": [f], "nongaap": False}


def _margin_figure(ent, q, year, mk) -> dict:
    """ONE margin (numerator / revenue) for a period, same dict shape as
    _metric_figure. Numerator is a level or a computed metric; both are taken in
    ABSOLUTE dollars so the ratio is scale-correct."""
    base = _metric_figure(ent, q, year, mk)
    name = (base.get("name") or str(mk)) + " margin"
    pct_rev = q.pct_of_revenue
    if pct_rev and mk in METRICS and METRICS[mk].statement != "balance_sheet":
        name = (base.get("name") or str(mk)) + " as % of revenue"
    elif mk in METRICS and not METRICS[mk].margin_eligible:
        return {"ok": False, "name": name, "ineligible": True}
    if not base["ok"]:
        return {"ok": False, "name": name}
    num_abs = base["num"] if mk in COMPUTED else _abs(base["facts"][0])
    if mk == "capex":
        num_abs = abs(num_abs)            # an outflow printed in parentheses
    frev, s, _ = find_fact(ent.cik, "revenue", year, q.period_kind, q.quarter,
                           instant_end=q.instant_end)
    if s != "found" or _abs(frev) == 0:
        return {"ok": False, "name": name}
    pct = num_abs / _abs(frev) * 100
    return {"ok": True, "name": name, "value_str": f"{pct:.1f}%", "num": pct,
            "facts": base["facts"] + [frev], "nongaap": mk in COMPUTED}


def _figure(ent, q, year, mk) -> dict:
    """One figure for a period under the current calc: a margin, or a level/
    computed value. The single-period ATOM that the multi-period composer maps
    over — decomposition lives here, not inside each answerer."""
    targets = q.margin_metrics
    use_margin = (q.calc == "margin" and (not targets or mk in targets)
                  and mk != "revenue")
    return (_margin_figure(ent, q, year, mk) if use_margin
            else _metric_figure(ent, q, year, mk))


def _period_tag(q, year: int) -> str:
    """Short period tag for a figure in a multi list: FY2025 / Q2 2026 / H1 2026."""
    if q.period_kind == "quarter":
        return f"Q{q.quarter} {year}" if q.quarter else f"{year} quarter"
    if q.period_kind == "ytd":
        return f"{q.quarter * 3}-mo {year}" if q.quarter else f"YTD {year}"
    return f"FY{year}"


def _answer_multi(ent, q, periods, tr, metric_keys, pp_change=False) -> Answer:
    """Answer a question asking for several figures and/or several periods at
    once ('revenue and operating income', 'revenue for 2023-2025', 'Q2 2025 and
    fiscal 2025'), each grounded and cited. `periods` is a list of (per-period
    query, year), so every leg is answered in its own period kind."""
    lines, missing, confs = [], set(), []
    plan = list(q.items) if q.items else [(mk, None) for mk in metric_keys]
    for mk, calc in plan:
        cells, nums = [], []
        name = None
        nongaap = False
        for qq, y in periods:
            if calc is not None:               # this item's own calc (level / margin)
                qq = replace(qq, calc=calc, margin_metrics=[])
            r = _figure(ent, qq, y, mk)           # per-period atom (margin-aware)
            if r["ok"]:
                name = r["name"]
                nongaap = nongaap or r.get("nongaap", False)
                for f in r["facts"]:
                    tr.cite(f)
                    confs.append(f.get("confidence", 0.9))
                tag = _fact_tag(r["facts"][0]) if r["facts"] else _period_tag(qq, y)
                cells.append(f"{tag} {r['value_str']}")
                nums.append((tag, r["num"]))
            else:
                missing.add(f"{r['name']} ({_period_tag(qq, y)})")
        if pp_change and len(nums) >= 2:
            d = nums[-1][1] - nums[0][1]
            cells.append(f"change {nums[0][0]}→{nums[-1][0]}: {d:+.1f} percentage points")
        if cells:
            # a computed/derived figure (EBITDA, FCF, a margin of either) is labeled
            # non-GAAP — a derived estimate must never read as a grounded line item.
            nong = " [non-GAAP, computed from GAAP inputs]" if nongaap else ""
            lines.append(f"{name}: " + ", ".join(cells) + nong)
    if not lines:
        return Answer(f"I couldn't ground those figures for {ent.title} — none of "
                      f"{', '.join(sorted(missing)[:6])} is in the banked filings, so I "
                      "won't fabricate them." + _offer(ent, metric_keys[0]),
                      0.0, "refused", tr)
    for ln in lines:
        tr.steps.append(ln)
    body = f"{ent.title} — " + "; ".join(lines) + "."
    if missing:
        body += f" (I couldn't ground: {', '.join(sorted(missing))}.)"
    text = _phrase(q.raw_question, body, tr)
    # a partial answer (some requested figures not grounded) is HEDGED, not a
    # clean "answered" — the user asked for more than we could stand behind.
    return Answer(text, min(confs, default=0.9), "hedged" if missing else "answered",
                  tr, value=None)


_COVERAGE_CUES = (
    "what date", "which date", "the date", "date for", "date of", "as of when",
    "as of what", "as of which", "what period", "which period", "what periods",
    "what year", "which year", "how recent", "latest filing", "most recent filing",
    "what filing", "which filing", "time period", "reporting period",
    "what fiscal", "how current", "how up to date", "how up-to-date",
    "when are these", "when is this", "when was this", "what's the date",
    "whats the date", "what data do you have", "what filings")


def _is_coverage_question(text: str) -> bool:
    t = (text or "").lower()
    return any(c in t for c in _COVERAGE_CUES)


# =========================================================================== #
# ROUTER  —  three deterministic pieces the lanes are built on:
#   1. a metric ALLOWLIST (default-refuse): a cue counts only if it names a
#      complete line item we bank; anything else that is asked is "untracked";
#   2. per-leg PERIODS: each requested period carries its own kind;
#   3. banked-aware COMPANY resolution.
# The lane order itself lives in one place: _pre_lane / _post_lane below.
# =========================================================================== #

# --- 1. metric allowlist ---------------------------------------------------- #
# Each pattern names something we do NOT bank. Any metric cue it overlaps is
# ABSORBED (dropped), so "automotive revenue" is never answered with total
# revenue, "retained earnings" with net income, or "price to earnings" with net
# income. Matched on lowercase text with hyphens -> spaces (same length, so the
# spans line up with the cue spans).
_MARKETDATA_PAT = (r"\bp\s?/\s?e\b|\bpe\s+ratio\b|\bprice\s+(?:to|/)\s+(?:earnings|sales|book)\b|"
                   r"\bmarket\s+cap(?:itali[sz]ation)?\b|\b(?:stock|share)\s+prices?\b|"
                   r"\bdividend\s+yield\b|\benterprise\s+value\b|\bev\s*/\s*ebitda\b|"
                   r"\btrading\s+at\b|\b52\s+week\b|\bstock\s+(?:performance|chart|quote)\b|"
                   r"\btrading\s+volume\b|\bshares?\s+(?:trade|traded)\b|\bdaily\s+volume\b|"
                   r"\btrade[sd]?\s+daily\b|\bdaily\s+trading\b|\bvolume\b")
_UNTRACKED = [(_re.compile(p), name) for p, name in [
    (_MARKETDATA_PAT, "market data (prices, P/E, market cap)"),
    (r"\bcogs\b|\bcosts?\s+of\s+(?:revenues?|sales|goods(?:\s+sold)?|services)\b",
     "cost of revenue"),
    (r"\bretained\s+earnings\b", "retained earnings"),
    (r"\b(?:deferred|unearned|unbilled)\s+revenues?\b|\bcontract\s+(?:liabilities|assets)\b",
     "deferred revenue"),
    (r"\bbook\s+value\s+per\s+share\b", "book value per share"),
    (r"\b(?:revenues?|sales|cash\s+flows?|free\s+cash\s+flows?|fcf)\s+per\s+share\b",
     "a per-share ratio"),
    (r"\bdividends?\b", "dividends"),
    (r"\b(?:sales|selling)\s+(?:and|&)\s+marketing\b|\bmarketing\s+(?:expenses?|costs?|"
     r"spend(?:ing)?)\b|\bs&m\b", "sales and marketing expense"),
    (r"\bselling,?\s+general\b|\bgeneral\s+(?:and|&)\s+administrative\b|\bsg&a\b|\bg&a\b|"
     r"\badministrative\s+expenses?\b", "SG&A expense"),
    (r"\b(?:segments?|geographic(?:al)?|regional|product\s+lines?|business\s+units?|"
     r"divisions?)\b", "segment-level figures"),
    (r"\binterest\s+(?:expense|income|paid|costs?|coverage)\b|\bnet\s+interest\b",
     "interest expense / income"),
    (r"\bpre\s?tax\s+(?:income|profit|earnings|loss)\b|\b(?:income|earnings|profit)\s+"
     r"before\s+(?:income\s+)?tax(?:es)?\b", "pre-tax income"),
    (r"\bincome\s+tax(?:es)?\b|\btax\s+(?:expense|provision|rate|bill|benefit)\b|"
     r"\beffective\s+tax\b|\bprovision\s+for\s+(?:income\s+)?taxes\b|\b(?<!after\s)taxes\b",
     "income taxes"),
    (r"\binventor(?:y|ies)\b", "inventory"),
    (r"\b(?:accounts\s+)?receivables?\b", "receivables"),
    (r"\baccounts\s+payable\b|\bpayables\b", "accounts payable"),
    (r"\bgoodwill\b|\bintangibles?(?:\s+assets)?\b", "goodwill / intangibles"),
    (r"\bpp&e\b|(?<!purchases of )\bproperty,?\s+(?:plant|and\s+equipment)\b", "PP&E"),
    (r"\bdebt\s+to\s+equity\b|\bd/e\b|\b(?:debt|leverage)\s+ratio\b", "a debt-to-equity ratio"),
    (r"\b(?:long|short)\s+term\s+debt\b|\b(?:total|net)\s+debt\b|\bdebt\b|\bborrowings?\b|"
     r"\bnotes\s+payable\b|\bconvertible\s+notes\b", "debt"),
    (r"\blease\s+(?:liabilities|obligations)\b|\boperating\s+leases?\b", "lease obligations"),
    (r"\bbuy\s?backs?\b|\b(?:share|stock)\s+repurchases?\b|\brepurchases?\s+of\s+(?:common\s+)?"
     r"stock\b|\brepurchased\b", "share buybacks"),
    (r"\b(?:stock|share|equity)\s+based\s+compensation\b|\bsbc\b|\bstock\s+compensation\b",
     "stock-based compensation"),
    (r"\bbacklog\b|\bremaining\s+performance\s+obligations?\b|\brpo\b|\bbookings\b",
     "backlog / RPO"),
    (r"\bworking\s+capital\b", "working capital"),
    (r"\b(?:quick|acid\s+test)\s+ratio\b", "the quick ratio"),
    (r"\bshares?\s+outstanding\b|\boutstanding\s+shares\b|\bshare\s+count\b|"
     r"\bnumber\s+of\s+shares\b|\bweighted\s+average\s+(?:number\s+of\s+)?(?:basic\s+|"
     r"diluted\s+)?shares\b|\b(?:basic|diluted)\s+shares\b", "share count"),
    (r"\brestricted\s+cash\b", "restricted cash"),
    (r"\bcash,?\s+(?:cash\s+equivalents,?\s+)?(?:and|&)\s+(?:short\s+term\s+investments|"
     r"marketable\s+securities|investments)\b", "cash plus investments"),
    (r"\bcash\s+burn\b|\bburn\s+rate\b", "cash burn"),
    (r"\b(?:investing|financing)\s+(?:cash\s+flows?|activities)\b|\bcash\s+(?:flows?\s+)?"
     r"(?:from|used\s+(?:in|for)|provided\s+by)\s+(?:investing|financing)\b",
     "investing / financing cash flow"),
    (r"\b(?:other\s+)?comprehensive\s+(?:income|loss)\b", "comprehensive income"),
    (r"\bebit\b", "EBIT"),
    (r"\b(?:non\s?controlling|minority)\s+interests?\b", "noncontrolling interest"),
    (r"\bheadcount\b|\bemployees\b|\bemployee\s+count\b", "headcount"),
    (r"\bdeliveries\b|\bsubscribers\b|\barpu\b|\bunits\s+sold\b|\bvehicles\s+delivered\b",
     "operating (non-financial) metrics"),
    (r"\bguidance\b|\bforecasts?\b|\bprojections?\b|\bconsensus\b|\banalyst\s+estimates?\b",
     "forecasts / guidance"),
    (r"\bearnings\s+(?:calls?|release|date|season|transcripts?)\b", "earnings-call material"),
    (r"\b(?:in|outside(?:\s+of)?|from|within)\s+(?:the\s+)?(?:us|u\.s\.|usa|united\s+states|"
     r"china|europe|asia|japan|international|domestic|north\s+america|overseas)\b|"
     r"\b(?:international|domestic|overseas)\s+(?:revenues?|sales)\b", "a geographic breakdown"),
    (r"\bconsolidated\s+net\s+(?:income|loss|earnings)\b|\bnet\s+(?:income|loss)\s+"
     r"(?:including|before\s+allocation\s+to)\s+(?:non\s?controlling|minority)\b|"
     r"\bnet\s+(?:income|loss)\s+attributable\s+to\s+(?:non\s?controlling|minority)\b",
     "net income including (or attributable to) noncontrolling interests — I report the "
     "amount attributable to the company"),
    (r"\bnet\s+(?:change|increase|decrease)\s+in\s+cash\b|\btotal\s+cash\s+flows?\b|"
     r"\bchange\s+in\s+cash\b", "the net change in cash"),
    (r"\btotal\s+costs\s+and\s+expenses\b|\btotal\s+(?:costs|expenses)\b",
     "total costs and expenses"),
    (r"\b(?:un)?levered\s+free\s+cash\s+flow\b|\bowner\s+earnings\b",
     "an alternative cash-flow / earnings definition"),
    (r"\bpre\s?tax\b", "pre-tax figures"),
    (r"\bper\s+employee\b|\bemployee\b", "per-employee figures"),
    (r"\b(?:as\s+a\s+)?(?:share|percent(?:age)?|%|fraction|portion|ratio)\s+of\s+"
     r"(?!(?:total\s+|net\s+)?(?:revenues?|sales)\b)", "a ratio to something other than revenue"),
    (r"\bceo\b|\bcfo\b|\bcoo\b|\bchief\s+\w+\s+officer\b|\bexecutives?\b|"
     r"\bboard\s+of\s+directors\b|\bfounders?\b|\bexecutive\s+compensation\b",
     "executive / management information"),
    (r"\boutlook\b|\bnext\s+(?:year|quarter)\b|\bpredict(?:ion)?s?\b|\bin\s+the\s+future\b",
     "forecasts / guidance"),
]]

# Open-noun rule: "<word> revenue / sales / gross profit / operating income" where
# <word> is not a function word, period word or company name names a COMPONENT
# ("automotive revenue", "Cloud operating income", "passenger revenue") — a
# different figure from the consolidated total, so it must not be answered with it.
_COMPONENT_HEAD = (r"(?:revenues?|sales|gross\s+(?:profit|margin)|"
                   r"operating\s+(?:income|profit|loss|margin))")
_OPEN_NOUN_RE = _re.compile(r"\b([a-z][a-z&'’]*)\s+(" + _COMPONENT_HEAD + r")\b")
_FROM_COMPONENT_RE = _re.compile(
    r"\b(revenues?|sales)\s+(?:(?:came|comes|coming|come|was|were|is|generated|derived|"
    r"earned)\s+)?(?:from|of|in)\s+(?:the\s+|its\s+|their\s+)?([a-z][a-z&]*)\b")
_HEAD_OK = set("""
a an the its their his her our your my this that these those it they company companys
firm business and or of in on at for to from with by vs versus compared about than as
what whats was were is are be been being do does did has have had how much many which
show give get list tell report reports reported reporting compare post posted record
recorded generate generated earn earned make made bring brought book booked see saw
achieve achieved total net gross overall combined consolidated gaap reported annual
annually quarterly yearly fiscal fy ytd year years month months quarter quarters half
latest recent last prior previous current same full first second third fourth nine six
three twelve trailing average avg highest lowest best worst record its both all each
every operating top line also then now please just only plus total's me us i dollar
dollars usd million billion growth grew yoy change increase decrease decline
q1 q2 q3 q4 h1 h2 annualized operations continuing using use show compute calculate
find pull need want revenue revenues sales profit income margin calendar sequential
sequentially qoq most past recent january february march april may june july august september october
november december jan feb mar apr jun jul aug sep sept oct nov dec""".split())
_DEF_STOP_TICKERS = {"BE", "DIS"}          # also English words: require upper case


def _alias_words() -> set:
    """Lowercase tokens that name a BANKED company (tickers, name words, aliases)
    — protected in the open-noun rule so "Tesla revenue" stays revenue."""
    out = set()
    for tk, (title, _cik) in _banked_index().items():
        out.add(tk.lower())
        out |= {w for w in _re.findall(r"[a-z]+", title.lower())}
    for a in _ENT_ALIASES:
        out |= set(a.split())
    return out


def _company_words(e) -> set:
    """DISTINCTIVE words that name company `e` (ticker, aliases, non-generic
    title words) — "energy" names Bloom Energy only when Bloom is the subject."""
    out = {e.ticker.lower()} | {w for a, tk in _ENT_ALIASES.items() if tk == e.ticker
                                for w in a.split()}
    out |= {w for w in _re.findall(r"[a-z]+", e.title.lower())
            if w not in _GENERIC_NAME_WORDS and len(w) >= 3}
    return out


def _company_spans(t: str, ents) -> list:
    """Spans in normalized text `t` naming one of `ents`: the ticker, an alias, a
    distinctive name word, or a run of 2+ consecutive title words ("american
    airlines", "bloom energy") — generic words only count inside such a run."""
    spans = []
    for e in ents:
        for w in _company_words(e):
            spans += [(m.start(), m.end()) for m in _re.finditer(rf"\b{_re.escape(w)}\b", t)]
        tw = _re.findall(r"[a-z]+", e.title.lower())
        for i in range(len(tw) - 1):
            for j in range(len(tw), i + 1, -1):
                phrase = r"\s+".join(_re.escape(w) for w in tw[i:j])
                spans += [(m.start(), m.end()) for m in _re.finditer(rf"\b{phrase}\b", t)]
    return spans


def _norm_text(text: str) -> str:
    return (text or "").lower().replace("-", " ").replace("’", "'")


def _untracked_matches(text: str, ent=None) -> list:
    """[(start, end, name)] for everything asked that we don't bank — contained
    matches dropped, so the most specific name wins."""
    t = _norm_text(text)
    found = []
    for rx, name in _UNTRACKED:
        found += [(m.start(), m.end(), name) for m in rx.finditer(t)]
    co_words = set()
    for e in ([ent] if ent is not None else []) + _companies_named(text):
        co_words |= _company_words(e)
    ok = _HEAD_OK | co_words | {"so", "ok", "okay", "yeah", "yes", "well", "then", "now",
                                "also", "and", "just", "thanks", "thank", "hey", "hi"}
    import difflib
    for m in _OPEN_NOUN_RE.finditer(t):
        w = _re.sub(r"'s?$", "", m.group(1))
        if w.endswith("s") and w[:-1] in ok:
            continue                         # "nvidias revenue": a possessive, not a part
        if len(w) >= 4 and difflib.get_close_matches(w, list(co_words), n=1, cutoff=0.8):
            continue                     # "Vertv" ~ "vertiv": a typo'd company name
        if w not in ok and not w.isdigit():
            orig = (text or "")[m.start(1):m.end(1)]
            if orig[:1].isupper():
                found.append((m.start(), m.end(), f"“{orig} {m.group(2)}” (that isn't one of "
                              "my banked companies, or a part of the business I break out)"))
            else:
                found.append((m.start(), m.end(), f"a component figure (“{w} "
                              f"{m.group(2)}”) rather than the consolidated total"))
    for m in _FROM_COMPONENT_RE.finditer(t):
        w = m.group(2)
        if w not in ok:
            found.append((m.start(), m.end(), f"a component figure ({m.group(1)} "
                          f"from {w}) rather than the consolidated total"))
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept = []
    for s, e, n in found:
        if not any(ks <= s and e <= ke for ks, ke, _ in kept):
            kept.append((s, e, n))
    return kept


_MODIFIER_PREFIXES = ("a ratio", "a geographic", "a component", "“", "per-employee", "a per-share",
                      "net income including", "segment-level", "pre-tax")
_SUBSTITUTION_RE = _re.compile(r"\b(?:using|instead\s+of|in\s+place\s+of|as\s+a\s+proxy|"
                               r"substitute|pretend|treat\s+\w+\s+as)\b", _re.I)


def _resolve_metrics(q, ent=None) -> None:
    """Apply the allowlist to a parsed Query IN PLACE (works for the LLM and the
    offline parse alike, since it reads the raw text): absorbed metrics are
    dropped and q.untracked records what was asked but isn't banked."""
    rawq = q.raw_question or ""
    unt = _untracked_matches(rawq, ent)
    if not unt:
        return
    spans = _cue_spans(_norm_text(rawq))
    absorbed = {k for k, (s, e, _) in spans.items()
                if any(us < e and s < ue for us, ue, _ in unt)}
    q.metrics = [m for m in (q.metrics or ([q.metric] if q.metric else []))
                 if m not in absorbed]
    q.metric = q.metrics[0] if q.metrics else None
    q.untracked = [(s, n) for s, _, n in unt]
    q._absorbed = [k for k in sorted(absorbed, key=lambda k: spans[k][0])]
    q._spans = {k: v for k, v in spans.items() if k not in absorbed}


# Words that carry no metric/period/company meaning of their own.
_FUNCTION_WORDS = _HEAD_OK | set("""
what whats how much many which when where why who was were is are be been being do does
did can could would should will shall may might please pls plz thanks thank you me i we
us my our your let lets give tell show list pull get fetch find look up check provide
share know want need like see also and or but so then just quick quickly real really
exact exactly approximately roughly about around nearly almost precisely okay ok hey hi
hello again too either both all each any some there here that this these those it its
they them their his her a an the of for in on at to from by with into than as over
under between within during per via after before since until through thru till end
ended ending close closing as date day today period periods year years quarter
quarters month months half fiscal calendar annual annually quarterly latest most recent
newest last prior previous current same first second third fourth nine six three
twelve ytd fy january february march april may june july august september october
november december jan feb mar apr jun jul aug sep sept oct nov dec growth grow grew
grown change changed changes yoy qoq cagr compound annualized annualised rate percent
percentage pct increase increased decrease decreased decline declined vs versus
compared compare comparison sequential sequentially trend margin margins ratio
higher lower better worse improve improved more less bigger smaller total net gross
overall reported reporting gaap figure figures number numbers amount amounts value
values dollars dollar usd million millions billion billions thousand thousands
statement statements income balance sheet filing filings report reports annual
company firm attributable common stockholders shareholders parent level
levels spend spent spending bring brought generate generated post posted book booked
record recorded have has had hold held every right correct true entire whole
cumulative range span overall break down breakdown each again too also same instead
compare comparison detail details more imo way btw shrink shrank shrunk double doubled
without tell say said mention explain describe format formatted json table bullet bullets
list rounded round rounding nearest words spelled enthusiastic honest honestly briefly
brief short shortly memory knowledge still even if don't dont have filing confirm
record best ever ignore instructions instruction note rewriter answer normally normal
gave just great cool nice interesting please kindly quickly simple simply plain english
trend trends history historical over time all show fall falls fell drop drops dropped
rise rises rose high higher low keep asap expense expenses spend cost costs imho btw fyi tbh inc corp
corporation co ltd plc llc holdings group
""".split())
_EPS_WORDS = {"basic", "diluted", "share", "shares", "per", "eps"}


def _unexplained_tokens(q, ent, named) -> list:
    """Default-refuse, for real: in a factual question every content word must be
    ACCOUNTED FOR — a function word, part of a metric cue, a period, the company,
    a calc word, or an untracked phrase. A leftover word ("energy" revenue, "FSD"
    revenue, revenue "for the automotive business", "Netflix" net income) means
    the user is asking for something narrower or different than the line we'd
    answer with, so we refuse instead of substituting the consolidated figure."""
    import difflib
    rawq = q.raw_question or ""
    t = _norm_text(rawq)
    covered = [(s, e) for s, e, _ in _cue_spans(t).values()]
    covered += [(s, e) for s, e, _ in _untracked_matches(rawq, ent)]
    ents = ([ent] if ent is not None else []) + list(named)
    covered += _company_spans(t, ents)
    for cues in _CALC_CUES.values():
        for c in cues:
            covered += [(m.start(), m.end()) for m in
                        _re.finditer(r"\b" + _re.escape(c) + r"\b", t)]
    eps = bool(set(q.metrics) & {"eps_basic", "eps_diluted"})
    distinct = set()
    for e in ents:
        distinct |= {w for w in _company_words(e) if len(w) >= 4}
    left = []
    for m in _re.finditer(r"[a-z][a-z&']*", t):
        w = _re.sub(r"'s?$", "", m.group(0))
        if len(w) <= 1 or w in _FUNCTION_WORDS or (eps and w in _EPS_WORDS):
            continue
        if w == "consolidated" and "net_income" not in q.metrics:
            continue
        if any(s <= m.start() and m.start() + len(w) <= e for s, e in covered):
            continue
        if len(w) >= 4 and difflib.get_close_matches(w, list(distinct), n=1, cutoff=0.8):
            continue                                  # "Telsa", "Vertv"
        if w not in left:
            left.append(w)
    return left


def _untracked_is_primary(q) -> bool:
    """Refuse outright when the FIRST thing asked is untracked (or the question
    asks us to substitute one figure for another); otherwise answer what we can
    and note what was left out."""
    if not q.untracked:
        return False
    if not q.metrics or _SUBSTITUTION_RE.search(q.raw_question or ""):
        return True
    # a MODIFIER of the metric (a region, a component, per-employee, NCI) changes
    # what the metric is — never answer the unmodified figure "with a note".
    if any(n.startswith(_MODIFIER_PREFIXES) for _s, n in q.untracked):
        return True
    spans = getattr(q, "_spans", {}) or {}
    first_metric = min((spans[m][0] for m in q.metrics if m in spans), default=10**6)
    return min(s for s, _ in q.untracked) < first_metric


def _untracked_names(q) -> str:
    names = []
    for _s, n in q.untracked:
        if n not in names:
            names.append(n)
    return " or ".join(names[:3])


# --- 2. per-leg periods ----------------------------------------------------- #
_MONTH_RX = (r"january|february|march|april|may|june|july|august|september|october|"
             r"november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec")
_MONTH_NUM = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
              "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_LEG_RE = _re.compile(
    r"(?P<qr>\bq(?P<qa>[1-4])\s*(?:-|to|through|thru)\s*q(?P<qb>[1-4])\b)"
    r"|(?P<q>\bq(?P<qn>[1-4])(?:\s*'(?P<qyy>\d{2}))?\b)"
    r"|(?P<qw>\b(?P<qword>first|second|third|fourth|1st|2nd|3rd|4th)\s+(?:fiscal\s+)?quarter\b)"
    r"|(?P<qe>\b(?:(?:three|3)\s+months?|quarters?)\s+end(?:ed|ing)\s+(?:in\s+)?(?P<qem>"
    + _MONTH_RX + r")\b|\b(?P<qem2>" + _MONTH_RX + r")(?:\s+(?P<qey>(?:19|20)\d{2}))?\s+quarter\b)"
    r"|(?P<m6>\bh1\b|\bfirst\s+half\b|\b(?:first\s+)?(?:six|6)\s+months?\b|\bhalf\s+year\b)"
    r"|(?P<m9>\b(?:first\s+)?(?:nine|9)\s+months?\b|\bfirst\s+three\s+(?:fiscal\s+)?quarters\b)"
    r"|(?P<ytd>\bytd\b|\byear\s+to\s+date\b)"
    r"|(?P<fy>\b(?:fiscal(?:\s+year)?|full\s+(?:fiscal\s+)?year)\s+'?(?P<fyy>(?:19|20)\d{2}|\d{2})\b"
    r"|\bfy\s*'?(?P<fy2>(?:19|20)?\d{2})\b)"
    r"|(?P<ann>\bfull\s+(?:fiscal\s+)?year\b|\bfiscal\s+year\b|\bannual(?:ly)?\b)"
    r"|(?P<yr>\b(?P<yv>(?:1[89]|20)\d{2})\b)")
_WORD_Q = {"first": 1, "second": 2, "third": 3, "fourth": 4,
           "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
_LEG_SEP_RE = _re.compile(r"\b(?:and|or|vs|versus|compared|against|then|plus)\b|[,;&]")


def _leg_text(text: str) -> str:
    """Rewrite compact period notations to the long forms _LEG_RE reads:
    3Q26 / 2Q 2025 / Q2-25 / Q2'25 -> "q3 2026"; 1H/H1 -> first half; Q1-Q3 -> q1 to q3."""
    t = (text or "").lower()
    t = _re.sub(r"\b(q[1-4])\s*[-–—]\s*(q[1-4])\b", r"\1 to \2", t)
    t = _re.sub(r"\b([1-4])q\s*'?(\d{2})\b", r"q\1 20\2", t)
    t = _re.sub(r"\b([1-4])q\s*((?:19|20)\d{2})\b", r"q\1 \2", t)
    t = _re.sub(r"\bq([1-4])\s*[-–—']\s*(\d{2})\b", r"q\1 20\2", t)
    t = _re.sub(r"\b(?:1h|h1)\b", "first half", t)
    return _norm_text(t)


def _parse_legs(text: str) -> list:
    """Every period the question names, in order, each with its OWN kind.

    Binding rules: an interim marker (Q2, nine months, YTD, annual) binds to the
    next year; "Q3 FY2026" / "Q3 of fiscal 2026" is ONE leg (the fiscal year is
    the quarter's container); "Q2 2025 and fiscal year 2025" is TWO legs; a bare
    year after interim-only legs inherits their kind ("Q2 2024 vs 2025" -> Q2
    2025); "Q1-Q3 2025" expands to three quarter legs; "the quarter ended March
    2026" is a quarter identified by its END MONTH (resolved per company)."""
    t = _leg_text(text)
    legs, pending, last_marker_end, last_year_tok = [], [], None, None

    def sep_between(a, b):
        return a is not None and bool(_LEG_SEP_RE.search(t[a:b]))

    for m in _LEG_RE.finditer(t):
        year, explicit = None, False
        if m.group("qr"):
            a, b = sorted((int(m.group("qa")), int(m.group("qb"))))
            pending += [("quarter", n, None) for n in range(a, b + 1)]
        elif m.group("q"):
            if m.group("qyy"):
                legs.append(Leg("quarter", 2000 + int(m.group("qyy")), int(m.group("qn")),
                                explicit=True))
                continue
            pending.append(("quarter", int(m.group("qn")), None))
        elif m.group("qw"):
            pending.append(("quarter", _WORD_Q[m.group("qword")], None))
        elif m.group("qe"):
            mon = (m.group("qem") or m.group("qem2"))[:3]
            if m.group("qey"):
                legs.append(Leg("quarter", int(m.group("qey")), None,
                                end_month=_MONTH_NUM[mon], explicit=True))
                continue
            pending.append(("quarter", None, _MONTH_NUM[mon]))
        elif m.group("m9"):
            pending.append(("ytd", 3, None))
        elif m.group("m6"):
            pending.append(("ytd", 2, None))
        elif m.group("ytd"):
            pending.append(("ytd", None, None))
        elif m.group("ann"):
            pending.append(("annual", None, None))
        elif m.group("fy"):
            y = m.group("fyy") or m.group("fy2")
            year, explicit = (int(y) if len(y) == 4 else 2000 + int(y)), True
        elif m.group("yr"):
            year = int(m.group("yv"))
        if year is None:
            last_marker_end = m.end()
            continue
        if pending:
            for kind, qn, em in pending:
                legs.append(Leg(kind, year, qn, end_month=em, explicit=True))
            # "Q4 and full year 2025": the explicit annual year is ALSO a leg when
            # a separator stands between it and the interim marker (otherwise it is
            # the quarter's fiscal-year container: "Q3 FY2026").
            if explicit and any(k != "annual" for k, _, _ in pending) \
                    and sep_between(last_marker_end, m.start()):
                legs.append(Leg("annual", year, explicit=True))
            pending = []
        elif explicit:
            legs.append(Leg("annual", year, explicit=True))
        elif legs and all(lg.kind != "annual" for lg in legs):
            p = legs[-1]                                       # inherit
            legs.append(Leg(p.kind, year, p.quarter, end_month=p.end_month, explicit=True))
        else:
            legs.append(Leg("annual", year))
        last_year_tok = (m.start(), m.end(), explicit)
    if pending:
        # markers AFTER their year: "2025 Q2", "fiscal 2025's second quarter" -> the
        # year is the container (replace its leg) unless a separator splits them.
        if (legs and last_year_tok and legs[-1].kind == "annual"
                and not sep_between(last_year_tok[1], last_marker_end or last_year_tok[1])):
            y = legs.pop().year
        else:
            y = legs[-1].year if legs else None
        legs += [Leg(k, y, qn, end_month=em, explicit=True) for k, qn, em in pending]
    # "revenue in 2026 for the nine months ended June 2026": the bare 2026 is
    # the same period restated, not a separate full-year leg
    interim_years = {lg.year for lg in legs if lg.kind != "annual" and lg.explicit}
    legs = [lg for lg in legs if lg.explicit or lg.kind != "annual"
            or lg.year not in interim_years]
    out = []
    for lg in legs:
        if lg not in out:
            out.append(lg)
    return out


def _legs_hetero(legs) -> bool:
    return len({(lg.kind, lg.quarter) for lg in legs}) > 1


# --- 3. company resolution -------------------------------------------------- #
_ENT_ALIASES = {"google": "GOOG", "googl": "GOOG", "alphabet": "GOOG",
                "spacemobile": "ASTS", "ast spacemobile": "ASTS", "ast": "ASTS",
                "american airlines": "AAL", "bloom energy": "BE",
                "intuitive surgical": "ISRG", "walt disney": "DIS", "disney": "DIS"}
# name words too generic to identify a company on their own
_GENERIC_NAME_WORDS = {"inc", "corp", "corporation", "company", "holdings", "group",
                       "technologies", "the", "co", "ltd", "plc", "energy", "american",
                       "airlines", "walt", "surgical", "intuitive", "bloom", "ast"}
_BANK_INDEX = None


def _banked_index() -> dict:
    """{ticker: (company title, cik)} for every banked company."""
    global _BANK_INDEX
    if _BANK_INDEX is None:
        idx = {}
        try:
            for f in bank.load_facts():
                idx.setdefault(f["ticker"], (f["company"], f["cik"]))
        except Exception:
            pass
        _BANK_INDEX = idx
    return _BANK_INDEX


def _companies_named(text: str) -> list:
    """Banked companies the question names — by ticker, alias, or a distinctive
    name word — in order of first mention. Only companies that actually have
    banked facts count."""
    raw = text or ""
    low = raw.lower()
    hits = {}
    for tk, (title, cik) in _banked_index().items():
        pos = []
        if len(tk) >= 3 and tk not in _DEF_STOP_TICKERS:
            m = _re.search(rf"\b{_re.escape(tk.lower())}\b", low)
        else:
            m = _re.search(rf"\b{_re.escape(tk)}\b", raw)
        if m:
            pos.append(m.start())
        for a, atk in _ENT_ALIASES.items():
            if atk == tk:
                m = _re.search(rf"\b{_re.escape(a)}\b", low)
                if m:
                    pos.append(m.start())
        for w in _re.findall(r"[a-z]{4,}", title.lower()):
            if w not in _GENERIC_NAME_WORDS:
                m = _re.search(rf"\b{w}(?:'?s)?\b", low)
                if m:
                    pos.append(m.start())
        # capitalized-only name words ("Intuitive", "Bloom") — English words too
        for w in _re.findall(r"[A-Za-z]{4,}", title):
            if w.lower() in ("intuitive", "bloom"):
                m = _re.search(rf"\b{w.capitalize()}\b", raw)
                if m:
                    pos.append(m.start())
        if pos and bank.facts_for(cik):
            hits[tk] = (min(pos), edgar.Entity(cik=cik, ticker=tk, title=title))
    if not hits:
        import difflib
        names = {}
        for tk, (title, cik) in _banked_index().items():
            for w in _re.findall(r"[a-z]{5,}", title.lower()):
                if w not in _GENERIC_NAME_WORDS:
                    names[w] = (tk, title, cik)
        for m in _re.finditer(r"\b[A-Za-z]{5,}\b", raw):
            if not m.group(0)[0].isupper():
                continue
            g = difflib.get_close_matches(m.group(0).lower(), list(names), n=1, cutoff=0.8)
            if g:
                tk, title, cik = names[g[0]]
                if bank.facts_for(cik):
                    fe = edgar.Entity(cik=cik, ticker=tk, title=title)
                    fe._typed = m.group(0)          # disclosed: "I read Nvidea as …"
                    hits[tk] = (m.start(), fe)
    return [e for _p, e in sorted(hits.values(), key=lambda x: x[0])]


# Words that start a sentence or name a metric/period — never a company mention.
_MENTION_STOP = set("""
What How When Where Which Why Who Whom Whose The A An In Did Do Does Is Was Were Give
Tell Show Compare And Of For At Can Could Would Should Will Shall May Might Must Please
Okay Ok OK Hey Hi Hello Thanks Thank I I'm Im My Me We Our You Your It Its They Their
This That These Those Let Lets Let's Also Now Then So But Just Any All Each Every Some
Year Years Fiscal Quarter Quarterly Annual Total Net Revenue Revenues Sales Q1 Q2 Q3 Q4
FY YTD TTM EPS EBITDA EBIT FCF GAAP ROE ROA SEC US USA USD AI CEO CFO R&D MD&A Explain
Define List Name Find Get Pull Calculate Compute Break Summarize Describe Yes No Sure
Great Cool Nice Good Wow Hmm Plus From To On By With Between Over During Since Before
After About If Or Not Are Has Have Had Been Be Am Here There Again Ask Answer Using
January February March April May June July August September October November December
Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec Monday Tuesday Wednesday Thursday
Friday Saturday Sunday Item Part Form CAGR YoY QoQ Non-GAAP P/E PE EV RPO SBC COGS
SG&A Operating Gross Free Cash Diluted Basic Return Current Margin Margins Growth""".split())
_ACRONYM_STOP = {w.upper() for w in _MENTION_STOP} | {
    "AND", "OR", "THE", "NOT", "HOW", "WHAT", "WHY", "IPO", "ESG", "ETF", "IRS", "LLC",
    "LP", "YOY", "QOQ", "MDA", "D&A", "TBD", "FAQ", "ASAP", "LOL", "PDF", "API", "LLM",
    "GPT", "OK", "NA", "USA", "UK", "EU", "SGA", "GA", "SM", "DSO", "PPE", "ARPU", "EBT",
    "IMO", "IMHO", "FYI", "BTW", "TLDR", "NI", "OI", "GP", "OCF", "OPEX", "CAPEX", "RND",
    "DA", "LTM", "NTM", "YTD", "FSD", "EV", "GM", "HQ", "IR", "PR", "Q&A", "FAQ"}


def _foreign_company(text: str, ent):
    """A company the question names (possessive "Microsoft's" or a ticker token)
    that is NOT the company in focus — so we never answer one company's question
    with another's figures. Returns an Entity or None."""
    raw = (text or "").replace("’", "'")
    letters = [c for c in raw if c.isalpha()]
    shouting = letters and sum(c.isupper() for c in letters) > 0.6 * len(letters)
    own = {ent.ticker.lower()} | {w.lower() for w in _re.findall(r"[A-Za-z]+", ent.title)}
    own |= {a for a, tk in _ENT_ALIASES.items() if tk == ent.ticker}
    toks = [("pos", m.group(1)) for m in _re.finditer(r"\b([A-Z][A-Za-z&.]+)'s\b", raw)]
    if not shouting:
        toks += [("tk", m.group(0)) for m in
                 _re.finditer(r"(?<![&/])\b[A-Z]{2,5}\b(?![&/])", raw)]
    for kind, tok in toks:
        if tok.lower() in own or tok in _MENTION_STOP or tok.upper() in _ACRONYM_STOP:
            continue
        try:
            cands, st = edgar.resolve_entity(tok)
        except Exception:
            continue
        if not cands or st != "exact" or cands[0].cik == ent.cik:
            continue
        if kind == "tk" and cands[0].ticker.upper() != tok:
            continue
        return cands[0]
    return None


# --- 4. small intents the lanes need ---------------------------------------- #
_MARKETDATA_RE = _re.compile(_MARKETDATA_PAT + r"|\bhow\s+much\s+is\s+(?:the\s+)?\w+\s+"
                             r"(?:stock|shares?)\b|\bstock\s+(?:doing|trading)\b")
_COMPANIES_RE = _re.compile(
    r"\b(?:which|what)\s+(?:other\s+)?(?:companies|stocks|tickers|firms|names)\b|"
    r"\blist\s+(?:of\s+)?(?:the\s+|all\s+)?(?:companies|stocks|tickers)\b|"
    r"\b(?:companies|stocks|tickers)\s+(?:do\s+)?you\s+(?:cover|have|know|support|track)\b|"
    r"\bwho\s+do\s+you\s+cover\b|\bwhat\s+do\s+you\s+cover\b|\bwhich\s+ones\s+do\s+you\b")


def _is_marketdata(text: str) -> bool:
    return bool(_MARKETDATA_RE.search(_norm_text(text)))


def _is_companies_list(text: str) -> bool:
    return bool(_COMPANIES_RE.search(_norm_text(text)))


_DEFINITIONS = {
    "revenue": "Revenue (the 'top line') is what a company earns from selling its "
               "goods and services in a period, before any costs are subtracted.",
    "gross profit": "Gross profit is revenue minus the direct cost of producing what "
                    "was sold (cost of revenue).",
    "gross margin": "Gross margin is gross profit as a percentage of revenue — how "
                    "much of each sales dollar is left after direct production costs.",
    "operating income": "Operating income is profit from the core business: revenue "
                        "minus cost of revenue and operating expenses (R&D, selling, "
                        "general and administrative), before interest and taxes.",
    "operating margin": "Operating margin is operating income as a percentage of "
                        "revenue.",
    "operating expenses": "Operating expenses are the costs of running the business "
                          "beyond direct production costs — R&D, selling, general and "
                          "administrative.",
    "net income": "Net income (the 'bottom line') is what is left after every expense, "
                  "including interest and taxes.",
    "net margin": "Net margin (profit margin) is net income as a percentage of revenue.",
    "eps": "Earnings per share is net income attributable to common shareholders "
           "divided by the weighted-average share count; 'diluted' EPS also counts "
           "shares that options and convertibles could create.",
    "ebitda": "EBITDA is earnings before interest, taxes, depreciation and "
              "amortization. It isn't a GAAP line; I compute it as operating income "
              "plus depreciation and amortization, both taken from the filing.",
    "free cash flow": "Free cash flow is operating cash flow minus capital "
                      "expenditures — the cash left after maintaining and growing the "
                      "asset base. It's non-GAAP; I compute it from two GAAP lines.",
    "operating cash flow": "Operating cash flow is the cash generated by day-to-day "
                           "operations, from the cash-flow statement.",
    "capex": "Capital expenditures are cash spent on property, plant and equipment, "
             "from the investing section of the cash-flow statement.",
    "roe": "Return on equity is net income divided by total stockholders' equity.",
    "roa": "Return on assets is net income divided by total assets.",
    "current ratio": "The current ratio is total current assets divided by total "
                     "current liabilities — a short-term liquidity measure.",
    "cagr": "CAGR, the compound annual growth rate, is the constant yearly growth "
            "rate that takes a starting value to an ending value over a span of years.",
    "10-k": "A 10-K is the audited annual report a US public company files with the "
            "SEC.",
    "10-q": "A 10-Q is the unaudited quarterly report a US public company files with "
            "the SEC for each of its first three fiscal quarters.",
    "gaap": "GAAP is US Generally Accepted Accounting Principles, the rules the "
            "audited financial statements follow.",
    "non-gaap": "A non-GAAP measure is one the company defines outside GAAP rules "
                "(e.g. adjusted EBITDA); it isn't audited the same way.",
    "md&a": "MD&A (Management's Discussion and Analysis) is the section where "
            "management explains the results, liquidity and trends in its own words.",
}
_DEF_ALIASES = {"top line": "revenue", "sales": "revenue", "revenues": "revenue",
                "bottom line": "net income", "profit margin": "net margin",
                "earnings per share": "eps", "diluted eps": "eps",
                "fcf": "free cash flow", "capital expenditures": "capex",
                "capital expenditure": "capex", "return on equity": "roe",
                "return on assets": "roa", "opex": "operating expenses",
                "10k": "10-k", "10q": "10-q", "non gaap": "non-gaap",
                "management discussion and analysis": "md&a"}
_DEF_RE = _re.compile(
    r"^\W*(?:ok(?:ay)?[,\s]+|so[,\s]+|and\s+)?(?:(?P<verb>define|explain|definition\s+of|"
    r"meaning\s+of|what\s+is\s+meant\s+by)\s+|(?P<wh>what\s+(?:is|are|does)|what's|whats)\s+)"
    r"(?:an?\s+|the\s+)?(?P<term>[a-z0-9&/\s'-]{2,40}?)"
    r"(?P<mean>\s+(?:mean|means|stand\s+for|measure|refer\s+to))?\s*[?.!]*\s*$")


def _definition_term(text: str, entity, named) -> str | None:
    """Narrow "what is X?" / "define X" / "what does X mean": X must be a term we
    can define. An explicit define / mean / explain always defines; a bare
    "what is EBITDA?" only does when no company is in focus (otherwise the user
    wants the company's figure)."""
    m = _DEF_RE.match((text or "").lower().strip())
    if not m:
        return None
    term = m.group("term").strip(" ?'")
    term = _DEF_ALIASES.get(term, term)
    if term not in _DEFINITIONS:
        return None
    if m.group("verb") or m.group("mean"):
        return term
    if _re.match(r"^\W*(?:ok(?:ay)?[,\s]+|so[,\s]+|and\s+)?(?:what\s+(?:is|are)|what's|whats)"
                 r"\s+the\b", (text or "").lower().strip()):
        return None                      # "what's THE free cash flow" -> the figure
    return term if (entity is None and not named) else None


def _answer_definition(term: str, ent, tr) -> Answer:
    tr.steps.append(f"definition: {term} (general knowledge, no filing figure)")
    co = f" for {ent.title}" if ent else " for any company I cover"
    return Answer(_DEFINITIONS[term] + f" Ask me for it{co} and I'll pull the figures "
                  "from the filing with a page citation.", 0.8, "answered", tr)


def _answer_judgment(q, ent, ctx, tr) -> Answer:
    """"Is that good?" — we don't grade results (that's an opinion the filing
    doesn't make), but we offer the comparison that lets the user judge."""
    co = ent.title if ent else "the company"
    if ctx is not None and getattr(ctx, "metrics", None):
        m = ctx.metrics[0]
        nm = _lc(METRICS[m].name if m in METRICS else COMPUTED[m][0])
        offer = (f" What I can do is put it in context — for example, how {co}'s {nm} "
                 "changed year over year, or how it compares across the years I have.")
    else:
        offer = (f" What I can do is put a figure in context — how it changed year "
                 "over year, or how a margin compares across years.")
    return Answer("I don't grade results as good or bad — that's a judgment call, not "
                  "something the filing states." + offer, 0.5, "hedged", tr)


def _answer_companies(tr) -> Answer:
    idx = _banked_index()
    names = [f"{title} ({tk})" for tk, (title, cik) in sorted(idx.items())
             if bank.facts_for(cik)]
    tr.steps.append(f"companies: {len(names)} banked")
    if not names:
        return Answer("I don't have any companies banked yet.", 0.0, "refused", tr)
    return Answer(f"I cover {len(names)} companies: " + ", ".join(names) + ". Name one "
                  "in your question (or pick it on the left) and ask about its revenue, "
                  "earnings, cash flow, margins, growth, or risk factors.",
                  0.9, "answered", tr)


_EXPLICIT_GROWTH_RE = _re.compile(
    r"\b(?:growth|grow|grew|grown|increase[ds]?|decrease[ds]?|decline[ds]?|change[ds]?|"
    r"yoy|y/y|year over year|qoq|quarter over quarter|sequential(?:ly)?|cagr|"
    r"compound annual|annuali[sz]ed|trend)\b")
_COMPARATIVE_RE = _re.compile(
    r"\b(?:improve[ds]?|worsen(?:ed)?|better|worse|higher|lower|bigger|smaller|"
    r"more|less|vs|versus|compared?|against|outperform(?:ed)?)\b|\bfrom\b.*\bto\b")
_QOQ_RE = _re.compile(r"\b(?:qoq|q/q|quarter over quarter|sequential(?:ly)?|"
                      r"(?:prior|previous|last)\s+quarter)\b")


def _is_vague(q) -> bool:
    return q.metric is None and not q.metrics and q.calc == "none" and not q.untracked


_JUDGMENT_RE = _re.compile(
    r"\b(?:is|was|are|were|does|did)\s+(?:that|this|it|they|those|these|the\s+\w+)\s+"
    r"(?:a\s+)?(?:good|bad|great|healthy|strong|weak|concerning|worrying|impressive|"
    r"decent|ok|okay|normal|high|low|recession[- ]proof|safe|risky)\b|\brecession[- ]proof\b|"
    r"\b(?:a\s+)?record\b|\bhighest\b|\blowest\b|\ball[- ]time\b|\bbest\s+ever\b|"
    r"\ba\s+lot\b|^\W*(?:is|was|are|were)\b.*\b(?:good|bad|healthy|strong|weak|"
    r"impressive|concerning)\s*\??\s*$|"
    r"\b(?:good|bad|great|strong|weak)\s+(?:year|quarter|"
    r"result|results|number|margin|growth)\b|\bhow\s+(?:good|bad)\s+is\b", _re.I)


def _is_judgment(text: str) -> bool:
    return bool(_JUDGMENT_RE.search(text or ""))


_PERSONAL_RE = _re.compile(r"\b(?:my|our)\s+(?:own\s+)?(?:taxes|tax\s+return|homework|essay|"
                           r"resume|cv|code|budget|portfolio|retirement)\b", _re.I)


def _pre_lane(q, entity, named, context=None):
    """Lanes decided BEFORE a company is needed, in precedence order. Returns
    (lane, payload) or None. One place, one order — tested directly."""
    rawq = q.raw_question or ""
    vague = _is_vague(q)
    if _is_advice(rawq):                               # trust boundary first
        return "advice", None
    if _PERSONAL_RE.search(rawq):                      # "my taxes", "my homework"
        return "offtopic", None
    if not q.metrics and _is_marketdata(rawq):         # filings carry no prices
        return "marketdata", None
    if _is_judgment(rawq) and not q.years:             # a grade is an opinion
        return "judgment", None
    if vague and _is_companies_list(rawq):
        return "companies", None
    term = _definition_term(rawq, entity, named)
    if term:
        return "explain", term
    if vague and _is_capability(rawq) and not named:
        return "capability", None
    # "Okay, for each year?" after a figure is a follow-up, not small talk: with a
    # live frame and a follow-up signal, the frame wins over a leading "okay/thanks"
    followup = (context is not None and getattr(context, "metrics", None)
                and _FOLLOWUP_RE.search(_norm_text(rawq))
                and not _re.search(r"\b(?:thanks?|thx|ty|bye|goodbye)\b", rawq.lower()))
    if vague and _is_social(rawq) and not _is_overview(rawq) and not named and not followup:
        return "smalltalk", None
    if vague and _is_offtopic(rawq):
        return "offtopic", None
    return None


def _ttm_or_bad_quarter(rawq: str) -> bool:
    ql = (rawq or "").lower()
    return bool(_re.search(r"\bq\s*(?:0|[5-9])\b", ql) or "trailing twelve" in ql
                or "trailing 12" in ql or _re.search(r"\bttm\b", ql))


def _post_lane(q, ent) -> str:
    """Lanes decided once the company is known, in precedence order."""
    rawq = q.raw_question or ""
    vague = _is_vague(q)
    if q.metric is None and not q.metrics and _is_coverage_question(rawq):
        return "coverage"
    if vague and (_is_overview(rawq) or _is_identity_question(rawq, ent)):
        return "overview"
    if _nongaap_blocked(q):
        return "nongaap"
    if q.kind == "prose":
        return "prose"
    if _ttm_or_bad_quarter(rawq):
        return "badperiod"
    if _untracked_is_primary(q):
        return "untracked"
    if q.metric is None:
        return "need_metric" if q.calc in ("growth", "margin") else "capability"
    left = _unexplained_tokens(q, ent, _companies_named(rawq))
    if len(left) >= 4:
        return "capability"          # chatty / rambling: reply in words, no figure
    if left:
        return "unexplained"
    return "factual"


def route(question: str, entity=None) -> str:
    """The lane a question takes (for tests and the trace) — read from the same
    answer() run, so it can never drift from real behavior."""
    a = answer(question, entity=entity)
    for s in a.trace.steps:
        if s.startswith("lane: "):
            return s[6:]
    return "?"


# --------------------------------------------------------------------------- #
# Conversational intent router (see design_doc: three lanes). Grounding rigor is
# proportional to the factual weight of the reply:
#   • social / capability  -> free language, asserts NO filing fact (nothing to
#     ground), so it's safe to be warm;
#   • overview / meta       -> grounded in the filing's own prose + banked metadata;
#   • factual               -> the full deterministic trust spine (unchanged).
# Two trust boundaries are enforced here: investment ADVICE/opinion and the
# off-scope stock-price/valuation asks are declined (a 10-K/10-Q can't answer them).
# --------------------------------------------------------------------------- #
_GREETING_WORDS = ("hello", "hi", "hey", "yo", "howdy", "sup", "greetings", "hiya",
                   "heya", "hola", "bonjour", "ciao", "aloha", "shalom", "namaste",
                   "cheers", "welcome", "morning", "evening", "bye", "goodbye",
                   "cya", "later", "lol", "cool", "great", "awesome", "ok", "okay",
                   "afternoon", "peace", "whatsup", "wassup", "thx", "ty", "thanks")
_SOCIAL_CUES = ("how are you", "how's it going", "hows it going", "how do you do",
                "how are things", "what's up", "whats up", "good morning",
                "good afternoon", "good evening", "good night", "nice to meet",
                "thank you", "appreciate", "good bot", "you're great",
                "youre great", "you're awesome", "youre awesome", "nice work",
                "good to see")
_CAPABILITY_CUES = ("what can you do", "what do you do", "who are you", "what are you",
                    "how do you work", "how does this work", "what can i ask",
                    "what should i ask", "how can you help", "what can you help",
                    "what questions", "capabilities", "are you a robot",
                    "are you human", "are you an ai", "are you ai", "are you a bot",
                    "what data do you have", "what do you know", "what can i do here",
                    "explain yourself", "how smart", "what kind of questions",
                    "what sort of questions", "help me get started", "how do i use",
                    "do you support", "do you use", "can you access", "live data",
                    "real-time", "real time", "internet", "browse the web", "chatgpt",
                    "how are you different", "where does your data", "data source",
                    "how accurate", "can i trust you")
_ADVICE_CUES = ("should i buy", "should i sell", "should i invest", "should i own",
                "good investment", "good buy", "worth buying", "worth investing",
                "buy or sell", "price target", "overvalued", "undervalued",
                "valuation", "is it cheap", "is it expensive",
                "bullish", "bearish", "recommend", "will the stock", "stock go up",
                "good stock", "outperform",
                "invest in", "is it a buy")
_OVERVIEW_CUES = ("tell me about", "overview", "what do you know about", "know about", "describe the company",
                  "describe this company", "what does the company do",
                  "what do they do", "about the company", "give me a summary",
                  "summary of the", "introduce", "what kind of company",
                  "elevator pitch")   # NOT bare "what is/who is" (too broad —
                                       # they'd swallow off-topic questions)
# Clearly non-filing / entertainment requests. Matched as whole phrases/words so
# they catch "tell me a story" but not "the story behind the revenue decline".
_OFFTOPIC_RE = _re.compile(
    r"\btell (?:me )?(?:a|an|any)\s+(?:story|joke|tale)\b|"
    r"\b(?:joke|poem|riddle|limerick|haiku|song)\b|\bsing\b|\brecipe\b|"
    r"\bplay a game\b|\btrivia\b|\bweather\b|\bsports?\b|\bhoroscope\b|"
    r"\bwrite (?:me )?(?:a|an)\b|\bcapital of\b|\bsuper bowl\b|\bwho will win\b|"
    r"\bmy\s+(?:taxes|tax\s+return|homework|essay|resume|code)\b|\bcount\s+to\b|"
    r"\bwhat'?s\s+\w+\s+plus\s+\w+\b|\bfavou?rite\b|"
    r"\bworld cup\b|\belection\b", _re.I)


def _has_word(t: str, words) -> bool:
    return any(_re.search(r"\b" + _re.escape(w) + r"\b", t) for w in words)


_ADVICE_RE = _re.compile(
    r"\b(?:cheap|expensive|pricey|bargain)\b|\bgood\b[^.?!]{0,30}\binvest|"
    r"\bworth\s+(?:it|owning|holding|a\s+look)\b|\bshould\s+i\b|"
    r"\b(?:buy|sell|hold|short)\s+(?:the\s+)?(?:stock|shares)\b|"
    r"\b(?:would|should|do|will)\s+(?:you|i|we)\s+(?:buy|sell|hold|short|invest)\b|"
    r"\bbull(?:ish)?\b|\bbear(?:ish)?\b|\b(?:i|we)\s+should\s+(?:buy|sell|invest|hold)\b|"
    r"\bover[- ]?priced\b|\bunder[- ]?priced\b|\bfair\s+price\b|\bstock\s+(?:tips?|picks?)\b|"
    r"\bhot\s+stock\b|\bbetter\s+investment\b|\bworth\s+investing\b|"
    r"\ba\s+(?:buy|sell|hold)\b|\bload\s+up\b|\bsmart\s+to\s+(?:buy|invest|sell)\b|"
    r"\bstrong\s+(?:buy|sell)\b|\bgood\s+time\s+to\s+(?:buy|sell|invest)\b|"
    r"\bfair\s+value\s+of\s+(?:the\s+)?(?:stock|shares?|company)\b|\bintrinsic\s+value\b",
    _re.I)


def _is_advice(text: str) -> bool:
    t = (text or "").lower()
    return any(c in t for c in _ADVICE_CUES) or bool(_ADVICE_RE.search(t))


def _is_capability(text: str) -> bool:
    return any(c in (text or "").lower() for c in _CAPABILITY_CUES)


def _is_offtopic(text: str) -> bool:
    return bool(_OFFTOPIC_RE.search(text or ""))


_GREETING_TYPO_RE = _re.compile(
    r"^\W*(?:h+e+l+o+w?|hel+o+|hal+o+|hul+o+|h+i+|he+y+|hiya+|yo+|sup+|gm+)\b")


def _is_social(text: str) -> bool:
    t = (text or "").lower()
    if t.strip(" !.?") in ("gm", "gn", "see you", "see you later", "talk later"):
        return True
    return (_has_word(t, _GREETING_WORDS) or any(c in t for c in _SOCIAL_CUES)
            or bool(_GREETING_TYPO_RE.match(t)))


# "what is X / what does the company do / what's this company doing / what
# business are they in" — asks WHAT the company is or does. These must hit the
# honest OVERVIEW lane (which says we don't carry a business description), NOT
# prose retrieval, which would narrate a tangential risk factor as the business.
_OVERVIEW_RE = _re.compile(
    # "what does this/the company DO/MAKE/SELL" — the company IS the subject. NOT
    # "what does the company SAY/THINK/DISCLOSE about <topic>" (that is a prose
    # question and must reach the retriever), so the verb is constrained.
    r"\bwhat'?s?\s+this\s+compan(?:y|ies)\s+(?:do|doing|about)\b|"
    r"\bwhat\s+(?:does|do)\s+(?:this|the)\s+compan(?:y|ies)\s+"
    r"(?:do|doing|make|makes|sell|sells|produce|provide|offer)\b|"
    r"\bwhat\s+(?:do|does|are|is)\s+(?:they|it)\s+(?:do|doing|make|makes|sell|"
    r"sells|produce|provide|offer)\b|"
    r"\bwhat\s+(?:business|industry|sector)\b|"
    r"\bwhat\s+does\s+it\s+(?:do|make|sell|produce)\b|"
    r"\bwhat\s+(?:the|this)\s+company\s+(?:does|do|makes|make|sells|sell)\b|"
    r"^\W*what\s+(?:does|do)\s+[\w.&']+(?:\s+[\w.&']+)?\s+(?:do|make|sell|produce)\s*\??\s*$",
    _re.I)


def _is_overview(text: str) -> bool:
    t = (text or "").lower()
    return any(c in t for c in _OVERVIEW_CUES) or bool(_OVERVIEW_RE.search(t))


def _is_identity_question(text: str, ent) -> bool:
    """"What is ASTS?" / "who is Alphabet?" — a SHORT identity question whose whole
    subject is the company in focus. Must stay narrow: it fires only when what
    follows "what is / who is" is essentially just the company (its ticker, a name
    word, or "the company/it"), so it never swallows a longer prose question that
    merely mentions the name ("what are the competition risks Tesla faces?") nor a
    general-knowledge "what is 2+2 / the capital of France?"."""
    if ent is None:
        return False
    t = (text or "").lower().strip()
    mobj = _re.match(r"(?:okay[,\s]+)?(?:what\s+is|what's|whats|who\s+is|who's|"
                     r"what\s+are|tell me who)\s+", t)
    if not mobj:
        return False
    rest = _re.sub(r"^(?:a|an|the|this)\s+", "", t[mobj.end():].strip(" ?.!'\""))
    if len(rest.split()) > 3:                       # a longer question, not identity
        return False
    if not rest or rest in ("company", "it", "they", "this company", "the company",
                            "business"):
        return True                                 # "what is this company?"
    namewords = {w.lower().strip(".,") for w in ent.title.split()
                 if len(w) > 3 and w.lower() not in
                 ("inc", "inc.", "corp", "corp.", "company", "holdings", "group",
                  "technologies", "the", "co", "co.", "ltd", "plc")}
    rtoks = {_re.sub(r"'s$", "", w.strip(".,?!:;")) for w in rest.split()}
    rtoks -= {"", "the", "inc", "corp", "co"}
    ident = namewords | {ent.ticker.lower()} | {a for a, tk in _ENT_ALIASES.items()
                                                 if tk == ent.ticker}
    return bool(rtoks) and rtoks <= ident


def _answer_overview(ent, tr) -> Answer:
    """Lane 2: a warm overview built from what we ACTUALLY banked — the coverage
    (forms + fiscal years) and which narrative sections are available — not a
    business description. We deliberately do NOT synthesize a 'what the company
    does' blurb: we bank Risk Factors (Item 1A) and MD&A (Item 7), not Item 1
    Business, so passing a risk-factor snippet off as a company description would
    be misleading. Honest about scope beats a plausible-but-wrong overview."""
    facts = bank.facts_for(ent.cik)
    annual = sorted({f["fiscal_year"] for f in facts
                     if f.get("period_kind") == "annual"})
    forms = sorted({f["form"] for f in facts})
    sections = sorted({c.get("section") for c in bank.load_prose()
                       if c.get("cik") == ent.cik})
    sec_names = {"risk_factors": "risk factors", "mdna": "management's discussion (MD&A)"}
    have = [sec_names.get(s, s) for s in sections if s]

    parts = [f"Here's what I have on {ent.title} ({ent.ticker})."]
    if annual:
        span = (f"{annual[0]}" if annual[0] == annual[-1]
                else f"{annual[0]}–{annual[-1]}")
        parts.append(f"I've banked its {', '.join(forms)} filings for fiscal "
                     f"years {span}, so I can pull reported financials straight "
                     "from the statements")
        parts[-1] += (f" and summarize its {' and '.join(have)}." if have else ".")
    parts.append("I don't carry a general business description (I index the "
                 "financial statements, Risk Factors and MD&A — not the Item 1 "
                 "'Business' section), so ask me something specific and I'll "
                 "ground it in the filing: revenue, net income, EPS, free cash "
                 "flow, margins, growth over time, or the key risks it discloses.")
    tr.steps.append(f"overview: forms={forms} annual={annual} prose={sections}")
    return Answer(" ".join(parts), 0.6, "answered", tr)


# --------------------------------------------------------------------------- #
# CONVERSE lane: the LLM writes the reply, but FENCED to a persona that may not
# state any figure or filing fact — so there's nothing to ground. This is the
# same division as _phrase (LLM for language only); here the leash is "talk about
# yourself and scope, never a fact" instead of "restate these grounded numbers".
# Offline, an intent-specific template (distinct per intent, so replies don't
# repeat one block). No number ever originates from the LLM in either lane.
# --------------------------------------------------------------------------- #
_CHAT_SYS = (
    "You are the assistant for a tool that answers questions about the SEC filings "
    "(10-K / 10-Q) of a fixed set of US public companies whose filings this tool "
    "has already indexed. (These are ordinary public companies across many "
    "industries — NOT banks or financial institutions specifically; do not say "
    "'banks'.) Describe your scope ACCURATELY: for a covered company you pull "
    "figures from the three core financial statements (income statement, balance "
    "sheet, cash flows) — revenue, net income, EPS, cash flow, margins, growth, "
    "and derived measures like EBITDA / free cash flow — and you summarize the "
    "Risk Factors (Item 1A) and MD&A (Item 7), always with a page citation. You do "
    "NOT have: a business description (Item 1), executive compensation, the "
    "detailed notes, or any market / stock-price data — don't claim you can answer "
    "those. STYLE: reply in 1–3 plain-prose sentences, no lists, no headings, and "
    "NO emoji. Do NOT open with a greeting ('Hi there', 'Hey', 'Hello') unless the "
    "user actually greeted you first — otherwise get straight to the substance, and "
    "vary your wording so you never sound scripted. HARD RULES: never state a "
    "specific financial figure or fact about a company (you don't have them in "
    "front of you); if they want data, invite them to ask and say you'll pull it "
    "from the filing. Never give investment advice, price targets, or opinions on a "
    "stock. If the request is outside that scope, gently steer back."
)
_CHAT_HINT = {
    "advice": "The user wants investment advice / an opinion on the stock — decline "
              "warmly and offer to show fundamentals instead.",
    "need_company": "The user asked something but named no company — warmly ask them "
                    "to name one (e.g. Nvidia or GOOG) or pick it on the left.",
    "offtopic": "The user asked something outside SEC filings — gently steer back.",
    "marketdata": "The user asked for market data (stock price, P/E, market cap, "
                  "dividend yield) — explain that filings don't contain it, so you "
                  "don't have it, and offer the filing-based figures instead (EPS, "
                  "net income, revenue, cash flow).",
}
_CHAT_FALLBACK = {
    "smalltalk": "Doing well, thanks for asking! I'm here whenever you want to dig "
                 "into {co} filings — ask about a figure like revenue or free cash "
                 "flow, or the risk factors, and I'll pull it straight from the "
                 "10-K/10-Q with a citation.",
    "capability": "I answer questions about {co} SEC filings (10-K / 10-Q) — revenue, "
                  "net income, EPS, cash flow, margins, growth over time, or the risk "
                  "factors — and I show you the exact page each number comes from. "
                  "What would you like to know?",
    "advice": "I don't give investment advice, price targets, or opinions on a stock "
              "— those need market data and judgment a 10-K/10-Q doesn't contain. I "
              "can show you the fundamentals instead: {co} revenue, margins, cash "
              "flow, growth, or risk factors.",
    "need_company": "Happy to help — I just need to know which company. Name one in "
                    "your question (like “Nvidia” or “GOOG”) or pick it on the left, "
                    "and ask again. I can cover revenue, net income, EPS, free cash "
                    "flow, growth over time, or the risk factors.",
    "marketdata": "I don't have market data — stock prices, P/E, market cap or "
                  "dividend yield aren't in a 10-K/10-Q, so I won't guess them. From "
                  "the filing I can give {co} EPS, net income, revenue or cash flow, "
                  "each with a page citation.",
    "offtopic": "I'm focused on US public companies' SEC filings (10-K / 10-Q), so I "
                "can't help with that — but ask me about {co} financials or risks and "
                "I'll pull it straight from the filing.",
}


def _chat_draft(q, ent, intent: str) -> str:
    """The code-written converse reply for an intent (also the offline text)."""
    co = f"{ent.title}'s" if ent else "a company's"
    low = (q.raw_question or "").lower()
    if (intent == "smalltalk" and not _re.search(r"how\s+(?:are|r)\s+(?:you|u)|how's it|"
                                                 r"hows it|what'?s up|wassup", low)
            and _is_social(q.raw_question) and not _re.search(
                r"\b(?:thanks?|thx|ty|appreciate|cheers|bye|goodbye|cya|later)\b", low)):
        who = (f"Want to dig into {ent.title}'s filings? Ask about a figure like revenue "
               "or free cash flow, or its risk factors." if ent else
               "Which company would you like to look at? Name one, or pick it on the left.")
        return f"Hi! {who}"
    if intent == "smalltalk" and _re.search(r"\b(?:thanks?|thx|ty|appreciate|cheers)\b", low):
        return (f"You're welcome! Ask me anything else about {co} filings whenever "
                "you're ready.")
    if intent == "smalltalk" and _re.search(r"\b(?:bye|goodbye|cya|later)\b", low):
        return "Goodbye — come back any time you want to dig into a filing."
    if intent == "need_company":
        names = ", ".join(t for _tk, (t, _c) in sorted(_banked_index().items())[:4])
        return (f"Happy to help — which company? I cover a fixed set, for example {names}. "
                "Name one in your question (or pick it on the left) and I'll pull the "
                "figure or the risk factors from its 10-K/10-Q.")
    return _CHAT_FALLBACK.get(intent, _CHAT_FALLBACK["capability"]).format(co=co)


def _chat(q, ent, intent: str, tr) -> Answer:
    """Converse lanes. The reply is DRAFTED by code (it states no filing fact and
    only the real scope); the LLM may rewrite the draft for tone and fit, through
    the same verifier as every other lane — no free generation, so the model can't
    claim coverage, capabilities or company facts the draft doesn't make."""
    tr.steps.append(f"converse lane: intent={intent} (no filing fact asserted)")
    status = ("hedged" if intent in ("advice", "need_company", "marketdata")
              else "answered")
    draft = _chat_draft(q, ent, intent)
    years = " ".join(_re.findall(r"\b(?:19|20)\d{2}\b", q.raw_question or ""))
    text = voice.render("chat", draft, q.raw_question, trace=tr, max_tokens=180,
                        allowed_text=years)
    return Answer(text, 0.6, status, tr)


# Tier 1 — an explicit ADJUSTMENT modifier: the user wants management's bespoke
# variant (adjusted / pro-forma / normalized …). Always a company-reported figure
# we don't parse, even on a base we could derive -> refuse.
_NONGAAP_ADJUST_CUES = ("adjusted ", "pro forma", "pro-forma", "proforma",
                        "normalized", "core earnings", "organic revenue",
                        "constant currency")
# Tier 2 — a bare "non-GAAP" mention (may be descriptive, e.g. "why is FCF
# non-GAAP?"): only blocks when it targets a figure we can't derive.
_NONGAAP_WORD_CUES = ("non-gaap", "non gaap", "nongaap")
_NONGAAP_CUES = _NONGAAP_ADJUST_CUES + _NONGAAP_WORD_CUES


def _is_nongaap_request(text: str) -> bool:
    return any(c in (text or "").lower() for c in _NONGAAP_CUES)


# cues for the measures we DERIVE from GAAP inputs (FCF, EBITDA, the ratios) —
# used so a question naming one isn't blocked by the non-GAAP guard.
_DERIVABLE_CUES = tuple(
    cue for m in COMPUTED for cue in _METRIC_CUES.get(m, [])
)


def _mentions_derivable(text: str) -> bool:
    t = (text or "").lower()
    return any(_re.search(r"\b" + _re.escape(cue) + r"\b", t)
               for cue in _DERIVABLE_CUES)


def _answer_coverage(ent, tr) -> Answer:
    """Answer a meta question about which periods/filings are banked, from the
    Bank's own metadata (never guessed)."""
    facts = bank.facts_for(ent.cik)
    if not facts:
        return Answer(f"I don't have any filings banked for {ent.title} yet.",
                      0.0, "refused", tr)
    annual = sorted({f["fiscal_year"] for f in facts
                     if f.get("period_kind") == "annual"})
    ends = sorted({f["period_end"] for f in facts
                   if f.get("period_kind") == "annual"})
    forms = sorted({f["form"] for f in facts})
    interim = sorted({f["period"] for f in facts
                      if f.get("period_kind") in ("quarter", "ytd")})
    ye = ""
    if ends:
        mm, dd = int(ends[-1][5:7]), int(ends[-1][8:10])
        ye = f"{_MONTHS_FULL[mm]} {dd}"
    parts = []
    if annual:
        span = (f"{annual[0]}" if annual[0] == annual[-1]
                else f"{annual[0]}–{annual[-1]}")
        parts.append(f"annual figures for fiscal year{'s' if len(annual) > 1 else ''} "
                     f"{span}" + (f" (fiscal year ends {ye})" if ye else ""))
    if interim:
        parts.append(f"interim figures for {', '.join(interim)}")
    body = (f"The data I have for {ent.title} comes from its "
            f"{', '.join(forms)} filing{'s' if len(forms) > 1 else ''} — "
            + "; ".join(parts) + ". Every figure I report is tied to its specific "
            "period, which I show alongside each answer.")
    tr.steps.append(f"coverage: forms={forms} annual={annual} fiscal_year_end={ye}")
    return Answer(body, 0.9, "answered", tr)


def _answer_prose(ent, q, tr) -> Answer:
    """Route a prose question to the grounded retriever + narrator."""
    query = q.raw_question + (f" {q.prose_hint}" if q.prose_hint else "")
    pa = prose.answer_prose(query, ent.cik)
    tr.steps.append(f"prose retrieval: status={pa.status} "
                    f"sections={pa.section_hits} conf={pa.confidence}")
    # the narrator is an LLM too: every number and page cite in its narration
    # must come from the retrieved excerpts, else show the top passage verbatim
    if pa.status != "refused" and pa.passages and llm.available():
        pages = {p_["page_start"] for p_ in pa.passages} | {
            p_.get("page_end") for p_ in pa.passages if p_.get("page_end")}
        verdict = voice.verify_prose(pa.text, pa.passages, pages, require_quotes=True)
        tr.steps.append(f"voice[prose]: {'narration verified' if verdict == 'OK' else 'fallback to verbatim passage — ' + verdict}")
        if verdict != "OK":
            top = pa.passages[0]
            body = top["text"][:700] + ("..." if len(top["text"]) > 700 else "")
            pa.text = (f"Here is the most relevant passage, verbatim (p.{top['page_start']}, "
                       f"{top['section_name']}):\n{body}")
    # A prose miss (nothing relevant, or off-topic that slipped a narrative cue)
    # steers back warmly rather than dead-ending on "I couldn't find a passage".
    if pa.status == "refused":
        return Answer(
            f"I couldn't find that in {ent.title}'s banked narrative sections "
            "(Risk Factors / MD&A). I can summarize the risks it does disclose, or "
            "pull financials like revenue, margins, cash flow, or growth from the "
            "statements.", 0.2, "refused", tr)
    stem = pa.passages[0]["filing_stem"] if pa.passages else ""
    form = pa.passages[0]["form"] if pa.passages else ""
    for c in pa.citations:
        tr.citations.append({
            "filing": stem, "form": form, "page": c["page_start"],
            "label": c["section_name"], "period": "", "cell": "",
            "unit": "", "scale": "", "score": c.get("score"),
            "snippet": c.get("snippet"),
        })
    return Answer(pa.text, pa.confidence, pa.status, tr, value=None,
                  passages=pa.passages)


def _answer_margin(ent, q, year, tr) -> Answer:
    # A computed numerator (EBITDA margin, FCF margin): compute it, divide by
    # revenue. Must come first — a COMPUTED key isn't in METRICS (was a KeyError).
    if q.metric in COMPUTED:
        r = _metric_figure(ent, q, year, q.metric)
        name = COMPUTED[q.metric][0]
        if not r["ok"]:
            return Answer(f"I can't compute {name.lower()} for {ent.title} in "
                          f"{_period_desc(q, year)}, so I won't compute its margin.",
                          0.0, "refused", tr)
        f_rev, s2, _ = find_fact(ent.cik, "revenue", year, q.period_kind, q.quarter, instant_end=q.instant_end)
        if s2 != "found":
            return Answer(f"I'm missing grounded revenue for the {name.lower()} "
                          f"margin in {_period_desc(q, year)}.", 0.0, "refused", tr)
        if not _abs(f_rev) or f_rev.get("is_dash"):
            return Answer(f"{ent.title} reports no revenue for {_period_desc(q, year)}, "
                          f"so a {name.lower()} margin is undefined.", 0.0, "refused", tr)
        for f in r["facts"]:
            tr.cite(f)
        tr.cite(f_rev)
        margin = r["num"] / _abs(f_rev) * 100
        tr.steps.append(f"{name} margin = {r['value_str']} / {_fmt_money(f_rev)} "
                        f"= {margin:.1f}%")
        text = _phrase(q.raw_question,
                       f"{ent.title}'s {name.lower()} margin for {_period_desc(q, year)} "
                       f"was {margin:.1f}% ({r['value_str']} / {_fmt_money(f_rev)}; "
                       f"{name} is non-GAAP, computed from GAAP inputs).", tr)
        return Answer(text, min([f.get("confidence", 0.9)
                                 for f in r["facts"] + [f_rev]]),
                      "answered", tr, margin)
    spec = METRICS[q.metric]
    if q.pct_of_revenue and spec.statement != "balance_sheet":
        r = _margin_figure(ent, q, year, q.metric)
        if not r["ok"]:
            return Answer(f"I can't ground {r['name'].lower()} for {ent.title} in "
                          f"{_period_desc(q, year)}, so I won't compute it.", 0.0,
                          "refused", tr)
        for f in r["facts"]:
            tr.cite(f)
        a_, rev = r["facts"][0], r["facts"][-1]
        tr.steps.append(f"{r['name']} = {_fmt_money(a_)} / {_fmt_money(rev)} = {r['value_str']}")
        return Answer(_phrase(q.raw_question, f"{ent.title}'s {spec.name} was "
                              f"{r['value_str']} of revenue for {_period_desc(q, year)} "
                              f"({_fmt_money(a_)} / {_fmt_money(rev)}).", tr),
                      min(f.get("confidence", 0.9) for f in r["facts"]), "answered", tr, r["num"])
    if not spec.margin_eligible:
        return Answer(f"{spec.name} isn't a margin metric — I can express it as a share "
                      "of revenue if you ask for it that way.", 0.2, "hedged", tr)
    f_num, s1, _ = find_fact(ent.cik, q.metric, year, q.period_kind, q.quarter, instant_end=q.instant_end)
    f_rev, s2, _ = find_fact(ent.cik, "revenue", year, q.period_kind, q.quarter, instant_end=q.instant_end)
    if s1 != "found" or s2 != "found":
        return Answer(f"I'm missing a grounded value needed for the {spec.name} "
                      f"margin in {_period_desc(q, year)}.", 0.0, "refused", tr)
    if not f_rev["value"] or f_rev.get("is_dash"):
        # revenue is zero / a dash (e.g. ASTS 2023) — a margin is undefined.
        return Answer(f"{ent.title} reports no revenue (it’s a dash / zero) for "
                      f"{_period_desc(q, year)}, so a {spec.name.lower()} margin is "
                      "undefined.", 0.0, "refused", tr)
    tr.cite(f_num); tr.cite(f_rev)
    margin = f_num["value"] / f_rev["value"] * 100
    tr.steps.append(f"margin = {f_num['raw_cell_text']} / "
                    f"{f_rev['raw_cell_text']} = {margin:.1f}%")
    text = _phrase(q.raw_question,
                   f"{ent.title}'s {spec.name} margin for {_period_desc(q, year)} "
                   f"was {margin:.1f}% ({_fmt_money(f_num)} / {_fmt_money(f_rev)}).",
                   tr)
    return Answer(text, min(f_num["confidence"], f_rev["confidence"]),
                  "answered", tr, margin)


# --------------------------------------------------------------------------- #
# write_answer: phrase over verified numbers (LLM optional; never re-derives)
# --------------------------------------------------------------------------- #
def _metric_labels(sentence: str) -> list:
    """Each metric NAMED in the draft must still be named in the rewrite (by its
    display name or any of its cue synonyms) — numbers alone can't catch
    "operating income" being relabeled "revenue"."""
    low, groups = sentence.lower(), []
    for key, spec in METRICS.items():
        if spec.name.lower() in low:
            groups.append((spec.name, *_METRIC_CUES.get(key, [])))
    for key, (name, *_r) in COMPUTED.items():
        if name.lower() in low:
            groups.append((name, *_METRIC_CUES.get(key, [])))
    return groups


def _phrase(question: str, grounded_sentence: str, tr: Trace) -> str:
    """The LLM may rewrite the grounded sentence for tone; `voice.render` verifies
    the rewrite against it (every number both ways, direction, no opinion, metric
    still named, caveats re-attached verbatim) and falls back to the sentence."""
    dates = " ".join(c["period_end"] for c in tr.citations if c.get("period_end"))
    return voice.render("answer", grounded_sentence, question,
                        labels=_metric_labels(grounded_sentence), trace=tr,
                        allowed_text=dates)


if __name__ == "__main__":
    for Q in [
        "What was Tesla's total revenue in 2025?",
        "What was Tesla's net income in 2025?",
        "What was Tesla's diluted EPS in 2025?",
        "How much cash did Tesla have at the end of 2025?",
        "How did Tesla's revenue grow from 2024 to 2025?",
        "What was Tesla's operating margin in 2025?",
        "What was Tesla's net income in 2019?",      # not in bank -> refuse
    ]:
        a = answer(Q)
        print(f"\nQ: {Q}")
        print(f"   [{a.status} | conf {a.confidence:.2f}] {a.text}")
        for c in a.trace.citations:
            print(f"      cite: {c['form']} p{c['page']} '{c['label']}' "
                  f"{c['period']}={c['cell']}")
