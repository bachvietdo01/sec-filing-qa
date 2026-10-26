"""The voice boundary: every LLM rewrite is verified against the code's draft.
Bad rewrites must be rejected (100%); faithful rewordings must pass, or the
assistant stays robotic. No live LLM: llm.complete is faked."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core      # noqa: E402
import edgar     # noqa: E402
import llm       # noqa: E402
import voice     # noqa: E402

REV = "Tesla, Inc.'s Total revenue for 2025 was $94,827 million."
NI = "Tesla, Inc.'s Net income for 2025 was $3,794 million."
MARGIN = ("Tesla, Inc.'s Operating income margin for 2025 was 4.6% "
          "($4,355 million / $94,827 million).")
GROWTH = ("Tesla, Inc.'s Net income went from $7,091 million (FY2024) to $3,794 million "
          "(FY2025), a change of -46.5%.")
SERIES = ("HERSHEY CO's Total revenue changed +4.7% over the whole period, from "
          "$11,164,992 thousand (FY2023) to $11,692,576 thousand (FY2025) — a +2.3% "
          "compound annual growth rate over 2 years. Year over year: FY2023→FY2024 "
          "+0.3%; FY2024→FY2025 +4.4%.")
CASH = "Tesla, Inc.'s Cash and cash equivalents as of 2025-12-31 was $16,513 million."


@pytest.mark.parametrize("draft,out", [
    (REV, "Tesla brought in $94,827 million of total revenue in 2025."),
    (REV, "For 2025, Tesla's total revenue came to $94.8 billion."),
    (MARGIN, "In 2025 Tesla kept 4.6% of its revenue as operating income "
             "($4.355 billion on $94.827 billion)."),
    (GROWTH, "Tesla's net income declined 46.5% in FY2025, from $7,091 million in "
             "FY2024 to $3,794 million."),
    (SERIES, "Hershey's total revenue rose +4.7% across FY2023 to FY2025, from "
             "$11,164,992 thousand to $11,692,576 thousand, a +2.3% compound annual "
             "growth rate over two years; year over year that was +0.3% (FY2023→FY2024) "
             "and +4.4% (FY2024→FY2025)."),
    (CASH, "At December 31, 2025, Tesla held $16,513 million in cash and cash equivalents."),
])
def test_faithful_rewrites_pass(draft, out):
    assert voice.verify("answer", draft, out) == "OK"


@pytest.mark.parametrize("draft,out,why", [
    (REV, "Tesla's total revenue for 2024 was $94,827 million.", "year swap"),
    (REV, "Tesla's total revenue for 2025 was $94,827 million, up 12% on demand.", "invented %"),
    (REV, "Tesla's revenue was $94,827 million in 2025; analysts expect $110 billion.", "forecast"),
    (REV, "Tesla's total revenue for 2025 was $94,827 thousand.", "scale swap"),
    (REV, "Tesla's total revenue for 2025 was $95 billion.", "over-rounded"),
    (REV, "Tesla's total revenue for 2025 was a strong $94,827 million.", "opinion"),
    (REV, "Tesla's total revenue for 2025 was roughly $94.8 billion.", "hedge word"),
    (NI, "Tesla posted a net loss of $3,794 million in 2025.", "sign flip"),
    ("Tesla, Inc.'s Diluted EPS for 2025 was $1.08.",
     "Tesla's diluted EPS for 2025 was $11.08.", "substring"),
    (MARGIN, "Operating margin in 2025 was 14.6% ($4,355 million / $94,827 million).", "margin"),
    (GROWTH, "Tesla's net income grew 46.5% from $7,091 million (FY2024) to "
             "$3,794 million (FY2025).", "direction"),
    (GROWTH, "Tesla's net income changed +46.5% from $7,091 million (FY2024) to "
             "$3,794 million (FY2025).", "explicit sign"),
    (SERIES, "Hershey's total revenue grew +4.7% from FY2023 to FY2025 "
             "($11,164,992 thousand to $11,692,576 thousand).", "dropped steps"),
    (CASH, "Tesla held $16,513 million in cash as of December 31, 2024.", "date swap"),
    (REV, "Tesla's revenue for 2025 was $94,827 million because of Model Y demand.", "cause"),
])
def test_bad_rewrites_rejected(draft, out, why):
    assert voice.verify("answer", draft, out) != "OK", why


def test_metric_label_must_survive():
    v = voice.verify("answer", REV, "Tesla's net income for 2025 was $94,827 million.",
                     labels=[("Total revenue", "revenue", "sales")])
    assert v.startswith("metric label")


def test_tails_are_protected_and_reattached(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "complete", lambda s, u, max_tokens=0, model=None:
                        "Tesla's total revenue in FY2025 was $94,827 million.")
    draft = ("Tesla, Inc. — Total revenue: FY2025 $94,827 million. "
             "(I couldn't ground: Net income (FY2026).)")
    out = voice.render("answer", draft, "revenue and net income")
    assert out.endswith("(I couldn't ground: Net income (FY2026).)")


# --- refusals: may be warm, may not answer ----------------------------------- #
REFUSE = ("I don't have Total revenue for 2019 in the banked filings for Tesla, Inc., "
          "so I won't fabricate it. I have it for FY2023–FY2025.")


def test_refusal_warm_rewrite_passes():
    out = ("I'm afraid 2019 isn't in the filings I have for Tesla, so I won't guess at "
           "its revenue — I do have FY2023–FY2025 if one of those helps.")
    assert voice.verify("refusal", REFUSE, out) == "OK"


@pytest.mark.parametrize("out", [
    "Tesla's 2019 revenue was $24,578 million.",                         # answers it
    "I don't have 2019, but it was roughly $24.6 billion.",              # leaks
    "Revenue for 2019 isn't banked; FY2023–FY2025 are available.",       # ok wording...
])
def test_refusal_cannot_answer(out):
    v = voice.verify("refusal", REFUSE, out)
    if "24" in out:
        assert v != "OK"
    else:
        assert v == "OK"


# --- chat leash ---------------------------------------------------------------- #
@pytest.mark.parametrize("out", [
    "Tesla made 97.7 billion in revenue.",
    "Tesla looks undervalued at 60x earnings; I'd accumulate.",
    "Its market cap is near 1.2 trillion.",
    "I cover every S&P 500 company.",
    "Revenue was up 12% last year.",
])
def test_chat_leash_rejects(out):
    assert voice.verify_chat(out, "how are you") != "OK"


def test_chat_leash_allows_plain_talk():
    assert voice.verify_chat("Doing well — which company should we look at? I can pull "
                             "figures from its 10-K or 10-Q.", "how are you") == "OK"


# --- prose narration over excerpts ------------------------------------------- #
def test_prose_numbers_must_come_from_excerpts():
    passages = [{"text": "Revenue increased 18% due to new contracts.", "page_start": 41}]
    assert voice.verify_prose("Revenue rose 18% on new contracts [p.41].", passages, [41]) == "OK"
    assert voice.verify_prose("Revenue rose 25% on new contracts [p.41].", passages, [41]) != "OK"
    assert voice.verify_prose("Revenue rose 18% [p.99].", passages, [41]) != "OK"


# --- round 2: the no-new-content-words leash (chat / refusal / hedge) -------- #
CHAT_DRAFT = ("You're welcome! Ask me anything else about Tesla, Inc.'s filings "
              "whenever you're ready.")


@pytest.mark.parametrize("out", [
    "You're welcome! I've got Tesla's quarterly filings going back to the IPO.",
    "Yep, Tesla is headquartered in Austin — ask me anything else about its filings.",
    "Yes, the company makes electric vehicles and solar panels; ask away.",
    "Sure, I also have their 8-Ks and proxy statements.",
])
def test_chat_cannot_agree_with_a_premise(out):
    assert voice.verify("chat", CHAT_DRAFT, out) != "OK"


def test_chat_natural_restyle_still_passes():
    out = "Happy to help — ask me anything else about Tesla's filings whenever you like."
    assert voice.verify("chat", CHAT_DRAFT, out) == "OK"


def test_refusal_meaning_drift_is_blocked():
    draft = ("I don't track a component figure (“higher revenue”) for Tesla, Inc., and I "
             "won't substitute a different line item for it.")
    out = "I'd need to give you the consolidated total instead, which might not fit."
    assert voice.verify("refusal", draft, out) != "OK"


@pytest.mark.parametrize("draft,out", [
    ("American Airlines Group Inc.'s Diluted EPS for Q2 2026 was $0.11.",
     "American Airlines' diluted EPS for Q2 2026 was $0.11."),                # tiny amount
    ("NVIDIA CORP's Total revenue changed +254.4% over the whole period, from $60,922 "
     "million (FY2024) to $215,938 million (FY2026) — a +88.3% compound annual growth "
     "rate over 2 years.",
     "NVIDIA's total revenue rose +254.4% from $60,922 million in FY2024 to $215,938 "
     "million in FY2026, a +88.3% compound annual growth rate over the two-year period."),
    ("SoFi Technologies, Inc.'s Basic EPS went from $-0.36 (FY2023) to $0.46 (FY2024), a "
     "change of n/a (off a negative base — a % change from a loss would mislead).",
     "SoFi's basic EPS improved from $-0.36 in FY2023 to $0.46 in FY2024, so a % change "
     "isn't meaningful off that negative base."),
])
def test_round2_false_positives_now_pass(draft, out):
    assert voice.verify("answer", draft, out) == "OK"


def test_quarter_end_date_may_replace_the_q_label():
    draft = "American Airlines Group Inc.'s Diluted EPS for Q2 2026 was $0.11."
    out = "For the quarter ended June 30, 2026, American Airlines' diluted EPS was $0.11."
    assert voice.verify("answer", draft, out, allowed_text="2026-06-30") == "OK"
    assert voice.verify("answer", draft, out) != "OK"       # an unsupported date


def test_negative_amount_sign_cannot_vanish():
    draft = "CoreWeave, Inc.'s Operating income for 2025 was $-46 million."
    assert voice.verify("answer", draft, "CoreWeave's operating income for 2025 was $46 million.") != "OK"
    assert voice.verify("answer", draft, "CoreWeave posted an operating loss of $46 million in 2025.") == "OK"


def test_prose_figures_must_be_quoted_verbatim():
    passages = [{"text": "Net cash provided by operating activities decreased by $176 "
                         "million primarily due to a decrease in net income excluding "
                         "non-cash expenses of $737 million.", "page_start": 72}]
    bad = "Net income decreased by $737 million [p.72]."
    good = ('Operating cash flow fell — "net cash provided by operating activities '
            'decreased by $176 million" [p.72].')
    assert voice.verify_prose(bad, passages, [72], require_quotes=True) != "OK"
    assert voice.verify_prose(good, passages, [72], require_quotes=True) == "OK"


def test_prose_quote_may_elide_but_not_invent():
    p = [{"text": "Our service providers have been subject to security breaches and we may "
                  "suffer losses.", "page_start": 23}]
    ok = 'Risks: "service providers...have been subject to security breaches" [p.23].'
    bad = 'Risks: "service providers have been hacked" [p.23].'
    assert voice.verify_prose(ok, p, [23], require_quotes=True) == "OK"
    assert voice.verify_prose(bad, p, [23], require_quotes=True) != "OK"


def test_chat_may_name_the_users_company_but_not_new_facts():
    d = "I don't give investment advice. I can show you NVIDIA CORP's fundamentals instead."
    assert voice.verify("chat", d, "I can't tell you whether to buy NVDA, but I can walk you "
                        "through NVIDIA's fundamentals.", names="NVDA NVIDIA CORP") == "OK"
    assert voice.verify("chat", d, "NVDA is based in Santa Clara; I can show fundamentals.",
                        names="NVDA NVIDIA CORP") != "OK"
