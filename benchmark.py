"""A small hand-verified benchmark: the objective trust check.

Each case pins a question to a value I read by hand from Tesla's FY2025 10-K
(the consolidated statements), plus the expected engine status. The engine must
land within a tolerance band and must *refuse* the out-of-corpus question rather
than fabricate. This is what lets us claim the system is trustworthy instead of
merely asserting it.

Run:  python3 benchmark.py   ->   prints PASS/FAIL per case + a summary.
"""
from __future__ import annotations

from dataclasses import dataclass

import core


@dataclass
class Case:
    question: str
    expect_status: str          # answered | refused | hedged
    expect_value: float | None  # None when status isn't 'answered'
    tol: float = 0.01           # relative tolerance (1%)
    note: str = ""


# Values hand-verified from the FY2025 10-K consolidated statements.
CASES = [
    Case("What was Tesla's total revenue in 2025?", "answered", 94_827,
         note="Total revenues, in $M"),
    Case("What was Tesla's net income in 2025?", "answered", 3_794,
         note="Net income attributable to common stockholders, $M"),
    Case("What was Tesla's operating income in 2025?", "answered", 4_355,
         note="Income from operations, $M"),
    Case("What was Tesla's diluted EPS in 2025?", "answered", 1.08,
         note="Diluted EPS, $/share"),
    Case("How much cash did Tesla have at the end of 2025?", "answered", 16_513,
         note="Cash and cash equivalents, $M"),
    Case("How did Tesla's revenue change from 2024 to 2025?", "answered", -2.931,
         tol=0.02, note="(94,827-97,690)/97,690"),
    Case("What was Tesla's operating margin in 2025?", "answered", 4.593,
         tol=0.02, note="4,355/94,827"),
    Case("What was Tesla's net income in 2019?", "refused", None,
         note="not in the banked filings -> must refuse, not fabricate"),
]


def run(verbose: bool = True) -> tuple[int, int]:
    passed = 0
    for c in CASES:
        a = core.answer(c.question)
        ok = (a.status == c.expect_status)
        if ok and c.expect_value is not None:
            if a.value is None:
                ok = False
            else:
                rel = abs(a.value - c.expect_value) / max(abs(c.expect_value), 1e-9)
                ok = rel <= c.tol
        passed += ok
        if verbose:
            mark = "PASS" if ok else "FAIL"
            got = f"{a.value:,.3f}" if a.value is not None else "-"
            exp = f"{c.expect_value:,.3f}" if c.expect_value is not None else "-"
            print(f"[{mark}] {c.question}")
            print(f"       status={a.status} (exp {c.expect_status})  "
                  f"value={got} (exp {exp})")
            if not ok:
                print(f"       -> {a.text}")
    if verbose:
        print(f"\n{passed}/{len(CASES)} cases passed")
    return passed, len(CASES)


if __name__ == "__main__":
    import sys
    p, n = run()
    sys.exit(0 if p == n else 1)
