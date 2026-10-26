"""Project configuration and constants.

Single source of truth for paths, the LLM model, and EDGAR access policy.
EDGAR is treated as a CONSTRAINED resource: a cached static file + low-volume
lookups, never a per-query pipeline (see design_doc/architecture.md).
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILINGS_DIR = ROOT / "filings"      # raw source documents (PDF/HTML)
STORE_DIR = ROOT / "store"          # THE BANK: parsed facts + prose index
CACHE_DIR = ROOT / ".cache"         # static EDGAR files, etc.
for _d in (FILINGS_DIR, STORE_DIR, CACHE_DIR):
    _d.mkdir(exist_ok=True)

# --- LLM (Anthropic): cheapest model; the LLM only does light language tasks. ---
ANTHROPIC_MODEL = "claude-haiku-4-5"
# Reserved escalation for the messy-table-structuring fallback ONLY:
ANTHROPIC_MODEL_ESCALATE = "claude-sonnet-5-5"

# --- EDGAR: be polite. SEC fair-use wants a User-Agent with real contact info. ---
# Set EDGAR_CONTACT to your email before running against EDGAR.
EDGAR_CONTACT = os.environ.get("EDGAR_CONTACT", "set-me@example.com")
EDGAR_UA = f"d71-takehome/0.1 ({EDGAR_CONTACT})"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
EDGAR_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
EDGAR_MIN_INTERVAL_S = 0.2  # throttle: ~5 requests/sec cap

# --- HTML -> PDF rendering via the installed Chrome (zero extra deps). ---
# We render EDGAR HTML to PDF on purpose: the hard problem is messy PDF tables,
# not HTML <table> tags. Parsing a rendered PDF confronts the real layout problem.
CHROME_BIN = os.environ.get(
    "CHROME_BIN",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
