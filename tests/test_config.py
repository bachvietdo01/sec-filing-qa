"""Unit tests for config.py — the single source of truth for models, EDGAR
policy, and paths. Asserts the load-bearing constants exist and are sane."""
from __future__ import annotations

from pathlib import Path

import config


def test_anthropic_model_is_haiku():
    assert config.ANTHROPIC_MODEL == "claude-haiku-4-5"


def test_escalate_model_set_and_distinct():
    assert isinstance(config.ANTHROPIC_MODEL_ESCALATE, str)
    assert config.ANTHROPIC_MODEL_ESCALATE          # non-empty
    assert config.ANTHROPIC_MODEL_ESCALATE != config.ANTHROPIC_MODEL


def test_edgar_ua_includes_contact():
    assert isinstance(config.EDGAR_UA, str)
    assert config.EDGAR_CONTACT in config.EDGAR_UA


def test_submissions_url_has_cik_placeholder():
    assert "{cik10}" in config.EDGAR_SUBMISSIONS_URL
    # it must actually format with a zero-padded CIK
    assert config.EDGAR_SUBMISSIONS_URL.format(cik10="0000320193").endswith(
        "CIK0000320193.json"
    )


def test_archive_url_has_all_placeholders():
    for ph in ("{cik}", "{acc_nodash}", "{doc}"):
        assert ph in config.EDGAR_ARCHIVE_URL
    built = config.EDGAR_ARCHIVE_URL.format(
        cik=1318605, acc_nodash="000162828025000000", doc="tsla.htm"
    )
    assert built.endswith("1318605/000162828025000000/tsla.htm")


def test_tickers_url_is_sec_json():
    assert config.EDGAR_TICKERS_URL.startswith("https://")
    assert config.EDGAR_TICKERS_URL.endswith(".json")
    assert "sec.gov" in config.EDGAR_TICKERS_URL


def test_min_interval_positive():
    assert isinstance(config.EDGAR_MIN_INTERVAL_S, (int, float))
    assert config.EDGAR_MIN_INTERVAL_S > 0


def test_dirs_exist_as_paths():
    for d in (config.FILINGS_DIR, config.STORE_DIR, config.CACHE_DIR):
        assert isinstance(d, Path)
        assert d.exists()
        assert d.is_dir()


def test_chrome_bin_is_str():
    assert isinstance(config.CHROME_BIN, str)
    assert config.CHROME_BIN            # non-empty
