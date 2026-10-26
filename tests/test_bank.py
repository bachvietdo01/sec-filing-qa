"""Unit tests for bank.py — THE BANK (the system of record).

Focus on the trust invariants that matter at this layer:
  * never persist an ungrounded number (write_facts gate),
  * idempotent appends keyed on identity (no silent duplication / drift),
  * deterministic read helpers (filter by CIK, stable label order, sorted periods),
  * lossless round-trips for both stores.

All file IO goes through tmp_path, so the real store/ is never touched.
"""
from __future__ import annotations

import json

import pytest

import bank

CIK = 1318605


# --------------------------------------------------------------------------- #
# write_facts: the grounding gate + round-trip
# --------------------------------------------------------------------------- #
def test_write_facts_rejects_any_ungrounded(tmp_path, mkfact):
    """Trust invariant: a single ungrounded fact must block the whole write."""
    p = tmp_path / "facts.json"
    good = mkfact()
    bad = mkfact(raw_label="Fabricated", grounding_ok=False)
    with pytest.raises(ValueError):
        bank.write_facts([good, bad], p)
    # the gate must fire before anything hits disk
    assert not p.exists()


def test_write_facts_rejects_missing_grounding_flag(tmp_path, mkfact):
    """A fact with no grounding_ok key at all is treated as ungrounded."""
    p = tmp_path / "facts.json"
    f = mkfact()
    f.pop("grounding_ok")
    with pytest.raises(ValueError):
        bank.write_facts([f], p)


def test_write_facts_all_grounded_succeeds_and_round_trips(tmp_path, mkfact):
    p = tmp_path / "facts.json"
    facts = [mkfact(), mkfact(period="FY2024", fiscal_year=2024, value=97_690.0,
                              raw_cell_text="97,690")]
    returned = bank.write_facts(facts, p)
    assert returned == p
    assert p.exists()
    loaded = bank.load_facts(p)
    assert loaded == facts
    # the on-disk form is JSON (deterministic, inspectable)
    assert json.loads(p.read_text()) == facts


def test_load_facts_missing_path_returns_empty(tmp_path):
    assert bank.load_facts(tmp_path / "nope.json") == []


# --------------------------------------------------------------------------- #
# append_facts: de-duplication on the identity key
# --------------------------------------------------------------------------- #
def test_append_same_fact_twice_does_not_grow(tmp_path, mkfact):
    p = tmp_path / "facts.json"
    f = mkfact()
    bank.write_facts([f], p)
    bank.append_facts([f], p)
    assert len(bank.load_facts(p)) == 1


def test_append_new_period_grows(tmp_path, mkfact):
    p = tmp_path / "facts.json"
    f2025 = mkfact()
    f2024 = mkfact(period="FY2024", fiscal_year=2024, value=97_690.0,
                   raw_cell_text="97,690")
    bank.write_facts([f2025], p)
    bank.append_facts([f2024], p)
    loaded = bank.load_facts(p)
    assert len(loaded) == 2
    assert {r["period"] for r in loaded} == {"FY2025", "FY2024"}


def test_append_dedups_on_full_identity_key(tmp_path, mkfact):
    """Key is (cik, filing_stem, statement, raw_label, period). A row whose
    value differs but whose key matches is still a duplicate (not re-banked);
    changing any key component makes it a distinct row."""
    p = tmp_path / "facts.json"
    base = mkfact()
    bank.write_facts([base], p)

    # same key, different value -> NOT added (dedup by identity)
    collision = mkfact(value=0.0, raw_cell_text="0")
    bank.append_facts([collision], p)
    assert len(bank.load_facts(p)) == 1

    # vary one key component at a time -> each is a new row
    bank.append_facts([mkfact(cik=999)], p)
    bank.append_facts([mkfact(filing_stem="OTHER_10K")], p)
    bank.append_facts([mkfact(statement="balance_sheet")], p)
    bank.append_facts([mkfact(raw_label="Different label")], p)
    assert len(bank.load_facts(p)) == 5


# --------------------------------------------------------------------------- #
# read helpers: facts_for / distinct_labels / periods_for
# --------------------------------------------------------------------------- #
def test_facts_for_filters_by_cik(tmp_path, mkfact):
    p = tmp_path / "facts.json"
    bank.write_facts([mkfact(cik=111, company="A"),
                      mkfact(cik=222, company="B"),
                      mkfact(cik=111, company="A", period="FY2024")], p)
    got = bank.facts_for(111, p)
    assert len(got) == 2
    assert all(f["cik"] == 111 for f in got)
    assert bank.facts_for(999, p) == []


def test_distinct_labels_unique_in_insertion_order(tmp_path, mkfact):
    p = tmp_path / "facts.json"
    bank.write_facts([
        mkfact(raw_label="B", statement="operations", period="FY2025"),
        mkfact(raw_label="A", statement="operations", period="FY2024"),
        mkfact(raw_label="B", statement="operations", period="FY2023"),  # dup label
        mkfact(raw_label="C", statement="balance_sheet",
               period="ASOF2025-12-31"),
    ], p)
    # insertion order preserved, duplicates collapsed
    assert bank.distinct_labels(CIK, path=p) == ["B", "A", "C"]


def test_distinct_labels_honors_statement_filter(tmp_path, mkfact):
    p = tmp_path / "facts.json"
    bank.write_facts([
        mkfact(raw_label="B", statement="operations", period="FY2025"),
        mkfact(raw_label="A", statement="operations", period="FY2024"),
        mkfact(raw_label="C", statement="balance_sheet",
               period="ASOF2025-12-31"),
    ], p)
    assert bank.distinct_labels(CIK, statement="operations", path=p) == ["B", "A"]
    assert bank.distinct_labels(CIK, statement="balance_sheet", path=p) == ["C"]


def test_periods_for_sorted_and_unique(tmp_path, mkfact):
    p = tmp_path / "facts.json"
    bank.write_facts([
        mkfact(raw_label="L1", period="FY2025"),
        mkfact(raw_label="L2", period="FY2023"),
        mkfact(raw_label="L3", period="FY2024"),
        mkfact(raw_label="L4", period="FY2023"),  # duplicate period
    ], p)
    assert bank.periods_for(CIK, p) == ["FY2023", "FY2024", "FY2025"]


# --------------------------------------------------------------------------- #
# Prose store round-trip
# --------------------------------------------------------------------------- #
def test_write_prose_load_prose_round_trip(tmp_path):
    pp = tmp_path / "prose.json"
    chunks = [
        {"filing_stem": "TSLA_10K_2025-12-31", "section": "Risk Factors",
         "page_no": 15, "text": "Our business is subject to many risks."},
        {"filing_stem": "TSLA_10K_2025-12-31", "section": "MD&A",
         "page_no": 40, "text": "Revenue decreased year over year.",
         "embedding": [0.1, 0.2, 0.3]},
    ]
    returned = bank.write_prose(chunks, pp)
    assert returned == pp
    assert bank.load_prose(pp) == chunks


def test_load_prose_missing_path_returns_empty(tmp_path):
    assert bank.load_prose(tmp_path / "nope.json") == []
