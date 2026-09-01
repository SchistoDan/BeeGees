"""Tests for barcoding_outcome.py core functions."""
import importlib.util
from pathlib import Path

import pytest
from beegees.utils.configs import get_package_dir

_script = get_package_dir() / "workflow" / "scripts" / "barcoding_outcome.py"
_spec = importlib.util.spec_from_file_location("bc_outcome", _script)
bc_outcome = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc_outcome)


class TestDetermineOutcome:
    def test_pass_when_yes_present(self):
        rows = [{"selected": "YES"}, {"selected": "NO"}]
        assert bc_outcome.determine_outcome(rows) == "PASS"

    def test_partial_when_only_no(self):
        rows = [{"selected": "NO"}, {"selected": "NO"}]
        assert bc_outcome.determine_outcome(rows) == "PARTIAL"

    def test_fail_when_no_rows(self):
        assert bc_outcome.determine_outcome([]) == "FAIL"

    def test_fail_when_neither_yes_nor_no(self):
        rows = [{"selected": "NA"}, {"selected": ""}]
        assert bc_outcome.determine_outcome(rows) == "FAIL"

    def test_yes_takes_priority_over_no(self):
        rows = [{"selected": "NO"}, {"selected": "YES"}, {"selected": "NO"}]
        assert bc_outcome.determine_outcome(rows) == "PASS"

    def test_whitespace_stripped(self):
        rows = [{"selected": " YES "}]
        assert bc_outcome.determine_outcome(rows) == "PASS"


class TestLookupObservedTaxonomy:
    def test_exact_match(self):
        result = bc_outcome.lookup_observed_taxonomy(
            "BOLD123", "BOLD123|Apis mellifera;BOLD456|Bombus terrestris"
        )
        assert result == "BOLD123|Apis mellifera"

    def test_no_match_returns_na(self):
        result = bc_outcome.lookup_observed_taxonomy(
            "BOLD999", "BOLD123|Apis mellifera;BOLD456|Bombus terrestris"
        )
        assert result == "NA"

    def test_empty_hit_returns_na(self):
        assert bc_outcome.lookup_observed_taxonomy("", "BOLD123|Apis mellifera") == "NA"

    def test_empty_taxonomy_returns_na(self):
        assert bc_outcome.lookup_observed_taxonomy("BOLD123", "") == "NA"

    def test_none_hit_returns_na(self):
        assert bc_outcome.lookup_observed_taxonomy(None, "BOLD123|Apis mellifera") == "NA"

    def test_whitespace_hit_returns_na(self):
        assert bc_outcome.lookup_observed_taxonomy("   ", "BOLD123|Apis mellifera") == "NA"
