"""Tests for pure functions in tv_blast2taxonomy.py."""
import importlib.util
import logging
from pathlib import Path

import pytest
from beegees.utils.configs import get_package_dir

_script = get_package_dir() / "workflow" / "scripts" / "tv_blast2taxonomy.py"
_spec = importlib.util.spec_from_file_location("tv_blast2tax", _script)
tv_blast2tax = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv_blast2tax)

_logger = logging.getLogger("test_tv_blast2taxonomy")


class TestBuildTaxonomyLineage:
    def test_full_lineage(self):
        tax = {
            "phylum": "Arthropoda",
            "class": "Insecta",
            "order": "Hymenoptera",
            "family": "Apidae",
            "genus": "Apis",
            "species": "Apis mellifera",
        }
        result = tv_blast2tax.build_taxonomy_lineage(tax)
        assert result == "Arthropoda;Insecta;Hymenoptera;Apidae;Apis;Apis mellifera"

    def test_missing_ranks_become_empty(self):
        tax = {"genus": "Apis", "species": "Apis mellifera"}
        result = tv_blast2tax.build_taxonomy_lineage(tax)
        parts = result.split(";")
        assert len(parts) == 6
        assert parts[4] == "Apis"
        assert parts[0] == ""

    def test_empty_dict(self):
        result = tv_blast2tax.build_taxonomy_lineage({})
        assert result == ";;;;;"


class TestExtractHitId:
    def test_boldistilled_pipe_format(self):
        assert tv_blast2tax.extract_hit_id("something|BOLD:AAA0001") == "BOLD:AAA0001"

    def test_multiple_pipes_returns_last(self):
        assert tv_blast2tax.extract_hit_id("a|b|c|BOLD:XYZ") == "BOLD:XYZ"

    def test_plain_accession(self):
        assert tv_blast2tax.extract_hit_id("AB000317.1") == "AB000317.1"

    def test_no_pipe_passthrough(self):
        assert tv_blast2tax.extract_hit_id("MYID123") == "MYID123"


class TestFindProcessId:
    def _tax(self):
        return {"BOLD123": {}, "BOLD456": {}}

    def test_exact_match(self):
        assert tv_blast2tax.find_process_id("BOLD123", self._tax()) == "BOLD123"

    def test_partial_match(self):
        # seq_id contains the process_id as a substring
        assert tv_blast2tax.find_process_id("BOLD123_r_1.3_s_50", self._tax()) == "BOLD123"

    def test_no_match_returns_none(self):
        assert tv_blast2tax.find_process_id("BOLD999", self._tax()) is None

    def test_empty_taxonomy(self):
        assert tv_blast2tax.find_process_id("BOLD123", {}) is None


class TestGetExpectedLineage:
    def test_extracts_four_ranks(self):
        tax = {
            "phylum": "Arthropoda",
            "order": "Hymenoptera",
            "family": "Apidae",
            "genus": "Apis",
            "species": "Apis mellifera",
        }
        lineage = tv_blast2tax.get_expected_lineage(tax)
        assert set(lineage.keys()) == {"order", "family", "genus", "species"}
        assert "phylum" not in lineage

    def test_missing_ranks_return_empty_string(self):
        lineage = tv_blast2tax.get_expected_lineage({"genus": "Apis"})
        assert lineage["order"] == ""
        assert lineage["genus"] == "Apis"

    def test_whitespace_stripped(self):
        lineage = tv_blast2tax.get_expected_lineage({"species": "  Apis mellifera  "})
        assert lineage["species"] == "Apis mellifera"


class TestFindExpectedTaxonomy:
    def _tax(self):
        return {"order": "Hymenoptera", "family": "Apidae", "genus": "Apis", "species": "Apis mellifera"}

    def test_exact_rank_found(self):
        value, rank = tv_blast2tax.find_expected_taxonomy(self._tax(), "species", _logger)
        assert value == "Apis mellifera"
        assert rank == "species"

    def test_fallback_to_higher_rank(self):
        tax = {"order": "Hymenoptera"}  # no species/genus/family
        value, rank = tv_blast2tax.find_expected_taxonomy(tax, "species", _logger)
        assert value == "Hymenoptera"
        assert rank == "order"

    def test_no_taxonomy_returns_empty(self):
        value, rank = tv_blast2tax.find_expected_taxonomy({}, "species", _logger)
        assert value == ""
        assert rank == ""

    def test_invalid_rank_returns_empty(self):
        value, rank = tv_blast2tax.find_expected_taxonomy(self._tax(), "kingdom", _logger)
        assert value == ""


class TestCheckHitMatchesLineage:
    def _lineage(self):
        return {"order": "Hymenoptera", "family": "Apidae", "genus": "Apis", "species": "Apis mellifera"}

    def test_species_match(self):
        hit = {"species": "Apis mellifera", "genus": "Apis", "family": "Apidae", "order": "Hymenoptera"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, self._lineage())
        assert tax == "Apis mellifera"
        assert rank == "species"

    def test_genus_match_when_species_differs(self):
        hit = {"species": "Apis cerana", "genus": "Apis", "family": "Apidae", "order": "Hymenoptera"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, self._lineage())
        assert rank == "genus"

    def test_no_match_returns_none(self):
        hit = {"species": "Bombus terrestris", "genus": "Bombus", "family": "Apidae", "order": "Hymenoptera"}
        # family and order match but genus and species don't — function checks most-specific first
        # family IS in lineage, so it will match at family level
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, self._lineage())
        # family "Apidae" matches
        assert rank == "family"

    def test_completely_different_returns_none(self):
        hit = {"species": "Drosophila melanogaster", "genus": "Drosophila",
               "family": "Drosophilidae", "order": "Diptera"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, self._lineage())
        assert tax is None
        assert rank is None

    def test_empty_hit_returns_none(self):
        tax, rank = tv_blast2tax.check_hit_matches_lineage({}, self._lineage())
        assert tax is None


class TestSortHitsByQuality:
    def _hit(self, pident, length, mismatch=0, evalue=1e-50):
        return {"pident": pident, "length": length, "mismatch": mismatch, "evalue": evalue}

    def test_higher_pident_first(self):
        hits = [self._hit(95.0, 500), self._hit(99.0, 500)]
        result = tv_blast2tax.sort_hits_by_quality(hits)
        assert result[0]["pident"] == 99.0

    def test_equal_pident_longer_length_first(self):
        hits = [self._hit(99.0, 400), self._hit(99.0, 600)]
        result = tv_blast2tax.sort_hits_by_quality(hits)
        assert result[0]["length"] == 600

    def test_empty_list(self):
        assert tv_blast2tax.sort_hits_by_quality([]) == []

    def test_single_hit_unchanged(self):
        hits = [self._hit(98.0, 500)]
        result = tv_blast2tax.sort_hits_by_quality(hits)
        assert result[0]["pident"] == 98.0


class TestExtractSeqIdValues:
    def test_standard_values(self):
        r, s, fcleaner = tv_blast2tax.extract_seq_id_values("SAMPLE_r_1.3_s_50_merge")
        assert r == pytest.approx(1.3)
        assert s == 50
        assert fcleaner is False

    def test_fcleaner_detected(self):
        _, _, fcleaner = tv_blast2tax.extract_seq_id_values("SAMPLE_r_1.3_s_50_fcleaner_merge")
        assert fcleaner is True

    def test_integer_r_value(self):
        r, _, _ = tv_blast2tax.extract_seq_id_values("SAMPLE_r_1_s_100")
        assert r == pytest.approx(1.0)

    def test_no_r_s_returns_none(self):
        r, s, _ = tv_blast2tax.extract_seq_id_values("SAMPLE_merge")
        assert r is None
        assert s is None


class TestSelectBestSequences:
    def _result(self, process_id, seq_id, match="YES", pident=99.0, length=658, gaps=0, mismatch=0, rank="species"):
        return {
            "Process_ID": process_id,
            "seq_id": seq_id,
            "match_taxonomy": match,
            "pident": pident,
            "length": length,
            "gaps": gaps,
            "mismatch": mismatch,
            "matched_rank": rank,
            "evalue": 1e-100,
        }

    def test_best_sequence_marked_yes(self):
        results = [
            self._result("P1", "P1_r_1.3_s_50_merge", pident=99.0),
            self._result("P1", "P1_r_1.5_s_50_merge", pident=97.0),
        ]
        out = tv_blast2tax.select_best_sequences(results, _logger)
        yes = [r for r in out if r["selected"] == "YES"]
        assert len(yes) == 1
        assert yes[0]["pident"] == 99.0

    def test_non_matching_all_marked_no(self):
        results = [
            self._result("P1", "P1_r_1.3_s_50_merge", match="NO"),
            self._result("P1", "P1_r_1.5_s_50_merge", match="NO"),
        ]
        out = tv_blast2tax.select_best_sequences(results, _logger)
        assert all(r["selected"] == "NO" for r in out)

    def test_multiple_process_ids_independent(self):
        results = [
            self._result("P1", "P1_r_1.3_s_50_merge"),
            self._result("P2", "P2_r_1.3_s_50_merge"),
        ]
        out = tv_blast2tax.select_best_sequences(results, _logger)
        yes_ids = {r["Process_ID"] for r in out if r["selected"] == "YES"}
        assert yes_ids == {"P1", "P2"}
