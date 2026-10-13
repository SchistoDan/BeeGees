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


class TestAllowedRanksFor:
    def test_family_floor_excludes_order(self):
        assert tv_blast2tax.allowed_ranks_for("family") == ["species", "genus", "family"]

    def test_order_floor_allows_everything(self):
        assert tv_blast2tax.allowed_ranks_for("order") == ["species", "genus", "family", "order"]

    def test_species_floor_allows_species_only(self):
        assert tv_blast2tax.allowed_ranks_for("species") == ["species"]

    def test_unrecognised_floor_allows_nothing(self):
        assert tv_blast2tax.allowed_ranks_for("kingdom") == []

    def test_empty_floor_allows_nothing(self):
        assert tv_blast2tax.allowed_ranks_for("") == []


class TestNormaliseTaxon:
    def test_case_folded(self):
        assert tv_blast2tax.normalise_taxon("Sciuridae") == tv_blast2tax.normalise_taxon("sciuridae")

    def test_upper_case_folded(self):
        assert tv_blast2tax.normalise_taxon("SCIURIDAE") == tv_blast2tax.normalise_taxon("Sciuridae")

    def test_internal_whitespace_collapsed(self):
        assert (tv_blast2tax.normalise_taxon("Sciurus  vulgaris")
                == tv_blast2tax.normalise_taxon("Sciurus vulgaris"))

    def test_surrounding_whitespace_stripped(self):
        assert tv_blast2tax.normalise_taxon("  Sciuridae\t") == tv_blast2tax.normalise_taxon("Sciuridae")

    def test_distinct_names_stay_distinct(self):
        assert tv_blast2tax.normalise_taxon("Sciuridae") != tv_blast2tax.normalise_taxon("Cricetidae")

    def test_empty_string(self):
        assert tv_blast2tax.normalise_taxon("") == ""


class TestCheckHitMatchesLineage:
    def _lineage(self):
        return {"order": "Hymenoptera", "family": "Apidae", "genus": "Apis", "species": "Apis mellifera"}

    def _family_floor(self):
        return tv_blast2tax.allowed_ranks_for("family")

    def test_species_match(self):
        hit = {"species": "Apis mellifera", "genus": "Apis", "family": "Apidae", "order": "Hymenoptera"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, self._lineage(), self._family_floor())
        assert tax == "Apis mellifera"
        assert rank == "species"

    def test_genus_match_when_species_differs(self):
        hit = {"species": "Apis cerana", "genus": "Apis", "family": "Apidae", "order": "Hymenoptera"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, self._lineage(), self._family_floor())
        assert rank == "genus"

    def test_no_match_returns_none(self):
        hit = {"species": "Bombus terrestris", "genus": "Bombus", "family": "Apidae", "order": "Hymenoptera"}
        # family and order match but genus and species don't — function checks most-specific first
        # family IS in lineage, so it will match at family level
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, self._lineage(), self._family_floor())
        # family "Apidae" matches
        assert rank == "family"

    def test_completely_different_returns_none(self):
        hit = {"species": "Drosophila melanogaster", "genus": "Drosophila",
               "family": "Drosophilidae", "order": "Diptera"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, self._lineage(), self._family_floor())
        assert tax is None
        assert rank is None

    def test_empty_hit_returns_none(self):
        tax, rank = tv_blast2tax.check_hit_matches_lineage({}, self._lineage(), self._family_floor())
        assert tax is None

    def test_order_only_hit_rejected_under_family_floor(self):
        # Shares the expected order but a different family: the UK016-H10 case
        hit = {"species": "Vespa crabro", "genus": "Vespa", "family": "Vespidae", "order": "Hymenoptera"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, self._lineage(), self._family_floor())
        assert tax is None
        assert rank is None

    def test_order_only_hit_accepted_under_order_floor(self):
        # Same hit passes when the sample's expected taxonomy only reaches order
        hit = {"species": "Vespa crabro", "genus": "Vespa", "family": "Vespidae", "order": "Hymenoptera"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(
            hit, self._lineage(), tv_blast2tax.allowed_ranks_for("order"))
        assert tax == "Hymenoptera"
        assert rank == "order"

    def test_family_hit_rejected_under_genus_floor(self):
        hit = {"species": "Bombus terrestris", "genus": "Bombus", "family": "Apidae", "order": "Hymenoptera"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(
            hit, self._lineage(), tv_blast2tax.allowed_ranks_for("genus"))
        assert tax is None

    def test_no_permitted_ranks_never_matches(self):
        hit = {"species": "Apis mellifera", "genus": "Apis", "family": "Apidae", "order": "Hymenoptera"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, self._lineage(), [])
        assert tax is None
        assert rank is None

    def test_lowercase_expected_matches_capitalised_hit(self):
        # The AH29 case: an expected-taxonomy CSV written in lowercase against BOLD's own casing
        hit = {"species": "Sciurus vulgaris", "genus": "Sciurus",
               "family": "Sciuridae", "order": "Rodentia"}
        lineage = {"species": "sciurus vulgaris", "genus": "sciurus",
                   "family": "sciuridae", "order": "rodentia"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, lineage, self._family_floor())
        assert rank == "species"
        # The database's casing is reported, not the expected file's
        assert tax == "Sciurus vulgaris"

    def test_uppercase_expected_matches_capitalised_hit(self):
        hit = {"species": "Apis mellifera", "genus": "Apis", "family": "Apidae", "order": "Hymenoptera"}
        lineage = {"species": "APIS MELLIFERA", "genus": "APIS",
                   "family": "APIDAE", "order": "HYMENOPTERA"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, lineage, self._family_floor())
        assert tax == "Apis mellifera"
        assert rank == "species"

    def test_internal_whitespace_difference_still_matches(self):
        hit = {"species": "Sciurus  vulgaris", "genus": "Sciurus", "family": "Sciuridae"}
        lineage = {"species": "Sciurus vulgaris", "genus": "Sciurus", "family": "Sciuridae"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, lineage, self._family_floor())
        assert rank == "species"

    def test_case_folding_does_not_match_different_taxa(self):
        # Folding case must not turn an unrelated hit into a match at any rank
        hit = {"species": "Drosophila melanogaster", "genus": "Drosophila",
               "family": "Drosophilidae", "order": "Diptera"}
        lineage = {"species": "apis mellifera", "genus": "apis",
                   "family": "apidae", "order": "hymenoptera"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(
            hit, lineage, tv_blast2tax.allowed_ranks_for("order"))
        assert tax is None
        assert rank is None

    def test_case_folding_does_not_bypass_the_floor(self):
        # Mis-cased input still obeys the validation floor: order-only match stays rejected
        hit = {"species": "Vespa crabro", "genus": "Vespa",
               "family": "Vespidae", "order": "Hymenoptera"}
        lineage = {"species": "apis mellifera", "genus": "apis",
                   "family": "apidae", "order": "hymenoptera"}
        tax, rank = tv_blast2tax.check_hit_matches_lineage(hit, lineage, self._family_floor())
        assert tax is None
        # ...and is accepted once the floor is lowered to order
        tax, rank = tv_blast2tax.check_hit_matches_lineage(
            hit, lineage, tv_blast2tax.allowed_ranks_for("order"))
        assert tax == "Hymenoptera"
        assert rank == "order"


class TestFindFirstMatchingHit:
    """Reproduces the UK016-H10 case: expected Megaselia (Phoridae, Diptera), where the only hit
    sharing anything with the expected lineage does so at order rank, and is the worst-scoring
    survivor of the quality filters."""

    def _hit(self, hit_id, pident, length, hit_num, mismatch=0, gaps=0, evalue=1e-90):
        return {"hit_id": hit_id, "pident": pident, "length": length, "mismatch": mismatch,
                "gaps": gaps, "evalue": evalue, "description": "", "hit_num": hit_num}

    def _hits(self):
        # Percent-identity descending, as tv_local_blast.py writes them
        return [
            self._hit("CAB095-06|BOLD:AAA0001", 100.0, 87, 1),
            self._hit("BGEPL1486-24|BOLD:AHE9669", 100.0, 234, 3),
            self._hit("BGLIB1267-24|BOLD:AGX0155", 90.698, 258, 10, mismatch=24),
        ]

    def _taxonomy_data(self):
        return {
            "BOLD:AAA0001": {"species": "Homo sapiens", "genus": "Homo",
                             "family": "Hominidae", "order": "Primates"},
            "BOLD:AHE9669": {"species": "Lithobius burzenlandicus", "genus": "Lithobius",
                             "family": "Lithobiidae", "order": "Lithobiomorpha"},
            "BOLD:AGX0155": {"species": "Beris geniculata", "genus": "Beris",
                             "family": "Stratiomyidae", "order": "Diptera"},
        }

    def _lineage(self):
        return {"order": "Diptera", "family": "Phoridae",
                "genus": "Megaselia", "species": "Megaselia pumila"}

    def test_order_only_match_rejected_under_family_floor(self):
        result = tv_blast2tax.find_first_matching_hit(
            self._hits(), self._taxonomy_data(), self._lineage(),
            tv_blast2tax.allowed_ranks_for("family"), 80.0, 100, _logger)
        assert result is None

    def test_order_only_match_accepted_under_order_floor(self):
        result = tv_blast2tax.find_first_matching_hit(
            self._hits(), self._taxonomy_data(), self._lineage(),
            tv_blast2tax.allowed_ranks_for("order"), 80.0, 100, _logger)
        assert result["hit_id"] == "BGLIB1267-24|BOLD:AGX0155"
        assert result["matched_rank"] == "order"
        assert result["taxonomy_match"] == "Diptera"

    def test_family_match_still_accepted_under_family_floor(self):
        hits = self._hits() + [self._hit("BGEXX0001-24|BOLD:AAA9999", 88.0, 300, 11)]
        taxonomy_data = dict(self._taxonomy_data())
        taxonomy_data["BOLD:AAA9999"] = {"species": "Megaselia rufipes", "genus": "Megaselia",
                                         "family": "Phoridae", "order": "Diptera"}
        result = tv_blast2tax.find_first_matching_hit(
            hits, taxonomy_data, self._lineage(),
            tv_blast2tax.allowed_ranks_for("family"), 80.0, 100, _logger)
        assert result["hit_id"] == "BGEXX0001-24|BOLD:AAA9999"
        assert result["matched_rank"] == "genus"

    def test_no_permitted_ranks_returns_none(self):
        # A sample with no expected taxonomy at any rank cannot be validated
        result = tv_blast2tax.find_first_matching_hit(
            self._hits(), self._taxonomy_data(), self._lineage(), [], 80.0, 100, _logger)
        assert result is None

    def test_quality_filters_still_applied(self):
        # The 87bp human hit is dropped on length before taxonomy is consulted
        result = tv_blast2tax.find_first_matching_hit(
            self._hits(), self._taxonomy_data(),
            {"order": "Primates", "family": "Hominidae", "genus": "Homo", "species": "Homo sapiens"},
            tv_blast2tax.allowed_ranks_for("family"), 80.0, 100, _logger)
        assert result is None

    def test_lowercase_expected_taxonomy_still_matches(self):
        # The AH29 case end to end: a good BOLD hit that previously failed on casing alone
        hits = self._hits() + [self._hit("CICA005-22|BOLD:AAE9374", 99.695, 656, 11, mismatch=2)]
        taxonomy_data = dict(self._taxonomy_data())
        taxonomy_data["BOLD:AAE9374"] = {"species": "Sciurus vulgaris", "genus": "Sciurus",
                                         "family": "Sciuridae", "order": "Rodentia"}
        lineage = {"species": "sciurus vulgaris", "genus": "sciurus",
                   "family": "sciuridae", "order": "rodentia"}
        result = tv_blast2tax.find_first_matching_hit(
            hits, taxonomy_data, lineage,
            tv_blast2tax.allowed_ranks_for("family"), 80.0, 100, _logger)
        assert result is not None
        assert result["hit_id"] == "CICA005-22|BOLD:AAE9374"
        assert result["matched_rank"] == "species"
        assert result["taxonomy_match"] == "Sciurus vulgaris"


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
