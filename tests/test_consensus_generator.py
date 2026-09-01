"""Tests for 05_consensus_generator.py core functions."""
import importlib.util
from pathlib import Path

import pytest
from beegees.utils.configs import get_package_dir

_script = get_package_dir() / "workflow" / "scripts" / "05_consensus_generator.py"
_spec = importlib.util.spec_from_file_location("cons_gen", _script)
cons_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cons_gen)


class TestCalculateCoverageStatistics:
    def test_empty_sequences(self):
        result = cons_gen.calculate_coverage_statistics([])
        assert result["cleaning_cov_percent"] == 0.0
        assert result["cleaning_cov_avg"] == 0.0

    def test_single_full_sequence(self):
        result = cons_gen.calculate_coverage_statistics(["ATCG"])
        assert result["cleaning_cov_percent"] == pytest.approx(100.0)
        assert result["cleaning_cov_max"] == 1
        assert result["cleaning_cov_min"] == 1

    def test_coverage_depth_with_gaps(self):
        # Two sequences, second has a gap at position 2
        seqs = ["ATCG", "AT-G"]
        result = cons_gen.calculate_coverage_statistics(seqs)
        # Position 2: only one sequence covers it → min coverage = 1
        assert result["cleaning_cov_min"] == 1
        assert result["cleaning_cov_max"] == 2

    def test_multiple_sequences_full_coverage(self):
        seqs = ["AAAA", "TTTT", "GGGG"]
        result = cons_gen.calculate_coverage_statistics(seqs)
        assert result["cleaning_cov_percent"] == pytest.approx(100.0)
        assert result["cleaning_cov_avg"] == pytest.approx(3.0)


class TestGenerateConsensusSequence:
    def test_unanimous(self):
        seqs = ["AAAA", "AAAA"]
        consensus, _ = cons_gen.generate_consensus_sequence(seqs)
        assert consensus == "AAAA"

    def test_majority_threshold(self):
        seqs = ["AAAA", "AAAA", "TTTT"]
        consensus, _ = cons_gen.generate_consensus_sequence(seqs, threshold=0.5)
        assert consensus == "AAAA"

    def test_below_threshold_gives_gap(self):
        seqs = ["AAAA", "TTTT"]
        consensus, _ = cons_gen.generate_consensus_sequence(seqs, threshold=0.6)
        assert consensus == "----"

    def test_empty_returns_empty(self):
        consensus, freqs = cons_gen.generate_consensus_sequence([])
        assert consensus == ""
        assert freqs == []


class TestCalculateATContent:
    def test_pure_at(self):
        assert cons_gen.calculate_at_content("AAATTT") == pytest.approx(1.0)

    def test_pure_gc(self):
        assert cons_gen.calculate_at_content("GCGCGC") == pytest.approx(0.0)

    def test_mixed(self):
        assert cons_gen.calculate_at_content("AATTGC") == pytest.approx(4 / 6)

    def test_empty(self):
        assert cons_gen.calculate_at_content("") == 0.0


class TestCountAmbiguousBases:
    def test_no_ambiguous(self):
        assert cons_gen.count_ambiguous_bases("ATCGATCG") == 0

    def test_n_counted(self):
        assert cons_gen.count_ambiguous_bases("ATCGNNN") == 3

    def test_iupac_ambiguous_counted(self):
        # R, Y, S, W, K, M, B, D, H, V are all IUPAC ambiguous
        assert cons_gen.count_ambiguous_bases("RYWWKM") == 6

    def test_gaps_not_counted_as_ambiguous(self):
        assert cons_gen.count_ambiguous_bases("A-T-G") == 0
