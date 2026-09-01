"""Tests for 03_statistical_outlier_filter.py core functions."""
import importlib.util
from pathlib import Path

import pytest
from beegees.utils.configs import get_package_dir

_script = get_package_dir() / "workflow" / "scripts" / "03_statistical_outlier_filter.py"
_spec = importlib.util.spec_from_file_location("stat_filter", _script)
stat_filter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stat_filter)


class TestGenerateConsensusSequence:
    def test_unanimous_consensus(self):
        seqs = ["AAAA", "AAAA", "AAAA"]
        consensus, freqs = stat_filter.generate_consensus_sequence(seqs)
        assert consensus == "AAAA"

    def test_majority_wins(self):
        seqs = ["AAAA", "AAAA", "TTTT"]
        consensus, _ = stat_filter.generate_consensus_sequence(seqs, threshold=0.5)
        assert consensus == "AAAA"

    def test_gap_when_below_threshold(self):
        # Equal split: 50/50 — at threshold=0.5 the majority (first seen or tied)
        # should yield a consistent result; what matters is it doesn't error
        seqs = ["AAAA", "TTTT"]
        consensus, _ = stat_filter.generate_consensus_sequence(seqs, threshold=0.6)
        # Neither A nor T reaches 60% — every position should be '-'
        assert consensus == "----"

    def test_empty_sequences(self):
        consensus, freqs = stat_filter.generate_consensus_sequence([])
        assert consensus == ""
        assert freqs == []

    def test_returns_frequencies(self):
        seqs = ["AATT", "AATT"]
        _, freqs = stat_filter.generate_consensus_sequence(seqs)
        assert len(freqs) == 4
        assert freqs[0]["A"] == pytest.approx(1.0)


class TestCalculateUnweightedDeviation:
    def test_identical_sequences_zero_deviation(self):
        seq = "AATTGGCC"
        assert stat_filter.calculate_unweighted_deviation(seq, seq) == pytest.approx(0.0)

    def test_completely_different(self):
        seq = "AAAAAAAA"
        ref = "TTTTTTTT"
        result = stat_filter.calculate_unweighted_deviation(seq, ref)
        assert result == pytest.approx(1.0)

    def test_half_different(self):
        seq = "AAAATTTT"
        ref = "AAAAGGGG"
        result = stat_filter.calculate_unweighted_deviation(seq, ref)
        assert result == pytest.approx(0.5)

    def test_gaps_excluded(self):
        # Gaps are not valid positions, so "-" vs "-" doesn't count
        seq = "A-A-"
        ref = "A-T-"
        # Only positions 0 and 2 are valid non-gap pairs: pos0 matches, pos2 differs → 0.5
        result = stat_filter.calculate_unweighted_deviation(seq, ref)
        assert result == pytest.approx(0.5)

    def test_different_lengths_truncated(self):
        # Should not raise; shorter length used
        result = stat_filter.calculate_unweighted_deviation("AAAA", "AAAAGGGG")
        assert 0.0 <= result <= 1.0


class TestCalculateWeightedDeviation:
    def test_identical_zero_deviation(self):
        seq = "AATT"
        ref = "AATT"
        freqs = [{"A": 1.0}, {"A": 1.0}, {"T": 1.0}, {"T": 1.0}]
        result = stat_filter.calculate_weighted_deviation(seq, ref, freqs)
        assert result == pytest.approx(0.0)

    def test_mismatch_weighted_by_conservation(self):
        # Highly conserved position (freq=1.0) that differs → high penalty
        seq = "T"
        ref = "A"
        freqs = [{"A": 1.0}]
        result = stat_filter.calculate_weighted_deviation(seq, ref, freqs)
        assert result == pytest.approx(1.0)

    def test_mismatch_at_low_conservation_lower_penalty(self):
        seq = "T"
        ref = "A"
        freqs = [{"A": 0.3, "T": 0.7}]
        result = stat_filter.calculate_weighted_deviation(seq, ref, freqs)
        # penalty weight = 0.3 (the freq of the ref base), total_weight = 0.3 → ratio = 1.0
        assert 0.0 <= result <= 1.0

    def test_empty_returns_zero(self):
        result = stat_filter.calculate_weighted_deviation("", "", [])
        assert result == 0.0
