"""Tests for 02_at_content_filter.py core functions."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from beegees.utils.configs import get_package_dir

# Import the script as a module
_script = get_package_dir() / "workflow" / "scripts" / "02_at_content_filter.py"
_spec = importlib.util.spec_from_file_location("at_filter", _script)
at_filter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(at_filter)


class TestCalculateATContent:
    def test_pure_at(self):
        assert at_filter.calculate_at_content("AAATTT") == pytest.approx(1.0)

    def test_pure_gc(self):
        assert at_filter.calculate_at_content("GCGCGC") == pytest.approx(0.0)

    def test_mixed(self):
        assert at_filter.calculate_at_content("AATTGC") == pytest.approx(4 / 6)

    def test_gaps_ignored(self):
        # "A-T-G-C" has 1 A, 1 T, 1 G, 1 C → 2/4 = 0.5
        assert at_filter.calculate_at_content("A-T-G-C") == pytest.approx(0.5)

    def test_empty_sequence(self):
        assert at_filter.calculate_at_content("") == 0.0

    def test_all_gaps(self):
        assert at_filter.calculate_at_content("----") == 0.0

    def test_case_insensitive(self):
        assert at_filter.calculate_at_content("aAtT") == pytest.approx(1.0)


class TestVectorisedATContent:
    def test_basic(self):
        arr = np.array([list("AATT"), list("GCGC")])
        result = at_filter.vectorised_at_content(arr)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)

    def test_with_gaps(self):
        arr = np.array([list("A-T-")])
        result = at_filter.vectorised_at_content(arr)
        assert result[0] == pytest.approx(1.0)

    def test_empty_array(self):
        arr = np.array([]).reshape(0, 0)
        result = at_filter.vectorised_at_content(arr)
        assert len(result) == 0


class TestFilterATContent:
    """Test the filter_at_content function with synthetic SeqRecord data.

    Signature: filter_at_content(records, consensus_seq, threshold, mode)
    Returns: (kept, removed, stats)
    """

    def _make_record(self, seq_str, record_id="seq"):
        from Bio.SeqRecord import SeqRecord
        from Bio.Seq import Seq
        return SeqRecord(Seq(seq_str), id=record_id, description="")

    def _make_records(self, seqs):
        return [self._make_record(s, f"seq{i}") for i, s in enumerate(seqs)]

    def test_all_pass_within_threshold(self):
        # All sequences identical AT content to consensus — all should pass
        seqs = ["AATTGGCC", "AATTGGCC", "AATTGGCC"]
        records = self._make_records(seqs)
        consensus = "AATTGGCC"
        kept, removed, _ = at_filter.filter_at_content(records, consensus, threshold=0.1, mode="absolute")
        assert len(kept) == 3
        assert len(removed) == 0

    def test_outlier_removed_absolute(self):
        # First 3 sequences: ~50% AT. Last: 100% AT — should be removed
        seqs = ["AATTGGCC"] * 3 + ["AAAAAAAA"]
        records = self._make_records(seqs)
        consensus = "AATTGGCC"
        kept, removed, _ = at_filter.filter_at_content(records, consensus, threshold=0.1, mode="absolute")
        removed_ids = [r.id for r in removed]
        assert "seq3" in removed_ids

    def test_higher_mode_only_removes_high_at(self):
        # Consensus ~50% AT; seq0 has 0% AT — should NOT be removed in "higher" mode
        seqs = ["GGGGGGGG", "AATTGGCC", "AATTGGCC", "AATTGGCC"]
        records = self._make_records(seqs)
        consensus = "AATTGGCC"
        kept, removed, _ = at_filter.filter_at_content(records, consensus, threshold=0.1, mode="higher")
        kept_ids = [r.id for r in kept]
        assert "seq0" in kept_ids

    def test_lower_mode_only_removes_low_at(self):
        # Consensus ~50% AT; 100% AT sequence should NOT be removed in "lower" mode
        seqs = ["AAAAAAAA", "AATTGGCC", "AATTGGCC", "AATTGGCC"]
        records = self._make_records(seqs)
        consensus = "AATTGGCC"
        kept, removed, _ = at_filter.filter_at_content(records, consensus, threshold=0.1, mode="lower")
        kept_ids = [r.id for r in kept]
        assert "seq0" in kept_ids

    def test_empty_records_returns_empty(self):
        kept, removed, stats = at_filter.filter_at_content([], "", threshold=0.1, mode="absolute")
        assert kept == []
        assert removed == []
