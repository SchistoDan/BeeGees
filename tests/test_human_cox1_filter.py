"""Tests for 01_human_cox1_filter.py core functions."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from beegees.utils.configs import get_package_dir

_script = get_package_dir() / "workflow" / "scripts" / "01_human_cox1_filter.py"
_spec = importlib.util.spec_from_file_location("cox1_filter", _script)
cox1_filter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cox1_filter)

HUMAN_REF = np.array(list(cox1_filter.HUMAN_COX1.upper()))


class TestVectorisedSequenceSimilarity:
    def test_identical_sequence_returns_one(self):
        seq = cox1_filter.HUMAN_COX1[:100]
        result = cox1_filter.vectorised_sequence_similarity(seq, HUMAN_REF)
        assert result == pytest.approx(1.0)

    def test_completely_different_returns_low(self):
        # Complement-ish: replace A→C, T→G
        seq = "CCGGCCGG" * 20
        result = cox1_filter.vectorised_sequence_similarity(seq, HUMAN_REF)
        assert result < 0.5

    def test_empty_sequence_returns_zero(self):
        assert cox1_filter.vectorised_sequence_similarity("", HUMAN_REF) == 0.0

    def test_short_sequence_below_min_overlap_returns_zero(self):
        # min_overlap default is 20; supply fewer than 20 bases
        result = cox1_filter.vectorised_sequence_similarity("ATCG" * 4, HUMAN_REF, min_overlap=20)
        assert result == 0.0

    def test_partial_match(self):
        # First half matches, second half does not
        seq = cox1_filter.HUMAN_COX1[:50] + "C" * 50
        result = cox1_filter.vectorised_sequence_similarity(seq, HUMAN_REF)
        assert 0.0 < result < 1.0

    def test_case_insensitive(self):
        seq_upper = cox1_filter.HUMAN_COX1[:100].upper()
        seq_lower = cox1_filter.HUMAN_COX1[:100].lower()
        r_upper = cox1_filter.vectorised_sequence_similarity(seq_upper, HUMAN_REF)
        r_lower = cox1_filter.vectorised_sequence_similarity(seq_lower, HUMAN_REF)
        assert r_upper == pytest.approx(r_lower)

    def test_similarity_between_zero_and_one(self):
        seq = "ATCGATCG" * 30
        result = cox1_filter.vectorised_sequence_similarity(seq, HUMAN_REF)
        assert 0.0 <= result <= 1.0
