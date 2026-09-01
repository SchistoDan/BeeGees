"""Tests for clean_fasta_header in val_csv_merger.py."""
import importlib.util
import math
from pathlib import Path

import pytest
from beegees.utils.configs import get_package_dir

pytest.importorskip("pandas", reason="pandas not installed; skipping val_csv_merger tests")

_script = get_package_dir() / "workflow" / "scripts" / "val_csv_merger.py"
_spec = importlib.util.spec_from_file_location("val_csv_merger", _script)
val_csv_merger = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(val_csv_merger)


class TestCleanFastaHeader:
    def test_strips_gt_prefix(self):
        assert val_csv_merger.clean_fasta_header(">SAMPLE123") == "SAMPLE123"

    def test_no_gt_unchanged(self):
        assert val_csv_merger.clean_fasta_header("SAMPLE123") == "SAMPLE123"

    def test_only_gt_returns_empty(self):
        assert val_csv_merger.clean_fasta_header(">") == ""

    def test_nan_returned_as_is(self):
        import pandas as pd
        result = val_csv_merger.clean_fasta_header(pd.NA)
        assert result is pd.NA or (isinstance(result, float) and math.isnan(result))

    def test_multiple_gt_only_leading_stripped(self):
        # lstrip strips ALL leading '>' characters
        result = val_csv_merger.clean_fasta_header(">>SAMPLE")
        assert result == "SAMPLE"

    def test_full_header_with_description(self):
        result = val_csv_merger.clean_fasta_header(">SAMPLE123 Apis mellifera COI")
        assert result == "SAMPLE123 Apis mellifera COI"
