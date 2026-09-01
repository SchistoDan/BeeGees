"""Tests for categorise_header in barcode_consensus_count.py."""
import importlib.util
from pathlib import Path
from beegees.utils.configs import get_package_dir

_script = get_package_dir() / "workflow" / "scripts" / "barcode_consensus_count.py"
_spec = importlib.util.spec_from_file_location("bcc", _script)
bcc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcc)


class TestCategoriseHeader:
    def test_fcleaner_concat(self):
        assert bcc.categorise_header(">BSNHM089-24_r_1.3_s_50_fcleaner_concat") == "fcleaner_concat"

    def test_fcleaner_merge(self):
        assert bcc.categorise_header(">BSNHM089-24_r_1.3_s_50_fcleaner_merge") == "fcleaner_merge"

    def test_fcleaner_se(self):
        assert bcc.categorise_header(">BSNHM089-24_r_1.3_s_50_fcleaner_se") == "fcleaner_se"

    def test_concat(self):
        assert bcc.categorise_header(">BSNHM089-24_r_1.3_s_50_concat") == "concat"

    def test_merge(self):
        assert bcc.categorise_header(">BSNHM089-24_r_1.3_s_50_merge") == "merge"

    def test_se(self):
        assert bcc.categorise_header(">BSNHM089-24_r_1.3_s_50_se") == "se"

    def test_unmarked_legacy_header_is_concat(self):
        # Headers without an explicit suffix fall back to "concat"
        assert bcc.categorise_header(">BSNHM089-24_r_1.3_s_50") == "concat"

    def test_fcleaner_wins_over_concat(self):
        # Longer _fcleaner_concat tag takes priority over bare _concat
        header = ">SAMPLE_concat_fcleaner_concat"
        assert bcc.categorise_header(header) == "fcleaner_concat"

    def test_sample_id_with_merge_substring_not_misclassified(self):
        # "merge" appearing inside the sample ID should not cause misclassification
        # if the header ends with _concat
        assert bcc.categorise_header(">mergetest_r_1_s_50_concat") == "concat"

    def test_trailing_whitespace_ignored(self):
        assert bcc.categorise_header(">SAMPLE_r_1_s_50_se   ") == "se"
