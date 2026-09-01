"""Tests for pure functions in 04_reference_filter.py."""
import importlib.util
from pathlib import Path
from beegees.utils.configs import get_package_dir

_script = get_package_dir() / "workflow" / "scripts" / "04_reference_filter.py"
_spec = importlib.util.spec_from_file_location("ref_filter", _script)
ref_filter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref_filter)


class TestGetSampleNameForReference:
    def test_r_pattern_splits_on_first_r(self):
        result = ref_filter.get_sample_name_for_reference("BSNHM089-24_r_1.3_s_50_align.fasta")
        assert result == "BSNHM089-24"

    def test_fasta_extension_stripped(self):
        result = ref_filter.get_sample_name_for_reference("SAMPLE01_r_1.5_s_100.fasta")
        assert result == "SAMPLE01"

    def test_fas_extension_stripped(self):
        result = ref_filter.get_sample_name_for_reference("SAMPLE01_r_1.5_s_100.fas")
        assert result == "SAMPLE01"

    def test_fa_extension_stripped(self):
        result = ref_filter.get_sample_name_for_reference("SAMPLE01_r_1.5_s_100.fa")
        assert result == "SAMPLE01"

    def test_fallback_strips_human_filtered_suffix(self):
        result = ref_filter.get_sample_name_for_reference("SAMPLE01_human_filtered.fasta")
        assert result == "SAMPLE01"

    def test_fallback_strips_at_filtered_suffix(self):
        result = ref_filter.get_sample_name_for_reference("SAMPLE01_at_filtered.fasta")
        assert result == "SAMPLE01"

    def test_fallback_strips_outlier_filtered_suffix(self):
        result = ref_filter.get_sample_name_for_reference("SAMPLE01_outlier_filtered.fasta")
        assert result == "SAMPLE01"

    def test_full_path_uses_basename(self):
        result = ref_filter.get_sample_name_for_reference(
            "/data/samples/BSNHM089-24_r_1.3_s_50.fasta"
        )
        assert result == "BSNHM089-24"


class TestGetOutputBasename:
    def test_strips_extension_only(self):
        result = ref_filter.get_output_basename("BSNHM089-24_r_1.3_s_50.fasta")
        assert result == "BSNHM089-24_r_1.3_s_50"

    def test_strips_outlier_filtered_suffix(self):
        result = ref_filter.get_output_basename("BSNHM089-24_r_1.3_s_50_outlier_filtered.fasta")
        assert result == "BSNHM089-24_r_1.3_s_50"

    def test_strips_at_filtered_suffix(self):
        result = ref_filter.get_output_basename("BSNHM089-24_r_1.3_s_50_at_filtered.fasta")
        assert result == "BSNHM089-24_r_1.3_s_50"

    def test_strips_human_filtered_suffix(self):
        result = ref_filter.get_output_basename("BSNHM089-24_r_1.3_s_50_human_filtered.fasta")
        assert result == "BSNHM089-24_r_1.3_s_50"

    def test_preserves_r_s_params(self):
        result = ref_filter.get_output_basename("SAMPLE01_r_1.5_s_100_at_filtered.fas")
        assert "r_1.5_s_100" in result

    def test_full_path_uses_basename(self):
        result = ref_filter.get_output_basename("/data/BSNHM089-24_r_1.3_s_50.fasta")
        assert result == "BSNHM089-24_r_1.3_s_50"
