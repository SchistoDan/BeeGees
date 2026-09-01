"""Tests for pure parsing functions in compile_barcoding_stats.py."""
import importlib.util
from pathlib import Path
from beegees.utils.configs import get_package_dir

_script = get_package_dir() / "workflow" / "scripts" / "compile_barcoding_stats.py"
_spec = importlib.util.spec_from_file_location("compile_stats", _script)
compile_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compile_stats)


class TestExtractSampleInfo:
    def test_standard_filename(self):
        base_id, full_id, params = compile_stats.extract_sample_info(
            "BGSNL096-23_r_1.3_s_50_align_BGSNL096-23.fas"
        )
        assert base_id == "BGSNL096-23"
        assert params == "r_1.3_s_50"
        assert full_id == "BGSNL096-23_r_1.3_s_50"

    def test_decimal_r_value(self):
        base_id, full_id, params = compile_stats.extract_sample_info(
            "SAMPLE01_r_1.5_s_100.fasta"
        )
        assert params == "r_1.5_s_100"
        assert base_id == "SAMPLE01"

    def test_integer_r_value(self):
        _, _, params = compile_stats.extract_sample_info("SAMPLE01_r_1_s_50.fasta")
        assert params == "r_1_s_50"

    def test_no_params_returns_base_id_only(self):
        base_id, full_id, params = compile_stats.extract_sample_info("SAMPLE01.fasta")
        assert base_id == "SAMPLE01"
        assert params == ""
        assert full_id == "SAMPLE01"

    def test_extension_stripped(self):
        base_id, _, _ = compile_stats.extract_sample_info("BGSNL096-23_r_1.3_s_50.fas")
        assert base_id == "BGSNL096-23"


class TestExtractCleaningInfo:
    def test_fcleaner_merge_suffix(self):
        base_id, mge_params = compile_stats.extract_cleaning_info(
            "BSNHM002-24_r_1.3_s_50_BSNHM002-24_fcleaner_merge"
        )
        assert base_id == "BSNHM002-24"
        assert "r_1.3_s_50" in mge_params
        assert "fcleaner_merge" in mge_params

    def test_fcleaner_concat_suffix(self):
        base_id, mge_params = compile_stats.extract_cleaning_info(
            "BSNHM002-24_r_1.3_s_50_BSNHM002-24_fcleaner_concat"
        )
        assert "fcleaner_concat" in mge_params

    def test_base_id_extracted(self):
        base_id, _ = compile_stats.extract_cleaning_info(
            "BSNHM002-24_r_1.3_s_50_BSNHM002-24_fcleaner_merge"
        )
        assert base_id == "BSNHM002-24"

    def test_no_params_returns_empty_mge_params(self):
        base_id, mge_params = compile_stats.extract_cleaning_info("SAMPLE01")
        assert base_id == "SAMPLE01"
        assert mge_params == ""


class TestExtractIdAndParamsFromHeader:
    def test_merge_mode(self):
        base_id, mge_params = compile_stats.extract_id_and_params_from_header(
            ">BSNHM004-24_r_1.5_s_50_BSNHM004-24_merge"
        )
        assert base_id == "BSNHM004-24"
        assert "r_1.5_s_50" in mge_params
        assert "merge" in mge_params

    def test_concat_mode(self):
        base_id, mge_params = compile_stats.extract_id_and_params_from_header(
            ">BSNHM004-24_r_1.5_s_50_BSNHM004-24"
        )
        assert base_id == "BSNHM004-24"
        # duplicate base_id should be removed from params
        assert "BSNHM004-24" not in mge_params

    def test_no_gt_prefix(self):
        base_id, mge_params = compile_stats.extract_id_and_params_from_header(
            "BSNHM004-24_r_1.5_s_50_BSNHM004-24_merge"
        )
        assert base_id == "BSNHM004-24"

    def test_fcleaner_merge(self):
        base_id, mge_params = compile_stats.extract_id_and_params_from_header(
            ">BSNHM004-24_r_1.5_s_50_BSNHM004-24_fcleaner_merge"
        )
        assert "fcleaner_merge" in mge_params

    def test_empty_header_returns_none(self):
        base_id, mge_params = compile_stats.extract_id_and_params_from_header("")
        assert base_id is None
        assert mge_params is None
