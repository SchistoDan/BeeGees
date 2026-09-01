"""Tests for parse_fastp_json in fastp_summary_parser.py."""
import importlib.util
import json
from pathlib import Path

import pytest
from beegees.utils.configs import get_package_dir

_script = get_package_dir() / "workflow" / "scripts" / "fastp_summary_parser.py"
_spec = importlib.util.spec_from_file_location("fastp_parser", _script)
fastp_parser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fastp_parser)


def _minimal_fastp_json(tmp_path, extra=None):
    """Write a minimal valid fastp JSON and return its path."""
    data = {
        "summary": {
            "before_filtering": {
                "total_reads": 1000000,
                "total_bases": 150000000,
                "q20_bases": 140000000,
                "q30_bases": 130000000,
                "q20_rate": 0.933,
                "q30_rate": 0.867,
                "gc_content": 0.48,
            },
            "after_filtering": {
                "total_reads": 950000,
                "total_bases": 142000000,
                "q20_bases": 138000000,
                "q30_bases": 128000000,
                "q20_rate": 0.972,
                "q30_rate": 0.901,
                "gc_content": 0.48,
            },
        },
        "filtering_result": {
            "passed_filter_reads": 950000,
            "low_quality_reads": 30000,
            "too_many_N_reads": 5000,
            "too_short_reads": 15000,
            "too_long_reads": 0,
        },
        "duplication": {"rate": 0.12},
        "insert_size": {"peak": 180},
    }
    if extra:
        data.update(extra)
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(data))
    return p


class TestParseFastpJson:
    def test_before_filtering_stats(self, tmp_path):
        p = _minimal_fastp_json(tmp_path)
        stats = fastp_parser.parse_fastp_json(str(p))
        assert stats["before_total_reads"] == 1000000
        assert stats["before_gc_content"] == pytest.approx(0.48)

    def test_after_filtering_stats(self, tmp_path):
        p = _minimal_fastp_json(tmp_path)
        stats = fastp_parser.parse_fastp_json(str(p))
        assert stats["after_total_reads"] == 950000
        assert stats["after_q30_rate"] == pytest.approx(0.901)

    def test_filtering_result_stats(self, tmp_path):
        p = _minimal_fastp_json(tmp_path)
        stats = fastp_parser.parse_fastp_json(str(p))
        assert stats["passed_filter_reads"] == 950000
        assert stats["low_quality_reads"] == 30000
        assert stats["too_short_reads"] == 15000

    def test_duplication_rate(self, tmp_path):
        p = _minimal_fastp_json(tmp_path)
        stats = fastp_parser.parse_fastp_json(str(p))
        assert stats["duplication_rate"] == pytest.approx(0.12)

    def test_insert_size_peak(self, tmp_path):
        p = _minimal_fastp_json(tmp_path)
        stats = fastp_parser.parse_fastp_json(str(p))
        assert stats["insert_size_peak"] == 180

    def test_missing_insert_size_returns_empty(self, tmp_path):
        # Single-end fastp output has no insert_size block
        data = json.loads(_minimal_fastp_json(tmp_path).read_text())
        del data["insert_size"]
        p = tmp_path / "se.json"
        p.write_text(json.dumps(data))
        stats = fastp_parser.parse_fastp_json(str(p))
        assert stats["insert_size_peak"] == ""

    def test_missing_duplication_returns_empty(self, tmp_path):
        data = json.loads(_minimal_fastp_json(tmp_path).read_text())
        del data["duplication"]
        p = tmp_path / "nodup.json"
        p.write_text(json.dumps(data))
        stats = fastp_parser.parse_fastp_json(str(p))
        assert stats["duplication_rate"] == ""

    def test_all_expected_keys_present(self, tmp_path):
        p = _minimal_fastp_json(tmp_path)
        stats = fastp_parser.parse_fastp_json(str(p))
        expected_keys = {
            "before_total_reads", "before_total_bases", "before_q20_bases",
            "before_q30_bases", "before_q20_rate", "before_q30_rate", "before_gc_content",
            "after_total_reads", "after_total_bases", "after_q20_bases",
            "after_q30_bases", "after_q20_rate", "after_q30_rate", "after_gc_content",
            "passed_filter_reads", "low_quality_reads", "too_many_N_reads",
            "too_short_reads", "too_long_reads", "duplication_rate", "insert_size_peak",
        }
        assert expected_keys.issubset(set(stats.keys()))
