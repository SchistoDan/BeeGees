"""Tests for pure functions in 06_aggregate_filter_metrics.py."""
import importlib.util
from pathlib import Path

import pytest
from beegees.utils.configs import get_package_dir

pytest.importorskip("pandas", reason="pandas not installed; skipping aggregate_filter_metrics tests")

_script = get_package_dir() / "workflow" / "scripts" / "06_aggregate_filter_metrics.py"
_spec = importlib.util.spec_from_file_location("agg_metrics", _script)
agg_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg_metrics)


class TestNormaliseBaseName:
    def test_no_suffix_unchanged(self):
        name, suffix = agg_metrics.normalise_base_name("BSNHM089-24")
        assert name == "BSNHM089-24"
        assert suffix == ""

    def test_strips_fcleaner_concat(self):
        name, suffix = agg_metrics.normalise_base_name("BSNHM089-24_fcleaner_concat")
        assert name == "BSNHM089-24"
        assert suffix == "_fcleaner_concat"

    def test_strips_fcleaner_merge(self):
        name, suffix = agg_metrics.normalise_base_name("BSNHM089-24_fcleaner_merge")
        assert name == "BSNHM089-24"
        assert suffix == "_fcleaner_merge"

    def test_strips_fcleaner_se(self):
        name, suffix = agg_metrics.normalise_base_name("BSNHM089-24_fcleaner_se")
        assert name == "BSNHM089-24"
        assert suffix == "_fcleaner_se"

    def test_strips_human_filtered(self):
        name, suffix = agg_metrics.normalise_base_name("BSNHM089-24_human_filtered")
        assert name == "BSNHM089-24"
        assert suffix == "_human_filtered"

    def test_strips_at_filtered(self):
        name, suffix = agg_metrics.normalise_base_name("BSNHM089-24_at_filtered")
        assert name == "BSNHM089-24"
        assert suffix == "_at_filtered"

    def test_strips_outlier_filtered(self):
        name, suffix = agg_metrics.normalise_base_name("BSNHM089-24_outlier_filtered")
        assert name == "BSNHM089-24"
        assert suffix == "_outlier_filtered"

    def test_strips_reference_filtered(self):
        name, suffix = agg_metrics.normalise_base_name("BSNHM089-24_reference_filtered")
        assert name == "BSNHM089-24"
        assert suffix == "_reference_filtered"

    def test_only_one_suffix_removed(self):
        # _fcleaner_concat is longer and matches first; _at_filtered is not stripped further
        name, suffix = agg_metrics.normalise_base_name("BSNHM089-24_at_filtered_fcleaner_concat")
        assert suffix == "_fcleaner_concat"
        assert name == "BSNHM089-24_at_filtered"

    def test_strips_leading_whitespace(self):
        name, _ = agg_metrics.normalise_base_name("  BSNHM089-24")
        assert name == "BSNHM089-24"


class TestAggregateMetrics:
    def _human(self, **kw):
        return {"input_reads": 100, "removed_human": 5, **kw}

    def test_all_sources_present(self):
        human = {"S1": {"input_reads": 100, "removed_human": 5}}
        at = {"S1": {"removed_at_distance": 3}}
        outlier = {"S1": {"removed_outliers": 2}}
        ref = {"S1": {"removed_reference": 1}}
        cons = {"S1": {
            "sample_name": "S1_fcleaner",
            "cleaned_reads": 89,
            "final_ambig_bases": 0,
            "cov_percent": 100.0,
            "cov_avg": 5.0,
            "cov_med": 5.0,
            "cov_max": 8,
            "cov_min": 3,
        }}
        result = agg_metrics.aggregate_metrics(human, at, outlier, ref, cons)
        assert "S1" in result
        row = result["S1"]
        assert row["input_reads"] == 100
        assert row["removed_human"] == 5
        assert row["removed_at_distance"] == 3
        assert row["removed_outliers"] == 2
        assert row["removed_reference"] == 1
        assert row["cleaned_reads"] == 89
        assert row["cov_percent"] == 100.0

    def test_missing_source_fills_zeros(self):
        human = {"S1": {"input_reads": 50, "removed_human": 2}}
        result = agg_metrics.aggregate_metrics(human, {}, {}, {}, {})
        row = result["S1"]
        assert row["removed_at_distance"] == 0
        assert row["removed_outliers"] == 0
        assert row["cleaned_reads"] == 0

    def test_union_of_all_sample_names(self):
        result = agg_metrics.aggregate_metrics(
            {"S1": {"input_reads": 10, "removed_human": 0}},
            {"S2": {"removed_at_distance": 1}},
            {}, {}, {}
        )
        assert "S1" in result
        assert "S2" in result

    def test_empty_inputs_returns_empty(self):
        result = agg_metrics.aggregate_metrics({}, {}, {}, {}, {})
        assert result == {}

    def test_sample_name_from_consensus(self):
        cons = {"S1": {"sample_name": "S1_fcleaner_merge", "cleaned_reads": 10,
                       "final_ambig_bases": 0, "cov_percent": 0, "cov_avg": 0,
                       "cov_med": 0, "cov_max": 0, "cov_min": 0}}
        result = agg_metrics.aggregate_metrics({}, {}, {}, {}, cons)
        assert result["S1"]["sample_name"] == "S1_fcleaner_merge"
