"""Tests for combine_csv_files in csv_combiner_mge.py."""
import csv
import importlib.util
from pathlib import Path

import pytest
from beegees.utils.configs import get_package_dir

_script = get_package_dir() / "workflow" / "scripts" / "csv_combiner_mge.py"
_spec = importlib.util.spec_from_file_location("csv_combiner", _script)
csv_combiner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(csv_combiner)


def _write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestCombineCsvFiles:
    def test_two_identical_schema_files(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        out = tmp_path / "out.csv"
        _write_csv(f1, [{"x": "1", "y": "2"}])
        _write_csv(f2, [{"x": "3", "y": "4"}])
        csv_combiner.combine_csv_files([str(f1), str(f2)], str(out))
        rows = list(csv.DictReader(open(out)))
        assert len(rows) == 2
        assert rows[0]["x"] == "1"
        assert rows[1]["x"] == "3"

    def test_missing_column_in_second_file_filled_empty(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        out = tmp_path / "out.csv"
        _write_csv(f1, [{"x": "1", "y": "2"}])
        # f2 lacks column "y"
        _write_csv(f2, [{"x": "3"}], fieldnames=["x"])
        csv_combiner.combine_csv_files([str(f1), str(f2)], str(out))
        rows = list(csv.DictReader(open(out)))
        assert rows[1]["y"] == ""

    def test_extra_column_in_second_file_ignored(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        out = tmp_path / "out.csv"
        _write_csv(f1, [{"x": "1"}])
        _write_csv(f2, [{"x": "2", "z": "extra"}])
        csv_combiner.combine_csv_files([str(f1), str(f2)], str(out))
        rows = list(csv.DictReader(open(out)))
        assert "z" not in rows[0]

    def test_single_file(self, tmp_path):
        f1 = tmp_path / "a.csv"
        out = tmp_path / "out.csv"
        _write_csv(f1, [{"x": "1"}, {"x": "2"}])
        csv_combiner.combine_csv_files([str(f1)], str(out))
        rows = list(csv.DictReader(open(out)))
        assert len(rows) == 2

    def test_no_input_files_returns_false(self, tmp_path):
        out = tmp_path / "out.csv"
        result = csv_combiner.combine_csv_files([], str(out))
        assert result is False

    def test_missing_input_file_returns_false(self, tmp_path):
        out = tmp_path / "out.csv"
        result = csv_combiner.combine_csv_files([str(tmp_path / "nonexistent.csv")], str(out))
        assert result is False

    def test_column_order_matches_first_file(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        out = tmp_path / "out.csv"
        _write_csv(f1, [{"a": "1", "b": "2"}], fieldnames=["a", "b"])
        _write_csv(f2, [{"b": "4", "a": "3"}], fieldnames=["b", "a"])
        csv_combiner.combine_csv_files([str(f1), str(f2)], str(out))
        with open(out) as f:
            header = next(csv.reader(f))
        assert header == ["a", "b"]
