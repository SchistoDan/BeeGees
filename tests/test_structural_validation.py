"""Tests for structural_validation.py pure functions (no nhmmer required)."""
import importlib.util
from pathlib import Path

import pytest
from beegees.utils.configs import get_package_dir

_script = get_package_dir() / "workflow" / "scripts" / "structural_validation.py"
_spec = importlib.util.spec_from_file_location("struct_val", _script)
struct_val = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(struct_val)


class TestTrimNCharacters:
    def test_no_n_unchanged(self):
        assert struct_val.trim_n_characters("ATCG") == "ATCG"

    def test_leading_n_trimmed(self):
        assert struct_val.trim_n_characters("NNATCG") == "ATCG"

    def test_trailing_n_trimmed(self):
        assert struct_val.trim_n_characters("ATCGNN") == "ATCG"

    def test_both_ends_trimmed(self):
        assert struct_val.trim_n_characters("NNATCGNN") == "ATCG"

    def test_internal_n_preserved(self):
        result = struct_val.trim_n_characters("ATNCG")
        assert "N" in result

    def test_all_n_returns_empty(self):
        result = struct_val.trim_n_characters("NNNN")
        assert result == ""

    def test_empty_string(self):
        assert struct_val.trim_n_characters("") == ""


class TestCalculateBarcodeBaseCount:
    def test_counts_non_gap_non_n(self):
        # ATCG → 4 bases
        assert struct_val.calculate_barcode_base_count("ATCG") == 4

    def test_gaps_not_excluded(self):
        # gaps count toward length; only N characters are subtracted
        assert struct_val.calculate_barcode_base_count("AT-CG") == 5

    def test_n_excluded(self):
        assert struct_val.calculate_barcode_base_count("ATNCG") == 4

    def test_empty_sequence(self):
        assert struct_val.calculate_barcode_base_count("") == 0

    def test_all_gaps(self):
        # no N characters, so length is returned unchanged
        assert struct_val.calculate_barcode_base_count("----") == 4


class TestCalculateBarcodeRank:
    def test_rank_1_long_perfect(self):
        rank = struct_val.calculate_barcode_rank(
            barcode_ambiguous_bases_original=0,
            stop_codons=0,
            reading_frame_valid=True,
            barcode_base_count=550,
        )
        assert rank == 1

    def test_rank_2_400_to_499(self):
        rank = struct_val.calculate_barcode_rank(0, 0, True, 450)
        assert rank == 2

    def test_rank_3_300_to_399(self):
        rank = struct_val.calculate_barcode_rank(0, 0, True, 350)
        assert rank == 3

    def test_rank_4_200_to_299(self):
        rank = struct_val.calculate_barcode_rank(0, 0, True, 250)
        assert rank == 4

    def test_rank_5_1_to_199(self):
        rank = struct_val.calculate_barcode_rank(0, 0, True, 100)
        assert rank == 5

    def test_rank_6_original_n_present(self):
        rank = struct_val.calculate_barcode_rank(1, 0, True, 600)
        assert rank == 6

    def test_rank_6_stop_codon(self):
        rank = struct_val.calculate_barcode_rank(0, 1, True, 600)
        assert rank == 6

    def test_rank_6_invalid_reading_frame(self):
        rank = struct_val.calculate_barcode_rank(0, 0, False, 600)
        assert rank == 6


class TestPassesQualityCriteria:
    def _result(self, **kwargs):
        base = {
            "barcode_ambiguous_bases_original": 0,
            "stop_codons": 0,
            "reading_frame": 0,
            "barcode_base_count": 500,
            "barcode_ambiguous_bases": 0,
        }
        base.update(kwargs)
        return base

    def test_good_sequence_passes(self):
        assert struct_val.passes_quality_criteria(self._result()) is True

    def test_original_n_fails(self):
        assert struct_val.passes_quality_criteria(self._result(barcode_ambiguous_bases_original=1)) is False

    def test_stop_codon_fails(self):
        assert struct_val.passes_quality_criteria(self._result(stop_codons=1)) is False

    def test_invalid_reading_frame_fails(self):
        assert struct_val.passes_quality_criteria(self._result(reading_frame=-1)) is False

    def test_too_short_fails(self):
        assert struct_val.passes_quality_criteria(self._result(barcode_base_count=200)) is False

    def test_high_ambiguity_fails(self):
        # 30% ambiguity exactly at threshold → fails (>= 0.30)
        result = self._result(barcode_base_count=100, barcode_ambiguous_bases=30)
        assert struct_val.passes_quality_criteria(result) is False

    def test_just_below_ambiguity_threshold_passes(self):
        # barcode_base_count must be > 300; use 400 to satisfy that check
        result = self._result(barcode_base_count=400, barcode_ambiguous_bases=29)
        assert struct_val.passes_quality_criteria(result) is True


class TestRunNhmmerOnSequence:
    """The nhmmer wrapper: --cpu wiring and temp-file cleanup.

    nhmmer is not available in CI, so subprocess.run is replaced with a fake that
    writes a tabular output file and records the command it was given.
    """

    @staticmethod
    def _fake_nhmmer(recorder, write_tblout=True):
        import subprocess as _subprocess

        def fake_run(cmd, *args, **kwargs):
            recorder.append(list(cmd))
            if write_tblout:
                tblout = cmd[cmd.index('--tblout') + 1]
                # A no-hit tabular output: comment lines only.
                Path(tblout).write_text("# target name  accession  query name\n#\n")
            return _subprocess.CompletedProcess(cmd, 0, "", "")

        return fake_run

    def test_cpu_defaults_to_one(self, monkeypatch):
        calls = []
        monkeypatch.setattr(struct_val.subprocess, "run", self._fake_nhmmer(calls))

        struct_val.run_nhmmer_on_sequence("ACGT" * 20, "SEQ1", "/fake/COI-5P.hmm")

        cmd = calls[0]
        assert cmd[cmd.index('--cpu') + 1] == "1"

    def test_cpu_reflects_threads_argument(self, monkeypatch):
        calls = []
        monkeypatch.setattr(struct_val.subprocess, "run", self._fake_nhmmer(calls))

        struct_val.run_nhmmer_on_sequence("ACGT" * 20, "SEQ1", "/fake/COI-5P.hmm", threads=8)

        cmd = calls[0]
        assert cmd[cmd.index('--cpu') + 1] == "8"

    def test_no_temp_files_left_behind(self, monkeypatch, tmp_path):
        """Regression: the scratch query/tabular files used to leak, two per sequence."""
        tmpdir = tmp_path / "scratch"
        tmpdir.mkdir()
        monkeypatch.setenv("TMPDIR", str(tmpdir))
        monkeypatch.setattr(struct_val.tempfile, "tempdir", None)
        calls = []
        monkeypatch.setattr(struct_val.subprocess, "run", self._fake_nhmmer(calls))

        for i in range(25):
            struct_val.run_nhmmer_on_sequence("ACGT" * 20, f"SEQ{i}", "/fake/COI-5P.hmm")

        assert len(calls) == 25
        # Guard against a vacuous pass: the scratch files must actually have been
        # created under the directory we are then asserting is empty.
        used = [c[c.index('--tblout') + 1] for c in calls]
        assert all(u.startswith(str(tmpdir)) for u in used), used[:2]
        assert list(tmpdir.iterdir()) == []

    def test_temp_files_cleaned_up_when_nhmmer_fails(self, monkeypatch, tmp_path):
        tmpdir = tmp_path / "scratch"
        tmpdir.mkdir()
        monkeypatch.setenv("TMPDIR", str(tmpdir))
        monkeypatch.setattr(struct_val.tempfile, "tempdir", None)

        import subprocess as _subprocess

        seen = []

        def failing_run(cmd, *args, **kwargs):
            seen.append(cmd[cmd.index('--tblout') + 1])
            return _subprocess.CompletedProcess(cmd, 1, "", "nhmmer: fatal error")

        monkeypatch.setattr(struct_val.subprocess, "run", failing_run)

        assert struct_val.run_nhmmer_on_sequence("ACGT" * 20, "SEQ1", "/fake/hmm") is None
        assert seen and seen[0].startswith(str(tmpdir)), seen
        assert list(tmpdir.iterdir()) == []

    def test_missing_tabular_output_is_not_reported_as_missing_nhmmer(self, monkeypatch, caplog):
        """An absent --tblout must read as 'no hits', not 'nhmmer not found in PATH'."""
        calls = []
        monkeypatch.setattr(struct_val.subprocess, "run",
                            self._fake_nhmmer(calls, write_tblout=False))

        result = struct_val.run_nhmmer_on_sequence("ACGT" * 20, "SEQ1", "/fake/hmm")

        assert result is None
        assert "nhmmer not found in PATH" not in caplog.text
