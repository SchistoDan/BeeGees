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


# nhmmer --tblout column order:
#   target acc query acc hmmfrom hmmto alifrom alito envfrom envto sqlen strand E score bias desc
def _tblout_row(seq_id, hmm_from, hmm_to, seq_from, seq_to, evalue, score=100.0, bias=10.0):
    return (f"{seq_id} - COI-5P - {hmm_from} {hmm_to} {seq_from} {seq_to} "
            f"{seq_from} {seq_to} 1587 + {evalue} {score} {bias} -")


def _tblout(seq_id, *rows):
    return "\n".join(["# target name  accession  query name", "#"] + list(rows))


class TestParseNhmmerResultEnvelopes:
    """nhmmer reports one envelope per contiguous match. Keeping only the best
    one truncated barcodes to whichever fragment scored highest - on UK016-H10 a
    299-column envelope at E=6.1e-63 was silently dropped in favour of a
    291-column one at E=3.9e-73, yielding 290 bases from a consensus with a
    900-base unambiguous stretch."""

    SEQ = "UK016-H10_r_1.5_s_50_fcleaner_concat"

    def test_real_h10_case_keeps_both_envelopes(self):
        content = _tblout(
            self.SEQ,
            _tblout_row(self.SEQ, 367, 657, 462, 752, "3.9e-73", 235.2, 38.6),
            _tblout_row(self.SEQ, 2, 300, 97, 395, "6.1e-63", 201.6, 29.2),
            _tblout_row(self.SEQ, 444, 461, 52, 69, "3.5", -7.0, 6.7),
        )
        result = struct_val.parse_nhmmer_result(content, self.SEQ)

        # Sorted by hmm_from, and the E=3.5 row is excluded as insignificant.
        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(2, 300), (367, 657)]

    def test_single_envelope_returned_as_list(self):
        content = _tblout(self.SEQ, _tblout_row(self.SEQ, 119, 657, 214, 752, "8.8e-124"))
        result = struct_val.parse_nhmmer_result(content, self.SEQ)

        assert len(result) == 1
        assert (result[0]["hmm_from"], result[0]["hmm_to"]) == (119, 657)

    def test_no_significant_envelope_returns_none(self):
        """None, not [] - run_nhmmer_on_sequence hands this to callers testing `is None`."""
        content = _tblout(self.SEQ, _tblout_row(self.SEQ, 444, 461, 52, 69, "3.5"))
        assert struct_val.parse_nhmmer_result(content, self.SEQ) is None

    def test_empty_content_returns_none(self):
        assert struct_val.parse_nhmmer_result("", self.SEQ) is None

    def test_name_mismatch_excluded(self):
        content = _tblout(self.SEQ, _tblout_row("SOME_OTHER_SEQ", 2, 300, 97, 395, "1e-50"))
        assert struct_val.parse_nhmmer_result(content, self.SEQ) is None

    def test_hmm_overlap_keeps_only_best(self):
        content = _tblout(
            self.SEQ,
            _tblout_row(self.SEQ, 100, 400, 100, 400, "1e-90"),
            _tblout_row(self.SEQ, 350, 657, 800, 1107, "1e-50"),  # HMM overlaps 350-400
        )
        result = struct_val.parse_nhmmer_result(content, self.SEQ)

        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(100, 400)]

    def test_sequence_overlap_keeps_only_best(self):
        """Disjoint in HMM space but sharing query bases: merging would place the
        same bases at two model positions."""
        content = _tblout(
            self.SEQ,
            _tblout_row(self.SEQ, 2, 300, 100, 398, "1e-90"),
            _tblout_row(self.SEQ, 367, 657, 350, 640, "1e-50"),  # Seq overlaps 350-398
        )
        result = struct_val.parse_nhmmer_result(content, self.SEQ)

        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(2, 300)]

    def test_best_envelope_always_survives(self):
        """Greedy best-first: the envelope that won under the old best-only logic
        is kept whatever else is present, so a barcode can never get shorter."""
        content = _tblout(
            self.SEQ,
            _tblout_row(self.SEQ, 300, 400, 300, 400, "1e-99"),
            _tblout_row(self.SEQ, 250, 350, 250, 350, "1e-98"),
            _tblout_row(self.SEQ, 350, 450, 350, 450, "1e-97"),
        )
        result = struct_val.parse_nhmmer_result(content, self.SEQ)

        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(300, 400)]

    def test_three_disjoint_envelopes_all_kept_and_sorted(self):
        content = _tblout(
            self.SEQ,
            _tblout_row(self.SEQ, 400, 500, 400, 500, "1e-70"),
            _tblout_row(self.SEQ, 2, 100, 2, 100, "1e-90"),
            _tblout_row(self.SEQ, 200, 300, 200, 300, "1e-80"),
        )
        result = struct_val.parse_nhmmer_result(content, self.SEQ)

        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(2, 100), (200, 300), (400, 500)]

    def test_discarded_overlap_is_logged(self, caplog):
        """The old code dropped significant envelopes with no log line at all."""
        content = _tblout(
            self.SEQ,
            _tblout_row(self.SEQ, 100, 400, 100, 400, "1e-90"),
            _tblout_row(self.SEQ, 350, 657, 800, 1107, "1e-50"),
        )
        with caplog.at_level("INFO"):
            struct_val.parse_nhmmer_result(content, self.SEQ)

        assert "Discarded envelope" in caplog.text
        assert "overlaps accepted envelope" in caplog.text


class TestEnvelopesOverlap:
    @staticmethod
    def _env(hmm_from, hmm_to, seq_from, seq_to):
        return {"hmm_from": hmm_from, "hmm_to": hmm_to,
                "seq_from": seq_from, "seq_to": seq_to}

    def test_disjoint_in_both(self):
        overlaps, _ = struct_val.envelopes_overlap(
            self._env(2, 300, 97, 395), self._env(367, 657, 462, 752))
        assert overlaps is False

    def test_hmm_only(self):
        overlaps, where = struct_val.envelopes_overlap(
            self._env(2, 400, 97, 495), self._env(367, 657, 900, 1190))
        assert overlaps is True
        assert where == "HMM"

    def test_sequence_only(self):
        overlaps, where = struct_val.envelopes_overlap(
            self._env(2, 300, 97, 500), self._env(367, 657, 400, 690))
        assert overlaps is True
        assert where == "sequence"

    def test_adjacent_ranges_do_not_overlap(self):
        """Inclusive coordinates: 1-300 and 301-600 touch but do not share a position."""
        overlaps, _ = struct_val.envelopes_overlap(
            self._env(1, 300, 1, 300), self._env(301, 600, 301, 600))
        assert overlaps is False

    def test_single_shared_position_overlaps(self):
        overlaps, _ = struct_val.envelopes_overlap(
            self._env(1, 300, 1, 300), self._env(300, 600, 900, 1200))
        assert overlaps is True

    def test_minus_strand_coordinates_normalised(self):
        """nhmmer reports minus-strand hits with seq_from > seq_to."""
        overlaps, where = struct_val.envelopes_overlap(
            self._env(2, 300, 500, 100), self._env(367, 657, 400, 690))
        assert overlaps is True
        assert where == "sequence"


class TestConstructHmmSpaceMultipleEnvelopes:
    @staticmethod
    def _env(hmm_from, hmm_to, seq_from, seq_to):
        return {"target_name": "SEQ1", "hmm_from": hmm_from, "hmm_to": hmm_to,
                "seq_from": seq_from, "seq_to": seq_to}

    def test_two_envelopes_both_placed_with_gap_between(self):
        seq = "A" * 10 + "C" * 10          # positions 1-10 A, 11-20 C
        result = struct_val.construct_hmm_space_from_alignment(
            [self._env(1, 10, 1, 10), self._env(21, 30, 11, 20)], seq, 30)

        assert result == "A" * 10 + "-" * 10 + "C" * 10

    def test_single_dict_still_accepted(self):
        result = struct_val.construct_hmm_space_from_alignment(
            self._env(1, 5, 1, 5), "ACGTA", 10)
        assert result == "ACGTA" + "-" * 5

    def test_envelope_order_does_not_matter(self):
        seq = "A" * 10 + "C" * 10
        forward = struct_val.construct_hmm_space_from_alignment(
            [self._env(1, 10, 1, 10), self._env(21, 30, 11, 20)], seq, 30)
        reversed_order = struct_val.construct_hmm_space_from_alignment(
            [self._env(21, 30, 11, 20), self._env(1, 10, 1, 10)], seq, 30)
        assert forward == reversed_order

    def test_h10_barcode_spans_both_envelopes_end_to_end(self):
        """Regression for the 290-base truncation: HMM 2-300 + 367-657 must give
        a 656-position barcode with 590 real bases, not 290."""
        envelopes = [self._env(2, 300, 97, 395), self._env(367, 657, 462, 752)]
        hmm_space = struct_val.construct_hmm_space_from_alignment(envelopes, "A" * 1587, 657)
        barcode = struct_val.trim_sequence_ends(struct_val.replace_gaps_with_n(hmm_space))

        assert len(barcode) == 656
        assert barcode.count("N") == 66                      # HMM 301-366 unfilled
        assert struct_val.calculate_barcode_base_count(barcode) == 590


class TestGapToSpan:
    @staticmethod
    def _env(hmm_from, hmm_to):
        return {"hmm_from": hmm_from, "hmm_to": hmm_to}

    def test_envelope_left_of_span(self):
        # C08: anchor 233-619, candidate 2-50 -> 233 - 50 - 1
        assert struct_val.gap_to_span(self._env(2, 50), 233, 619) == 182

    def test_envelope_right_of_span(self):
        assert struct_val.gap_to_span(self._env(400, 500), 100, 300) == 99

    def test_abutting_ranges_have_zero_gap(self):
        assert struct_val.gap_to_span(self._env(301, 400), 100, 300) == 0

    def test_interior_envelope_is_free(self):
        """An envelope inside the accepted span fills existing N, so it costs nothing."""
        assert struct_val.gap_to_span(self._env(200, 250), 100, 500) == 0


class TestNetGainRule:
    """An extra envelope is merged only if it spans more model positions than the
    gap it opens. The gap has no data and becomes N, so a small distant envelope
    adds more ambiguity than sequence - which is what cost UK016-C08 its barcode."""

    SEQ = "SEQ1"

    def _parse(self, *rows):
        return struct_val.parse_nhmmer_result(_tblout(self.SEQ, *rows), self.SEQ)

    def test_c08_small_distant_envelope_rejected(self):
        """Real case: 49 positions, 182 away. +48 bases for +183 N took the
        barcode from 5% ambiguous (passing) to 33% (failing)."""
        result = self._parse(
            _tblout_row(self.SEQ, 233, 619, 274, 660, "2e-83", 269.1, 28.6),
            _tblout_row(self.SEQ, 2, 50, 43, 91, "3.6e-06", 14.3, 3.4),
        )
        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(233, 619)]

    def test_g07_large_envelope_small_gap_accepted(self):
        """Real case: 316 positions, gap 54. This is the split the envelope fix exists for."""
        result = self._parse(
            _tblout_row(self.SEQ, 373, 657, 462, 746, "4.5e-73", 235.0, 37.5),
            _tblout_row(self.SEQ, 3, 318, 92, 407, "3.7e-61", 195.7, 39.1),
        )
        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(3, 318), (373, 657)]

    def test_e11_accepted(self):
        """Real case: 267 positions, gap 97."""
        result = self._parse(
            _tblout_row(self.SEQ, 2, 263, 91, 352, "1.4e-67", 216.9, 33.1),
            _tblout_row(self.SEQ, 361, 627, 450, 716, "2.6e-48", 153.3, 20.9),
        )
        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(2, 263), (361, 627)]

    def test_h10_both_cases_accepted(self):
        """Real cases: 299 positions vs gap 66, and 142 vs gap 106."""
        wide = self._parse(
            _tblout_row(self.SEQ, 367, 657, 462, 752, "3.9e-73", 235.2, 38.6),
            _tblout_row(self.SEQ, 2, 300, 97, 395, "6.1e-63", 201.6, 29.2),
        )
        assert [(e["hmm_from"], e["hmm_to"]) for e in wide] == [(2, 300), (367, 657)]

        narrow = self._parse(
            _tblout_row(self.SEQ, 367, 657, 462, 752, "7e-73", 234.4, 39.3),
            _tblout_row(self.SEQ, 119, 260, 214, 354, "2.9e-35", 110.3, 18.6),
        )
        assert [(e["hmm_from"], e["hmm_to"]) for e in narrow] == [(119, 260), (367, 657)]

    def test_equal_span_and_gap_rejected(self):
        """Strictly greater: breaking even is not worth the ambiguity."""
        # candidate 1-100 (100 positions), anchor 201-500 -> gap = 201 - 100 - 1 = 100
        result = self._parse(
            _tblout_row(self.SEQ, 201, 500, 201, 500, "1e-90"),
            _tblout_row(self.SEQ, 1, 100, 1, 100, "1e-50"),
        )
        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(201, 500)]

    def test_one_position_better_than_gap_accepted(self):
        # candidate 1-101 (101 positions), anchor 201-500 -> gap 99
        result = self._parse(
            _tblout_row(self.SEQ, 201, 500, 201, 500, "1e-90"),
            _tblout_row(self.SEQ, 1, 101, 1, 101, "1e-50"),
        )
        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(1, 101), (201, 500)]

    def test_rejecting_one_envelope_does_not_widen_the_span(self):
        """A rejected envelope must not extend the span, or a later candidate
        would be measured against a gap that was never opened."""
        result = self._parse(
            _tblout_row(self.SEQ, 2, 100, 2, 100, "1e-90"),
            _tblout_row(self.SEQ, 400, 500, 400, 500, "1e-80"),   # 101 vs gap 299 -> reject
            _tblout_row(self.SEQ, 200, 210, 200, 210, "1e-20"),   # 11 vs gap 99 -> reject
        )
        # If 400-500 had widened the span, 200-210 would have looked interior (free).
        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(2, 100)]

    def test_interior_envelope_free_when_span_already_wide(self):
        result = self._parse(
            _tblout_row(self.SEQ, 2, 300, 2, 300, "1e-90"),
            _tblout_row(self.SEQ, 380, 657, 380, 657, "1e-80"),   # 278 > gap 79 -> accept
            _tblout_row(self.SEQ, 320, 330, 320, 330, "1e-05"),   # 11 positions, interior -> free
        )
        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(2, 300), (320, 330), (380, 657)]

    def test_anchor_is_always_kept(self):
        """Whatever else happens, the best-E-value envelope survives, so a barcode
        can never come out shorter than the pre-fix single-envelope behaviour."""
        result = self._parse(
            _tblout_row(self.SEQ, 300, 400, 300, 400, "1e-99"),
            _tblout_row(self.SEQ, 1, 20, 1, 20, "1e-40"),
            _tblout_row(self.SEQ, 600, 620, 600, 620, "1e-30"),
        )
        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(300, 400)]

    def test_net_loss_is_logged_with_the_comparison(self, caplog):
        with caplog.at_level("INFO"):
            self._parse(
                _tblout_row(self.SEQ, 233, 619, 274, 660, "2e-83"),
                _tblout_row(self.SEQ, 2, 50, 43, 91, "3.6e-06"),
            )
        assert "spans 49 model positions but would open a gap of 182" in caplog.text
        assert "net loss 133" in caplog.text

    def test_net_gain_is_logged_with_the_comparison(self, caplog):
        with caplog.at_level("INFO"):
            self._parse(
                _tblout_row(self.SEQ, 373, 657, 462, 746, "4.5e-73"),
                _tblout_row(self.SEQ, 3, 318, 92, 407, "3.7e-61"),
            )
        assert "span 316 > gap 54" in caplog.text
        assert "net gain +262 model positions" in caplog.text

    def test_overlap_still_takes_precedence(self):
        """Overlap is rejected before the net-gain test is reached."""
        result = self._parse(
            _tblout_row(self.SEQ, 100, 400, 100, 400, "1e-90"),
            _tblout_row(self.SEQ, 350, 657, 800, 1107, "1e-50"),
        )
        assert [(e["hmm_from"], e["hmm_to"]) for e in result] == [(100, 400)]
