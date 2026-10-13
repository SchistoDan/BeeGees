"""Tests for 01_human_mitogenome_filter.py core functions.

The most important test here is TestMetricsCsvContract: the previous filter
reported zero human removals on every sample, and an incompatible metrics CSV
would reintroduce exactly that symptom further down the pipeline, silently.
"""
import csv
import importlib.util
import io

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from beegees.utils.configs import get_package_dir

_scripts = get_package_dir() / "workflow" / "scripts"


def _load(name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, _scripts / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hfilter = _load("01_human_mitogenome_filter.py", "human_mitogenome_filter")
aggregate = _load("06_aggregate_filter_metrics.py", "aggregate_filter_metrics")


def _record(seq, rec_id, description=None):
    return SeqRecord(Seq(seq), id=rec_id, description=description or rec_id)


class TestDegapSequences:
    def test_removes_dashes_and_dots(self):
        out = hfilter.degap_sequences([_record("AC--GT..A", "r1")])
        assert str(out[0].seq) == "ACGTA"

    def test_preserves_id_and_description(self):
        rec = _record("A-C", "r1", "r1 some description")
        out = hfilter.degap_sequences([rec])
        assert out[0].id == "r1"
        assert out[0].description == "r1 some description"

    def test_drops_records_that_are_all_gaps(self):
        out = hfilter.degap_sequences([_record("----", "empty"), _record("ACGT", "keep")])
        assert [r.id for r in out] == ["keep"]


class TestWriteFastaUnwrapped:
    def test_one_line_per_record_regardless_of_length(self, tmp_path):
        # 1587 columns is a realistic MGE alignment width; SeqIO would wrap at 60.
        seq = "ACGT-" * 400
        path = tmp_path / "out.fasta"
        hfilter.write_fasta_unwrapped([_record(seq, "r1")], str(path))

        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert lines[1] == seq

    def test_header_written_verbatim(self, tmp_path):
        # Read names carry ':' and '+' which must survive untouched.
        header = "AV251604:3560_JB:1:10301:1480_1:N:0:ATCGTCCTGT+AAGTTACCAG_merged_65_0"
        path = tmp_path / "out.fasta"
        hfilter.write_fasta_unwrapped([_record("ACGT", header, header)], str(path))
        assert path.read_text().splitlines()[0] == f">{header}"

    def test_round_trips_an_alignment_byte_identically(self, tmp_path):
        # Gap columns must survive: structural_validation.py counts them when
        # computing barcode_ambiguous_bases_original.
        source = tmp_path / "align.fas"
        source.write_text(
            ">read_one\nAC--GTAC--GT\n"
            ">read_two\n--CGGTAC--GT\n"
        )
        from Bio import SeqIO
        records = list(SeqIO.parse(str(source), "fasta"))

        out = tmp_path / "copy.fas"
        hfilter.write_fasta_unwrapped(records, str(out))
        assert out.read_bytes() == source.read_bytes()


class TestCheckBwaIndex:
    def test_false_when_index_absent(self, tmp_path, capsys):
        ref = tmp_path / "ref.fasta"
        ref.write_text(">a\nACGT\n")
        assert hfilter.check_bwa_index(str(ref)) is False
        err = capsys.readouterr().err
        assert "BWA index not found" in err
        assert "bwa index" in err

    def test_false_when_index_partial(self, tmp_path):
        ref = tmp_path / "ref.fasta"
        ref.write_text(">a\nACGT\n")
        for ext in [".amb", ".ann", ".bwt"]:
            (tmp_path / f"ref.fasta{ext}").write_text("x")
        assert hfilter.check_bwa_index(str(ref)) is False

    def test_true_when_all_five_present(self, tmp_path):
        ref = tmp_path / "ref.fasta"
        ref.write_text(">a\nACGT\n")
        for ext in hfilter.BWA_INDEX_EXTENSIONS:
            (tmp_path / f"ref.fasta{ext}").write_text("x")
        assert hfilter.check_bwa_index(str(ref)) is True

    def test_does_not_build_an_index(self, tmp_path):
        # Indexing belongs to the bwa_index_human_ref Snakemake rule.
        assert not hasattr(hfilter, "build_bwa_index")


class TestBuildSummaryRow:
    def test_row_matches_column_order(self):
        row = hfilter.build_summary_row({
            'file_path': '/x/UK016-H10_align_1.fas',
            'base_name': 'UK016-H10_1',
            'reason': 'processed',
            'mapped_count': 606,
            'input_count': 643,
            'kept_count': 37,
            'removed_count': 606,
        })
        assert len(row) == len(hfilter.METRICS_COLUMNS)
        as_dict = dict(zip(hfilter.METRICS_COLUMNS, row))
        assert as_dict['sequence_id'] == hfilter.FILE_SUMMARY
        assert as_dict['base_name'] == 'UK016-H10_1'
        assert as_dict['input_count'] == 643
        assert as_dict['removed_count'] == 606
        assert as_dict['mapped_to_human'] == 606

    def test_tolerates_a_partial_error_result(self):
        row = hfilter.build_summary_row({
            'file_path': '/x/y.fas', 'base_name': 'y', 'reason': 'executor_error: boom',
        })
        as_dict = dict(zip(hfilter.METRICS_COLUMNS, row))
        assert as_dict['sequence_id'] == hfilter.FILE_SUMMARY
        assert as_dict['input_count'] == 0


class TestMetricsCsvContract:
    """The CSV this script writes must parse through 06_aggregate_filter_metrics.

    parse_human_metrics() reads only rows where sequence_id == 'FILE_SUMMARY' and
    keys samples off base_name. Its parse is wrapped in a bare try/except, so a
    schema mismatch produces no warning - just removed_human = 0 everywhere.
    """

    def _write_metrics(self, path, results):
        with open(path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(hfilter.METRICS_COLUMNS)
            for r in results:
                writer.writerow(hfilter.build_summary_row(r))

    def test_aggregator_reads_real_counts(self, tmp_path):
        metrics = tmp_path / "human_filter_metrics.csv"
        self._write_metrics(metrics, [{
            'file_path': '/x/UK016-H10_r_1_s_100_align_merge.fas',
            'base_name': 'UK016-H10_r_1_s_100_fcleaner_merge',
            'reason': 'processed',
            'mapped_count': 606,
            'input_count': 643,
            'kept_count': 37,
            'removed_count': 606,
        }])

        parsed = aggregate.parse_human_metrics(str(metrics))

        assert parsed, "aggregator found no FILE_SUMMARY rows"
        key = 'UK016-H10_r_1_s_100'  # _fcleaner_merge suffix is normalised away
        assert key in parsed
        assert parsed[key]['input_reads'] == 643
        assert parsed[key]['removed_human'] == 606

    def test_multiple_samples_are_keyed_separately(self, tmp_path):
        metrics = tmp_path / "m.csv"
        self._write_metrics(metrics, [
            {'file_path': '/a.fas', 'base_name': 'S1_fcleaner_merge', 'reason': 'processed',
             'mapped_count': 10, 'input_count': 100, 'kept_count': 90, 'removed_count': 10},
            {'file_path': '/b.fas', 'base_name': 'S2_fcleaner_merge', 'reason': 'processed',
             'mapped_count': 20, 'input_count': 200, 'kept_count': 180, 'removed_count': 20},
        ])
        parsed = aggregate.parse_human_metrics(str(metrics))
        assert parsed['S1']['removed_human'] == 10
        assert parsed['S2']['removed_human'] == 20

    def test_base_name_column_is_present(self):
        # Dropping it is the specific mistake that makes the aggregator silently
        # report zero removals for every sample.
        assert 'base_name' in hfilter.METRICS_COLUMNS
        assert 'sequence_id' in hfilter.METRICS_COLUMNS
