"""Tests for tv_local_blast.py: header sanitising, chunking, demux and the chunked run.

No BLAST binary is required - `blastn` is replaced with a fake subprocess.run for the
end-to-end tests, and everything else under test is a pure staticmethod.
"""
import subprocess
from pathlib import Path

import pytest

from tests.conftest import load_script_module

tv_local_blast = load_script_module("tv_local_blast")

BLASTRunner = tv_local_blast.BLASTRunner
QueryRec = tv_local_blast.QueryRec
ChunkSpec = tv_local_blast.ChunkSpec
DemuxError = tv_local_blast.DemuxError
BLAST_TSV_HEADER = tv_local_blast.BLAST_TSV_HEADER

sanitize = BLASTRunner.sanitize_header


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def write_fasta(path, headers, seq="ACGT" * 10, wrap=None):
    """Write a FASTA with the given headers (without '>') and a shared sequence."""
    lines = []
    for header in headers:
        lines.append(f">{header}")
        if wrap:
            lines.extend(seq[i:i + wrap] for i in range(0, len(seq), wrap))
        else:
            lines.append(seq)
    path.write_text("\n".join(lines) + "\n")
    return path


def hit_row(qseqid, sseqid, pident, evalue="1e-50"):
    """One outfmt-6 row with all 13 columns."""
    return "\t".join([
        qseqid, sseqid, str(pident), "658", "0", "0",
        "1", "658", "1", "658", evalue, "1200", f"{sseqid} description",
    ])


# ---------------------------------------------------------------------------
# Existing behaviour
# ---------------------------------------------------------------------------
class TestSanitizeHeader:
    def test_strips_gt(self):
        assert not sanitize(">SAMPLE123").startswith(">")

    def test_replaces_pipe(self):
        assert "|" not in sanitize(">SAMPLE|extra")

    def test_replaces_space(self):
        assert " " not in sanitize(">SAMPLE with spaces")

    def test_replaces_colon(self):
        assert ":" not in sanitize(">SAMPLE:colon")

    def test_replaces_slash(self):
        assert "/" not in sanitize(">path/to/seq")

    def test_plain_id_unchanged_except_gt(self):
        assert sanitize(">BSNHM089-24") == "BSNHM089-24"

    def test_long_header_truncated_at_100(self):
        long_header = ">" + "A" * 200
        assert len(sanitize(long_header)) <= 100

    def test_empty_string(self):
        assert sanitize("") == ""

    def test_no_gt_prefix_unchanged(self):
        assert sanitize("BSNHM089-24") == "BSNHM089-24"


# ---------------------------------------------------------------------------
# collect_pending
# ---------------------------------------------------------------------------
class TestCollectPending:
    def test_all_pending_when_no_outputs_exist(self, tmp_path):
        fasta = write_fasta(tmp_path / "in.fasta", ["SEQ1", "SEQ2", "SEQ3"])
        out = tmp_path / "out"
        out.mkdir()

        scan = BLASTRunner.collect_pending(fasta, out)

        assert scan.records == 3
        assert [r.stem for r in scan.pending] == ["SEQ1", "SEQ2", "SEQ3"]
        assert scan.resumed == 0
        assert scan.skipped == 0
        assert scan.headers == {"SEQ1": ">SEQ1", "SEQ2": ">SEQ2", "SEQ3": ">SEQ3"}
        assert scan.qseqid_to_stem == {"SEQ1": "SEQ1", "SEQ2": "SEQ2", "SEQ3": "SEQ3"}

    def test_existing_output_is_skipped_but_still_tracked(self, tmp_path):
        fasta = write_fasta(tmp_path / "in.fasta", ["SEQ1", "SEQ2"])
        out = tmp_path / "out"
        out.mkdir()
        (out / "SEQ1.tsv").write_text(BLAST_TSV_HEADER)

        scan = BLASTRunner.collect_pending(fasta, out)

        assert [r.stem for r in scan.pending] == ["SEQ2"]
        assert scan.resumed == 1
        # Still present for the summary CSV, which emits a row per input sequence.
        assert "SEQ1" in scan.headers
        assert scan.qseqid_to_stem["SEQ1"] == "SEQ1"

    def test_prefix_used_for_filename_but_not_for_the_key(self, tmp_path):
        fasta = write_fasta(tmp_path / "in.fasta", ["SEQ1"])
        out = tmp_path / "out"
        out.mkdir()
        (out / "plate1_SEQ1.tsv").write_text(BLAST_TSV_HEADER)

        scan = BLASTRunner.collect_pending(fasta, out, prefix="plate1")

        assert scan.pending == []
        assert scan.resumed == 1
        # The dict key stays bare; only the filename carries the prefix.
        assert list(scan.headers) == ["SEQ1"]

    def test_prefix_output_path(self, tmp_path):
        fasta = write_fasta(tmp_path / "in.fasta", ["SEQ1"])
        out = tmp_path / "out"
        out.mkdir()

        scan = BLASTRunner.collect_pending(fasta, out, prefix="plate1")

        assert scan.pending[0].out_path == out / "plate1_SEQ1.tsv"

    def test_duplicate_headers_deduped(self, tmp_path):
        fasta = write_fasta(tmp_path / "in.fasta", ["SEQ1", "SEQ1", "SEQ2"])
        out = tmp_path / "out"
        out.mkdir()

        scan = BLASTRunner.collect_pending(fasta, out)

        assert [r.stem for r in scan.pending] == ["SEQ1", "SEQ2"]
        assert scan.records == 3
        assert scan.skipped == 1

    def test_headers_colliding_after_truncation_deduped(self, tmp_path):
        base = "A" * 100
        fasta = write_fasta(tmp_path / "in.fasta", [base + "_one", base + "_two"])
        out = tmp_path / "out"
        out.mkdir()

        scan = BLASTRunner.collect_pending(fasta, out)

        assert len(scan.pending) == 1
        assert scan.skipped == 1

    def test_header_with_description(self, tmp_path):
        fasta = write_fasta(tmp_path / "in.fasta", ["SEQ1 some description"])
        out = tmp_path / "out"
        out.mkdir()

        scan = BLASTRunner.collect_pending(fasta, out)

        rec = scan.pending[0]
        assert rec.stem == "SEQ1_some_description"
        assert rec.qseqid == "SEQ1"
        assert rec.full_header == ">SEQ1 some description"
        # The index that lets the summary CSV find the hits.
        assert scan.qseqid_to_stem == {"SEQ1": "SEQ1_some_description"}

    def test_bare_gt_header_excluded(self, tmp_path):
        fasta = write_fasta(tmp_path / "in.fasta", ["", "SEQ2"])
        out = tmp_path / "out"
        out.mkdir()

        scan = BLASTRunner.collect_pending(fasta, out)

        assert [r.stem for r in scan.pending] == ["SEQ2"]
        assert scan.skipped == 1

    def test_empty_fasta(self, tmp_path):
        fasta = tmp_path / "in.fasta"
        fasta.write_text("")
        out = tmp_path / "out"
        out.mkdir()

        scan = BLASTRunner.collect_pending(fasta, out)

        assert scan.records == 0
        assert scan.pending == []


# ---------------------------------------------------------------------------
# write_chunks
# ---------------------------------------------------------------------------
class TestWriteChunks:
    def _scan_and_chunk(self, tmp_path, n_records, chunk_size, **kwargs):
        fasta = write_fasta(tmp_path / "in.fasta",
                            [f"SEQ{i}" for i in range(1, n_records + 1)], **kwargs)
        out = tmp_path / "out"
        out.mkdir()
        temp = tmp_path / "tmp"
        temp.mkdir()
        scan = BLASTRunner.collect_pending(fasta, out)
        chunks = BLASTRunner.write_chunks(fasta, scan.pending, chunk_size, temp)
        return scan, chunks

    def test_remainder_chunk(self, tmp_path):
        """120 records at 50 per chunk is 50 / 50 / 20."""
        _, chunks = self._scan_and_chunk(tmp_path, 120, 50)
        assert [len(c.members) for c in chunks] == [50, 50, 20]

    def test_exactly_one_full_chunk(self, tmp_path):
        _, chunks = self._scan_and_chunk(tmp_path, 50, 50)
        assert [len(c.members) for c in chunks] == [50]

    def test_one_over_a_full_chunk(self, tmp_path):
        _, chunks = self._scan_and_chunk(tmp_path, 51, 50)
        assert [len(c.members) for c in chunks] == [50, 1]

    def test_single_record(self, tmp_path):
        _, chunks = self._scan_and_chunk(tmp_path, 1, 50)
        assert [len(c.members) for c in chunks] == [1]

    def test_default_chunk_size_is_50(self):
        assert tv_local_blast.CHUNK_SIZE == 50

    def test_every_record_present_exactly_once(self, tmp_path):
        scan, chunks = self._scan_and_chunk(tmp_path, 120, 50)
        stems = [rec.stem for c in chunks for rec in c.members.values()]
        assert sorted(stems) == sorted(r.stem for r in scan.pending)
        assert len(stems) == len(set(stems))

    def test_member_ids_unique_across_chunks(self, tmp_path):
        _, chunks = self._scan_and_chunk(tmp_path, 120, 50)
        ids = [blast_id for c in chunks for blast_id in c.members]
        assert len(ids) == len(set(ids))

    def test_chunks_in_file_order(self, tmp_path):
        scan, chunks = self._scan_and_chunk(tmp_path, 7, 3)
        stems = [rec.stem for c in chunks for rec in c.members.values()]
        assert stems == [r.stem for r in scan.pending]

    def test_synthetic_deflines_written(self, tmp_path):
        _, chunks = self._scan_and_chunk(tmp_path, 3, 50)
        text = chunks[0].query_fa.read_text()
        assert [line for line in text.splitlines() if line.startswith(">")] == [
            ">q1", ">q2", ">q3"]

    def test_wrapped_sequence_preserved_byte_for_byte(self, tmp_path):
        seq = "ACGT" * 40
        _, chunks = self._scan_and_chunk(tmp_path, 2, 50, seq=seq, wrap=60)
        body = []
        for line in chunks[0].query_fa.read_text().splitlines():
            if not line.startswith(">"):
                body.append(line)
        # Two records, each wrapped identically.
        assert "".join(body) == seq * 2
        assert any(len(line) == 60 for line in body), "wrapping was not preserved"

    def test_tabular_is_not_a_tsv(self, tmp_path):
        """Chunk tabulars must not be picked up by the summary CSV's rglob('*.tsv')."""
        _, chunks = self._scan_and_chunk(tmp_path, 3, 50)
        for chunk in chunks:
            assert chunk.tabular.suffix == ".tab"
            assert chunk.query_fa.suffix == ".fa"

    def test_no_pending_gives_no_chunks(self, tmp_path):
        fasta = write_fasta(tmp_path / "in.fasta", ["SEQ1"])
        temp = tmp_path / "tmp"
        temp.mkdir()
        assert BLASTRunner.write_chunks(fasta, [], 50, temp) == []

    def test_records_not_pending_are_excluded(self, tmp_path):
        fasta = write_fasta(tmp_path / "in.fasta", ["SEQ1", "SEQ2", "SEQ3"])
        out = tmp_path / "out"
        out.mkdir()
        (out / "SEQ2.tsv").write_text(BLAST_TSV_HEADER)
        temp = tmp_path / "tmp"
        temp.mkdir()

        scan = BLASTRunner.collect_pending(fasta, out)
        chunks = BLASTRunner.write_chunks(fasta, scan.pending, 50, temp)

        stems = [rec.stem for c in chunks for rec in c.members.values()]
        assert stems == ["SEQ1", "SEQ3"]
        assert "SEQ2" not in chunks[0].query_fa.read_text()


# ---------------------------------------------------------------------------
# split_chunk_fasta
# ---------------------------------------------------------------------------
class TestSplitChunkFasta:
    def test_splits_into_singletons(self, tmp_path):
        fasta = write_fasta(tmp_path / "in.fasta", ["SEQ1", "SEQ2", "SEQ3"])
        out = tmp_path / "out"
        out.mkdir()
        temp = tmp_path / "tmp"
        temp.mkdir()
        scan = BLASTRunner.collect_pending(fasta, out)
        chunk = BLASTRunner.write_chunks(fasta, scan.pending, 50, temp)[0]

        singles = BLASTRunner.split_chunk_fasta(chunk, temp)

        assert len(singles) == 3
        assert all(len(s.members) == 1 for s in singles)
        assert [next(iter(s.members.values())).stem for s in singles] == [
            "SEQ1", "SEQ2", "SEQ3"]
        for single in singles:
            text = single.query_fa.read_text()
            assert text.count(">") == 1
            assert "ACGT" in text


# ---------------------------------------------------------------------------
# _pident_key
# ---------------------------------------------------------------------------
class TestPidentKey:
    def test_numeric_pident_negated_for_descending_sort(self):
        assert BLASTRunner._pident_key(hit_row("q1", "s1", 98.5)) == -98.5

    def test_non_numeric_pident_is_zero(self):
        row = "\t".join(["q1", "s1", "not-a-number", "658"])
        assert BLASTRunner._pident_key(row) == 0.0

    def test_short_row_is_zero(self):
        assert BLASTRunner._pident_key("q1\ts1") == 0.0

    def test_malformed_rows_sort_last(self):
        rows = ["q1\ts1\tbad\n", hit_row("q1", "s2", 50.0) + "\n",
                hit_row("q1", "s3", 99.0) + "\n"]
        rows.sort(key=BLASTRunner._pident_key)
        assert [r.split("\t")[1] for r in rows] == ["s3", "s2", "s1"]


# ---------------------------------------------------------------------------
# demux_chunk_tabular
# ---------------------------------------------------------------------------
class TestDemuxChunkTabular:
    def _members(self, out_dir, names, prefix=None):
        members = {}
        for i, name in enumerate(names, 1):
            stem = sanitize(f">{name}")
            qseqid = name.split(None, 1)[0]
            filename = f"{prefix}_{stem}.tsv" if prefix else f"{stem}.tsv"
            members[f"q{i}"] = QueryRec(stem=stem, full_header=f">{name}",
                                        qseqid=qseqid, out_path=out_dir / filename)
        return members

    def test_one_tsv_per_member_with_header(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        members = self._members(out, ["SEQ1", "SEQ2", "SEQ3"])
        tabular = tmp_path / "chunk_1.tab"
        tabular.write_text("\n".join([
            hit_row("q1", "ref_a", 99.0),
            hit_row("q1", "ref_b", 95.0),
            hit_row("q3", "ref_c", 88.0),
        ]) + "\n")

        with_hits, without_hits = BLASTRunner.demux_chunk_tabular(tabular, members)

        assert (with_hits, without_hits) == (2, 1)
        for name in ("SEQ1", "SEQ2", "SEQ3"):
            assert (out / f"{name}.tsv").exists()
            assert (out / f"{name}.tsv").read_text().splitlines()[0] == \
                BLAST_TSV_HEADER.rstrip("\n")

    def test_zero_hit_member_gets_header_only_file(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        members = self._members(out, ["SEQ1", "SEQ2"])
        tabular = tmp_path / "chunk_1.tab"
        tabular.write_text(hit_row("q1", "ref_a", 99.0) + "\n")

        BLASTRunner.demux_chunk_tabular(tabular, members)

        assert (out / "SEQ2.tsv").read_text() == BLAST_TSV_HEADER

    def test_rows_sorted_pident_descending(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        members = self._members(out, ["SEQ1"])
        tabular = tmp_path / "chunk_1.tab"
        tabular.write_text("\n".join([
            hit_row("q1", "low", 80.0),
            hit_row("q1", "high", 99.5),
            hit_row("q1", "mid", 90.0),
        ]) + "\n")

        BLASTRunner.demux_chunk_tabular(tabular, members)

        rows = (out / "SEQ1.tsv").read_text().splitlines()[1:]
        assert [r.split("\t")[1] for r in rows] == ["high", "mid", "low"]

    def test_ties_keep_input_order(self, tmp_path):
        """Stability matters: downstream taxonomy takes the first matching hit."""
        out = tmp_path / "out"
        out.mkdir()
        members = self._members(out, ["SEQ1"])
        tabular = tmp_path / "chunk_1.tab"
        tabular.write_text("\n".join([
            hit_row("q1", "first", 99.0),
            hit_row("q1", "second", 99.0),
            hit_row("q1", "third", 99.0),
        ]) + "\n")

        BLASTRunner.demux_chunk_tabular(tabular, members)

        rows = (out / "SEQ1.tsv").read_text().splitlines()[1:]
        assert [r.split("\t")[1] for r in rows] == ["first", "second", "third"]

    def test_query_id_rewritten_to_real_qseqid(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        members = self._members(out, ["SEQ1 with a description"])
        tabular = tmp_path / "chunk_1.tab"
        tabular.write_text(hit_row("q1", "ref_a", 99.0) + "\n")

        BLASTRunner.demux_chunk_tabular(tabular, members)

        rows = (out / "SEQ1_with_a_description.tsv").read_text().splitlines()[1:]
        assert rows[0].split("\t")[0] == "SEQ1"

    def test_unknown_query_id_raises(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        members = self._members(out, ["SEQ1"])
        tabular = tmp_path / "chunk_1.tab"
        tabular.write_text(hit_row("q99", "ref_a", 99.0) + "\n")

        with pytest.raises(DemuxError):
            BLASTRunner.demux_chunk_tabular(tabular, members)

    def test_short_rows_dropped(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        members = self._members(out, ["SEQ1"])
        tabular = tmp_path / "chunk_1.tab"
        tabular.write_text("q1\tref_a\n" + hit_row("q1", "ref_b", 99.0) + "\n")

        BLASTRunner.demux_chunk_tabular(tabular, members)

        rows = (out / "SEQ1.tsv").read_text().splitlines()[1:]
        assert len(rows) == 1
        assert rows[0].split("\t")[1] == "ref_b"

    def test_blank_lines_ignored(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        members = self._members(out, ["SEQ1"])
        tabular = tmp_path / "chunk_1.tab"
        tabular.write_text("\n" + hit_row("q1", "ref_a", 99.0) + "\n\n")

        BLASTRunner.demux_chunk_tabular(tabular, members)

        assert len((out / "SEQ1.tsv").read_text().splitlines()) == 2

    def test_prefix_honoured(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        members = self._members(out, ["SEQ1"], prefix="plate1")
        tabular = tmp_path / "chunk_1.tab"
        tabular.write_text(hit_row("q1", "ref_a", 99.0) + "\n")

        BLASTRunner.demux_chunk_tabular(tabular, members)

        assert (out / "plate1_SEQ1.tsv").exists()

    def test_no_part_files_left_behind(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        members = self._members(out, ["SEQ1", "SEQ2"])
        tabular = tmp_path / "chunk_1.tab"
        tabular.write_text(hit_row("q1", "ref_a", 99.0) + "\n")

        BLASTRunner.demux_chunk_tabular(tabular, members)

        assert list(out.glob("*.part")) == []


# ---------------------------------------------------------------------------
# End-to-end, with blastn faked out
# ---------------------------------------------------------------------------
class FakeBlast:
    """Stand-in for subprocess.run that writes a canned tabular for each chunk.

    Records every invocation, and can be told to fail for chunks containing a
    given synthetic query id.
    """

    def __init__(self, fail_for_stems=(), hits_per_query=2):
        self.calls = []
        # Query ids per invocation, captured while the temp FASTA still exists.
        self.queried = []
        self.fail_for_stems = set(fail_for_stems)
        self.hits_per_query = hits_per_query

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(list(cmd))
        query_fa = Path(cmd[cmd.index('-query') + 1])
        out_path = Path(cmd[cmd.index('-out') + 1])
        ids = [line[1:].strip() for line in query_fa.read_text().splitlines()
               if line.startswith('>')]
        self.queried.append(ids)
        # Failure is keyed off the synthetic ids present in this chunk's FASTA.
        failing = self.fail_for_stems & set(ids)
        if failing:
            raise subprocess.CalledProcessError(
                2, cmd, output="", stderr=f"BLAST Database error for {sorted(failing)}")
        rows = []
        for blast_id in ids:
            for h in range(self.hits_per_query):
                rows.append(hit_row(blast_id, f"ref_{blast_id}_{h}", 99.0 - h))
        out_path.write_text("\n".join(rows) + ("\n" if rows else ""))
        return subprocess.CompletedProcess(cmd, 0, "", "")


class TestChunkedRun:
    def test_120_records_in_3_chunks(self, blast_runner, tmp_path, monkeypatch):
        fasta = write_fasta(tmp_path / "in.fasta",
                            [f"SEQ{i}" for i in range(1, 121)])
        fake = FakeBlast()
        monkeypatch.setattr(tv_local_blast.subprocess, "run", fake)

        processed, skipped, failed = blast_runner.process_fasta_chunked(
            fasta, blast_runner.output_dir)

        assert (processed, skipped, failed) == (120, 0, 0)
        assert len(fake.calls) == 3, "expected ceil(120/50) blastn invocations"
        assert len(list(blast_runner.output_dir.glob("*.tsv"))) == 120
        rows = (blast_runner.output_dir / "SEQ7.tsv").read_text().splitlines()
        assert rows[0] == BLAST_TSV_HEADER.rstrip("\n")
        assert rows[1].split("\t")[0] == "SEQ7"
        assert len(rows) == 3  # header + 2 hits

    def test_resume_only_reblasts_missing(self, blast_runner, tmp_path, monkeypatch):
        fasta = write_fasta(tmp_path / "in.fasta", [f"SEQ{i}" for i in range(1, 6)])
        done = blast_runner.output_dir / "SEQ2.tsv"
        done.write_text(BLAST_TSV_HEADER + hit_row("SEQ2", "kept", 100.0) + "\n")
        original = done.read_text()
        fake = FakeBlast()
        monkeypatch.setattr(tv_local_blast.subprocess, "run", fake)

        processed, skipped, failed = blast_runner.process_fasta_chunked(
            fasta, blast_runner.output_dir)

        assert (processed, skipped, failed) == (4, 1, 0)
        # The pre-existing file is left exactly as it was.
        assert done.read_text() == original
        # Four sequences reached blastn, in one chunk; SEQ2 was never sent.
        assert [len(ids) for ids in fake.queried] == [4]

    def test_failed_chunk_retries_individually(self, blast_runner, tmp_path, monkeypatch):
        fasta = write_fasta(tmp_path / "in.fasta", [f"SEQ{i}" for i in range(1, 6)])
        # SEQ3 is the third record, so it gets synthetic id q3.
        fake = FakeBlast(fail_for_stems={"q3"})
        monkeypatch.setattr(tv_local_blast.subprocess, "run", fake)

        processed, skipped, failed = blast_runner.process_fasta_chunked(
            fasta, blast_runner.output_dir)

        assert (processed, skipped, failed) == (4, 0, 1)
        assert blast_runner.failed_sequences == ["SEQ3"]
        # Its four chunk-mates were salvaged by the individual retry...
        for name in ("SEQ1", "SEQ2", "SEQ4", "SEQ5"):
            assert (blast_runner.output_dir / f"{name}.tsv").exists()
        # ...and the failure left no file that a later resume would trust.
        assert not (blast_runner.output_dir / "SEQ3.tsv").exists()

    def test_no_temp_files_left_behind(self, blast_runner, tmp_path, monkeypatch):
        fasta = write_fasta(tmp_path / "in.fasta", [f"SEQ{i}" for i in range(1, 6)])
        monkeypatch.setattr(tv_local_blast.subprocess, "run", FakeBlast())

        blast_runner.process_fasta_chunked(fasta, blast_runner.output_dir)

        assert list(blast_runner.output_dir.glob("*.tab")) == []
        assert list(blast_runner.output_dir.glob("*.part")) == []
        assert list(blast_runner.output_dir.glob("*.fa")) == []

    def test_nothing_pending_is_a_no_op(self, blast_runner, tmp_path, monkeypatch):
        fasta = write_fasta(tmp_path / "in.fasta", ["SEQ1"])
        (blast_runner.output_dir / "SEQ1.tsv").write_text(BLAST_TSV_HEADER)
        fake = FakeBlast()
        monkeypatch.setattr(tv_local_blast.subprocess, "run", fake)

        processed, skipped, failed = blast_runner.process_fasta_chunked(
            fasta, blast_runner.output_dir)

        assert (processed, skipped, failed) == (0, 1, 0)
        assert fake.calls == []

    def test_process_input_exits_when_sequences_fail(self, blast_runner, tmp_path,
                                                    monkeypatch):
        fasta = write_fasta(tmp_path / "in.fasta", [f"SEQ{i}" for i in range(1, 4)])
        blast_runner.output_csv = str(tmp_path / "summary.csv")
        monkeypatch.setattr(tv_local_blast.subprocess, "run",
                            FakeBlast(fail_for_stems={"q2"}))

        with pytest.raises(SystemExit) as excinfo:
            blast_runner.process_input(fasta)

        assert excinfo.value.code == 1
        # No summary CSV: a failed sequence must not be presented as a no-match.
        assert not Path(blast_runner.output_csv).exists()

    def test_allow_partial_failures_writes_summary(self, blast_runner, tmp_path,
                                                  monkeypatch):
        fasta = write_fasta(tmp_path / "in.fasta", [f"SEQ{i}" for i in range(1, 4)])
        blast_runner.output_csv = "summary.csv"
        blast_runner.allow_partial_failures = True
        monkeypatch.setattr(tv_local_blast.subprocess, "run",
                            FakeBlast(fail_for_stems={"q2"}))

        blast_runner.process_input(fasta)

        assert (blast_runner.output_dir / "summary.csv").exists()


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------
class TestSummaryCsv:
    def _read_csv(self, path):
        import csv as _csv
        with open(path, newline="") as fh:
            return list(_csv.DictReader(fh))

    def test_hits_recorded_for_plain_headers(self, blast_runner, tmp_path, monkeypatch):
        fasta = write_fasta(tmp_path / "in.fasta", ["SEQ1", "SEQ2"])
        blast_runner.output_csv = "summary.csv"
        monkeypatch.setattr(tv_local_blast.subprocess, "run", FakeBlast())

        blast_runner.process_input(fasta)

        rows = {r["seq_id"]: r for r in
                self._read_csv(blast_runner.output_dir / "summary.csv")}
        assert rows["SEQ1"]["hit1"] == "ref_q1_0"
        assert rows["SEQ1"]["hit1_pident"] == "99.0"

    def test_hits_recorded_for_headers_with_a_description(self, blast_runner, tmp_path,
                                                          monkeypatch):
        """Regression: BLAST reports only the first token, so the sanitized full
        header used as the CSV key never matched and every hit was discarded."""
        fasta = write_fasta(tmp_path / "in.fasta", ["SEQ1 some description"])
        blast_runner.output_csv = "summary.csv"
        monkeypatch.setattr(tv_local_blast.subprocess, "run", FakeBlast())

        blast_runner.process_input(fasta)

        rows = self._read_csv(blast_runner.output_dir / "summary.csv")
        assert len(rows) == 1
        assert rows[0]["seq_id"] == "SEQ1_some_description"
        assert rows[0]["original_header"] == ">SEQ1 some description"
        assert rows[0]["hit1"] == "ref_q1_0"
        assert rows[0]["hit1_pident"] == "99.0"

    def test_no_hit_sequence_still_gets_a_row(self, blast_runner, tmp_path, monkeypatch):
        fasta = write_fasta(tmp_path / "in.fasta", ["SEQ1", "SEQ2"])
        blast_runner.output_csv = "summary.csv"
        monkeypatch.setattr(tv_local_blast.subprocess, "run",
                            FakeBlast(hits_per_query=0))

        blast_runner.process_input(fasta)

        rows = self._read_csv(blast_runner.output_dir / "summary.csv")
        assert sorted(r["seq_id"] for r in rows) == ["SEQ1", "SEQ2"]
        assert all(r["hit1"] == "" for r in rows)

    def test_resolve_seq_id_falls_back_to_sanitize(self, blast_runner):
        """A TSV on disk from an earlier run whose input we did not scan."""
        blast_runner.qseqid_to_stem = {}
        assert blast_runner._resolve_seq_id("SEQ1", {"SEQ1": {}}) == "SEQ1"
        assert blast_runner._resolve_seq_id("SEQ9", {"SEQ1": {}}) is None
