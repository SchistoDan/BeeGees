"""Smoke test: Snakemake dryrun with a minimal config and samples.csv.

Requires snakemake and MitoGeneExtractor to be on PATH (i.e. the BeeGees
conda environment must be active).  The test is automatically skipped when
either is absent, so it is safe to run in bare Python environments.

What this test checks:
- The Snakefile parses without syntax errors.
- Snakemake can resolve the full DAG from a valid config without errors.

It does NOT submit or execute any jobs.
"""
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from beegees.utils.configs import get_package_dir

# Skip the whole module when snakemake is not available.
pytest.importorskip("snakemake", reason="snakemake not installed; skipping dryrun tests")

SNAKEFILE = get_package_dir() / "workflow" / "Snakefile"


@pytest.fixture()
def minimal_run(tmp_path):
    """Create the minimum files needed for a Snakemake dryrun."""
    # Create dummy fastq.gz files — Snakemake --dryrun still requires source
    # input files (those not produced by any rule) to exist on disk.
    r1 = tmp_path / "SAMPLE01_R1.fastq.gz"
    r2 = tmp_path / "SAMPLE01_R2.fastq.gz"
    r1.write_bytes(b"")
    r2.write_bytes(b"")

    # samples.csv pointing at the real (empty) dummy files
    samples = tmp_path / "samples.csv"
    samples.write_text(f"ID,forward,reverse\nSAMPLE01,{r1},{r2}\n")

    # sequence_references.csv — required when run_gene_fetch: false.
    # parse_sequence_references() expects ID and protein_reference_path columns.
    protein_ref = tmp_path / "SAMPLE01_protein_ref.fasta"
    protein_ref.write_bytes(b"")
    seq_refs = tmp_path / "sequence_references.csv"
    seq_refs.write_text(f"ID,protein_reference_path\nSAMPLE01,{protein_ref}\n")

    # Dummy validation paths — validate_taxonomic_validation_config always runs
    # (no toggle), so blast_db/db_taxonomy/expected_taxonomy must exist on disk.
    blast_db = tmp_path / "blast_db"
    blast_db.mkdir()
    db_taxonomy = tmp_path / "db_taxonomy.tsv"
    db_taxonomy.write_bytes(b"")
    expected_taxonomy = tmp_path / "expected_taxonomy.tsv"
    expected_taxonomy.write_bytes(b"")

    # Minimal config pointing at the fake samples file
    config_text = textwrap.dedent(f"""\
        run_name: "DRYRUN_TEST"
        samples_file: "{samples}"
        output_dir: "{tmp_path / 'output'}"
        run_gene_fetch: false
        sequence_reference_file: "{seq_refs}"

        fastp:
          adapter_r1: ""
          adapter_r2: ""
          extra_fastp_args: ""

        downsampling:
          enabled: false
          max_reads: 0

        barcode_recovery:
          r:
            - 1
            - 1.3
            - 1.5
          s:
            - 50
            - 100
          n: 0
          C: 5
          t: 0.5

        fasta_cleaner:
          consensus_threshold: 0.5
          human_threshold: 0.95
          at_difference: 0.1
          at_mode: "absolute"
          outlier_percentile: 90.0
          reference_dir: null
          reference_filter_mode: "remove_similar"

        structural_validation:
          target: "cox1"
          verbose: false

        taxonomic_validation:
          blast_db: "{blast_db}"
          db_taxonomy: "{db_taxonomy}"
          taxval_rank: "family"
          expected_taxonomy: "{expected_taxonomy}"
          verbose: false
          min_pident: 80
          min_length: 100

        rules:
          gene_fetch:
            mem_mb: 4096
            threads: 1
            partition: short
          fastp_qc:
            mem_mb: 4096
            threads: 1
          clean_headers_merge:
            mem_mb: 4096
            threads: 1
          fastq_concat:
            mem_mb: 4096
            threads: 1
          quality_trim:
            mem_mb: 4096
            threads: 1
          downsample:
            mem_mb: 4096
            threads: 1
          MitoGeneExtractor:
            mem_mb: 4096
            threads: 1
            partition: short
          rename_and_combine_cons:
            mem_mb: 4096
            threads: 1
          gzip_merged_clean:
            mem_mb: 4096
            threads: 1
          human_cox1_filter:
            mem_mb: 4096
            threads: 1
          at_content_filter:
            mem_mb: 4096
            threads: 1
          statistical_outlier_filter:
            mem_mb: 4096
            threads: 1
          reference_filter:
            mem_mb: 4096
            threads: 1
          consensus_generation:
            mem_mb: 4096
            threads: 1
          extract_stats_to_csv:
            mem_mb: 4096
            threads: 1
          structural_validation:
            mem_mb: 4096
            threads: 1
            partition: short
          taxonomic_validation:
            mem_mb: 4096
            threads: 4
            partition: short
          blast2taxonomy:
            mem_mb: 4096
            threads: 1
          download_taxdump:
            mem_mb: 4096
            threads: 1
          multiqc_plots:
            mem_mb: 4096
            threads: 1
          multiqc:
            mem_mb: 4096
            threads: 1
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_text)
    return tmp_path, config_file


class TestSnakefileDryrun:
    def test_snakefile_parses_without_syntax_errors(self, minimal_run):
        """snakemake --dryrun should resolve the full DAG without errors."""
        tmp_path, config_file = minimal_run
        result = subprocess.run(
            [
                sys.executable, "-m", "snakemake",
                "--snakefile", str(SNAKEFILE),
                "--configfile", str(config_file),
                "--dryrun",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, (
            f"Snakemake dryrun failed.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
