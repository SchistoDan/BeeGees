#!/usr/bin/env python3
"""
tv_local_blast.py

This script provides parallel processing capabilities for running BLASTn searches on FASTA files.
It can handle single sequences, multi-FASTA files, or entire directories of FASTA files, and
automatically creates BLAST databases from FASTA files when needed.

Chunked execution:
Query sequences are batched into chunks of CHUNK_SIZE (50) sequences and one blastn process is
launched per chunk, with up to --processes chunks running concurrently. The final chunk holds
whatever remainder is left over. Batching matters because every blastn invocation re-opens and
re-loads the BLAST database; against a large database that fixed startup cost dominates the
search itself, so a 500-sequence run drops from 500 invocations to 10.

Each chunk is written with synthetic query IDs (>q1, >q2, ...) so that results can be
demultiplexed back to their source sequence by exact lookup rather than by re-deriving the
sequence name from BLAST's qseqid (which is only the first whitespace token of the header).
The real query ID is written back into the output before the per-sequence TSV is saved, so
per-sequence output is unchanged.

Database Handling:
The script accepts three types of database input via the -d/--database argument:
1. Directory containing BLAST databases (auto-detects if only one database present)
2. Specific path to an existing BLAST database
3. FASTA file to create a database from (creates nucleotide database using makeblastdb)

When a FASTA file is provided for database creation:
- Checks if a database with the same name already exists in the same directory
- If database exists: uses the existing database (saves time on repeat runs)
- If database doesn't exist: creates a new nucleotide database using makeblastdb
- Database name is derived from the FASTA filename (without extension)

Input Processing:
- Single FASTA files: Processes as single sequence or splits multi-FASTA automatically
- Directories: Processes all .fasta and .fa files found in the directory
- Multi-FASTA files: Automatically chunked for parallel processing
- Output files: Named based on sequence headers, organized in subdirectories for multi-FASTA input
- Sequences whose output TSV already exists are skipped, so an interrupted run resumes cheaply

Output Format:
- Tab-separated values (TSV) format with standard BLAST fields and headers, ordered by pident (highest first)
- Default output format: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore stitle
- One output file per input sequence (sequences with no hits get a header-only file)
- Files named using sanitized sequence headers
- Optional summary CSV file with top hits per query in a flattened format (includes sequences with no hits)

Failure Handling:
If a chunk's blastn call fails, its sequences are retried individually so that one bad record
cannot take down its 49 neighbours. Sequences that still fail are reported and no output file is
written for them (an empty file would be indistinguishable from a genuine no-hit result). The
script then exits non-zero without writing the summary CSV, so a caller such as Snakemake can
retry; successful per-sequence TSVs remain on disk, so each retry only redoes the failures.
Pass --allow-partial-failures to downgrade this to a warning.

Requirements:
- Python 3.6+
- BLAST+ suite (blastn and makeblastdb commands must be in PATH)

Usage Examples:
    # Use existing database directory (auto-detect)
    python tv_local_blast.py -i sequences.fasta -d /path/to/databases/ -o results/

    # Use specific existing database
    python tv_local_blast.py -i sequences.fasta -d /path/to/databases/nt -o results/

    # Create database from FASTA file
    python tv_local_blast.py -i queries.fasta -d reference_genome.fasta -o results/

    # Process directory with custom settings and CSV summary
    python tv_local_blast.py -i /fasta_dir/ -d database.fasta -o results/ -p 16 --output-csv summary.csv

    # Custom BLAST parameters
    python tv_local_blast.py -i input.fasta -d db.fasta -o out/ --blast-opts "-evalue 1e-10 -max_target_seqs 5"

Author: D. Parsons @NHMUK
License: MIT
Version: 2.4
"""
import os
import sys
import subprocess
import argparse
import tempfile
import shutil
import csv
import time
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Tuple, Optional, Dict, Set, NamedTuple
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Number of query sequences per blastn invocation. The final chunk of a run holds
# the remainder (< CHUNK_SIZE). Every blastn call reloads the BLAST database, so
# batching queries is what buys the speed-up; retune here if a measurement on a
# particular database calls for it.
CHUNK_SIZE = 50

# The outfmt 6 column specification, and the matching header line written at the
# top of every per-sequence TSV. Kept together so the two cannot drift apart.
BLAST_OUTFMT_COLS = ("qseqid sseqid pident length mismatch gapopen qstart qend "
                     "sstart send evalue bitscore stitle")
BLAST_TSV_HEADER = "\t".join(BLAST_OUTFMT_COLS.split()) + "\n"


class DemuxError(RuntimeError):
    """A chunk's BLAST output referenced a query ID that is not a chunk member."""


class QueryRec(NamedTuple):
    """One query sequence and where its result should be written."""
    stem: str           # sanitized full header; the summary CSV keys on this
    full_header: str    # original '>...' line, verbatim
    qseqid: str         # first whitespace token, i.e. what BLAST would report
    out_path: Path      # destination TSV


class ChunkSpec(NamedTuple):
    """One blastn invocation: a query FASTA, its output tabular, and its members."""
    index: object                   # int for a normal chunk, 'N.M' for a retry singleton
    query_fa: Path
    tabular: Path
    members: Dict[str, QueryRec]    # synthetic BLAST id ('q7') -> QueryRec


class ScanResult(NamedTuple):
    """Outcome of scanning an input FASTA for work to do."""
    records: int                    # '>' lines seen
    pending: List[QueryRec]         # sequences that still need BLASTing, in file order
    headers: Dict[str, str]         # stem -> full header, for every record seen
    qseqid_to_stem: Dict[str, str]  # qseqid -> stem, for every record seen
    resumed: int                    # skipped because their output already exists
    skipped: int                    # skipped as duplicate or unusable


class ChunkResult(NamedTuple):
    """Outcome of one chunk (including any individual retries it triggered)."""
    processed: int
    with_hits: int
    without_hits: int
    failed: List[str]               # qseqids with no output written
    seconds: float


class BLASTRunner:
    def __init__(self, db_path: str, output_dir: str, blast_options: str, num_processes: int = None,
                 output_csv: str = None, allow_partial_failures: bool = False):
        """
        Initialize BLAST runner with configuration parameters.

        Args:
            db_path: Path to BLAST database directory, specific database, or FASTA file to create database from
            output_dir: Directory to save BLAST results
            blast_options: Additional BLAST options string
            num_processes: Number of concurrent blastn chunks (default: CPU count)
            output_csv: Filename for summary CSV file (optional)
            allow_partial_failures: Continue (with a warning) when some sequences fail BLAST
        """
        self.db_input = Path(db_path)
        self.output_dir = Path(output_dir)
        self.blast_options = blast_options
        self.num_processes = num_processes or cpu_count()
        self.output_csv = output_csv
        self.allow_partial_failures = allow_partial_failures

        # Track all processed sequences for CSV summary
        self.processed_sequences: Dict[str, str] = {}  # seq_id -> original_header
        # Map BLAST's qseqid back to the sanitized seq_id used above. BLAST reports
        # only the first whitespace token, so for a header with a description the
        # two differ and the summary CSV cannot be keyed off qseqid alone.
        self.qseqid_to_stem: Dict[str, str] = {}
        # qseqids that failed BLAST and therefore have no output file
        self.failed_sequences: List[str] = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find/create and validate database
        self.db_path, self.db_name = self._handle_database()

        # Validate setup
        self._validate_setup()

    def _handle_database(self) -> Tuple[Path, str]:
        """
        Handle database input - either find existing database or create from FASTA.

        Returns:
            Tuple of (full_db_path, db_name)
        """
        # Check if input is a FASTA file
        if self.db_input.is_file() and self.db_input.suffix.lower() in ['.fasta', '.fa', '.fas']:
            logger.info(f"FASTA file provided: {self.db_input}")
            return self._handle_fasta_input()
        else:
            # Original behavior - directory or database path
            return self._find_database()

    def _handle_fasta_input(self) -> Tuple[Path, str]:
        """
        Handle FASTA file input - check for existing database or create new one.

        Returns:
            Tuple of (full_db_path, db_name)
        """
        fasta_path = self.db_input
        db_name = fasta_path.stem  # filename without extension
        db_path = fasta_path.parent / db_name

        # Check if database already exists
        if self._is_valid_database(db_path):
            logger.info(f"Database already exists for {fasta_path.name}, using existing database: {db_name}")
            return db_path, db_name
        else:
            logger.info(f"Creating BLAST database from {fasta_path.name}")
            self._create_blast_database(fasta_path, db_path)
            return db_path, db_name

    def _create_blast_database(self, fasta_path: Path, db_path: Path) -> None:
        """
        Create BLAST database from FASTA file.

        Args:
            fasta_path: Path to input FASTA file
            db_path: Path for output database (without extension)
        """
        try:
            cmd = [
                'makeblastdb',
                '-in', str(fasta_path),
                '-dbtype', 'nucl',
                '-out', str(db_path),
                '-title', f"Database created from {fasta_path.name}"
            ]

            logger.info("Running makeblastdb...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.stdout:
                logger.debug(f"makeblastdb output: {result.stdout}")

            logger.info(f"✓ Database created successfully: {db_path}")

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to create BLAST database: {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except FileNotFoundError:
            raise RuntimeError("makeblastdb command not found. Please make sure BLAST+ is installed and in your PATH")

    def _find_database(self) -> Tuple[Path, str]:
        """
        Find BLAST database from the given path.
        Can handle both directory (auto-detect) and specific database path.

        Returns:
            Tuple of (full_db_path, db_name)
        """
        if self.db_input.is_dir():
            # Directory provided - find databases automatically
            available_dbs = self._find_available_databases_in_dir(self.db_input)

            if not available_dbs:
                raise RuntimeError(f"No BLAST databases found in directory: {self.db_input}")
            elif len(available_dbs) == 1:
                # Only one database found - use it automatically
                db_name = available_dbs[0]
                logger.info(f"Auto-detected database: {db_name}")
                return self.db_input / db_name, db_name
            else:
                # Multiple databases found - let user choose
                db_list = '\n'.join([f"  - {db}" for db in available_dbs])
                raise RuntimeError(f"Multiple BLAST databases found in {self.db_input}:\n{db_list}\n"
                                 f"Please specify the exact database path instead of the directory.")
        else:
            # Specific file/database name provided
            if self.db_input.exists():
                # Full path to database file provided
                return self.db_input, self.db_input.name
            else:
                # Check if it's a database name in current directory or if parent dir exists
                parent_dir = self.db_input.parent
                db_name = self.db_input.name

                if parent_dir.exists():
                    # Check if database exists in the parent directory
                    potential_db = parent_dir / db_name
                    if self._is_valid_database(potential_db):
                        return potential_db, db_name

                raise FileNotFoundError(f"Database not found: {self.db_input}")

    def _find_available_databases_in_dir(self, db_dir: Path) -> List[str]:
        """Find available BLAST databases in a directory."""
        if not db_dir.exists():
            return []

        db_files = []
        # Look for nucleotide database files
        for pattern in ['*.nhr', '*.00.nhr', '*.nal']:
            db_files.extend(db_dir.glob(pattern))

        # Extract database names (remove extensions)
        db_names = set()
        for db_file in db_files:
            name = db_file.name
            # Remove common BLAST database extensions
            name = re.sub(r'\.(nhr|nal)$', '', name)
            name = re.sub(r'\.00\.nhr$', '', name)
            db_names.add(name)

        return sorted(list(db_names))

    def _is_valid_database(self, db_path: Path) -> bool:
        """Check if a database path points to a valid BLAST database."""
        db_extensions = ['.nhr', '.00.nhr', '.nal']
        return any((db_path.parent / f"{db_path.name}{ext}").exists() for ext in db_extensions)

    def _validate_setup(self):
        """Validate BLAST installation and database availability."""
        # Check if blastn is available
        try:
            subprocess.run(['blastn', '-version'],
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Error: blastn command not found. Please make sure BLAST+ is installed and in your PATH")

        # Database validation is now handled in _handle_database()
        if not self._is_valid_database(self.db_path):
            available_dbs = self._find_available_databases_in_dir(self.db_path.parent)
            raise RuntimeError(f"Error: BLAST database '{self.db_name}' not found\n"
                             f"Available databases in {self.db_path.parent}: {available_dbs}")

    @staticmethod
    def sanitize_header(header: str) -> str:
        """
        Sanitize FASTA header for use as filename.

        Args:
            header: FASTA header string

        Returns:
            Sanitized string safe for use as filename
        """
        # Remove '>' character and replace problematic characters
        sanitized = header.lstrip('>')
        sanitized = re.sub(r'[/|:*"<>? ]', '_', sanitized)
        # Limit length to avoid filename issues
        return sanitized[:100]

    # ------------------------------------------------------------------
    # Step 1 - scan the input for work to do
    # ------------------------------------------------------------------
    @staticmethod
    def collect_pending(fasta_file: Path, output_dir: Path, prefix: Optional[str] = None) -> ScanResult:
        """
        Scan a (multi-)FASTA once and decide which sequences still need BLASTing.

        Every record seen is recorded in `headers` and `qseqid_to_stem` - including
        records skipped below - because the summary CSV emits a row for every input
        sequence, whether or not it produced hits.

        A record is excluded from `pending` when:
          - its output TSV already exists (resume; counted in `resumed`)
          - its sanitized name duplicates an earlier record (counted in `skipped`)
          - its header yields no usable query ID (counted in `skipped`)

        Args:
            fasta_file: Path to the input FASTA
            output_dir: Directory the per-sequence TSVs are written to
            prefix: Optional filename prefix (used for single-sequence inputs)

        Returns:
            A ScanResult.
        """
        fasta_file = Path(fasta_file)
        output_dir = Path(output_dir)

        pending: List[QueryRec] = []
        headers: Dict[str, str] = {}
        qseqid_to_stem: Dict[str, str] = {}
        seen: Set[str] = set()
        records = 0
        resumed = 0
        skipped = 0

        with open(fasta_file, 'r') as fh:
            for line in fh:
                if not line.startswith('>'):
                    continue
                records += 1
                full_header = line.strip()
                stem = BLASTRunner.sanitize_header(full_header)
                body = full_header.lstrip('>').strip()
                qseqid = body.split(None, 1)[0] if body else ''

                if not qseqid or not stem:
                    skipped += 1
                    logger.warning(f"Skipping record with unusable header: {full_header!r}")
                    continue

                # Track every record for the summary CSV, even if skipped below.
                headers[stem] = full_header
                qseqid_to_stem.setdefault(qseqid, stem)

                if stem in seen:
                    skipped += 1
                    logger.warning(f"Duplicate sequence name after sanitisation, "
                                   f"skipping repeat: {stem}")
                    continue
                seen.add(stem)

                # NOTE: the filename carries the prefix but the summary CSV key
                # (stem) deliberately does not.
                out_name = f"{prefix}_{stem}.tsv" if prefix else f"{stem}.tsv"
                out_path = output_dir / out_name
                if out_path.exists():
                    resumed += 1
                    logger.debug(f"○ Skipping {stem} - output file already exists")
                    continue

                pending.append(QueryRec(stem=stem, full_header=full_header,
                                        qseqid=qseqid, out_path=out_path))

        return ScanResult(records=records, pending=pending, headers=headers,
                          qseqid_to_stem=qseqid_to_stem, resumed=resumed, skipped=skipped)

    # ------------------------------------------------------------------
    # Step 2 - write the chunk FASTAs
    # ------------------------------------------------------------------
    @staticmethod
    def write_chunks(fasta_file: Path, pending: List[QueryRec], chunk_size: int,
                     temp_dir: Path) -> List[ChunkSpec]:
        """
        Stream the input FASTA and write the pending sequences out in contiguous
        chunks of `chunk_size`, in file order. The final chunk holds the remainder.

        Records are written with synthetic query IDs ('>q1', '>q2', ...) so that
        results can be demultiplexed by exact lookup. Sequence lines are copied
        byte-for-byte, so the search itself is unaffected - blastn scores sequence,
        not the defline.

        Args:
            fasta_file: Path to the input FASTA
            pending: Sequences to include (from collect_pending)
            chunk_size: Maximum sequences per chunk
            temp_dir: Directory for the chunk FASTAs and their tabular outputs

        Returns:
            List of ChunkSpec, in file order.
        """
        temp_dir = Path(temp_dir)
        pending_by_stem = {rec.stem: rec for rec in pending}
        chunks: List[ChunkSpec] = []
        if not pending_by_stem:
            return chunks

        handle = None
        query_fa = None
        chunk_index = 0
        members: Dict[str, QueryRec] = {}
        global_index = 0
        emitted: Set[str] = set()
        include = False

        with open(fasta_file, 'r') as fh:
            for line in fh:
                if line.startswith('>'):
                    stem = BLASTRunner.sanitize_header(line.strip())
                    rec = pending_by_stem.get(stem)
                    include = rec is not None and stem not in emitted
                    if not include:
                        continue
                    emitted.add(stem)

                    # Start a new chunk on the first record and whenever the
                    # current one is full.
                    if handle is None or len(members) >= chunk_size:
                        if handle is not None:
                            handle.close()
                            chunks.append(ChunkSpec(
                                index=chunk_index,
                                query_fa=query_fa,
                                tabular=temp_dir / f"chunk_{chunk_index}.tab",
                                members=members,
                            ))
                        chunk_index += 1
                        query_fa = temp_dir / f"chunk_{chunk_index}.fa"
                        handle = open(query_fa, 'w')
                        members = {}

                    global_index += 1
                    blast_id = f"q{global_index}"
                    members[blast_id] = rec
                    handle.write(f">{blast_id}\n")
                elif include and handle is not None:
                    handle.write(line)

        if handle is not None:
            handle.close()
            chunks.append(ChunkSpec(
                index=chunk_index,
                query_fa=query_fa,
                tabular=temp_dir / f"chunk_{chunk_index}.tab",
                members=members,
            ))

        return chunks

    @staticmethod
    def split_chunk_fasta(spec: ChunkSpec, temp_dir: Path) -> List[ChunkSpec]:
        """
        Split an already-written chunk FASTA into one single-record ChunkSpec per
        member, reusing the synthetic IDs. Used to retry a failed chunk one
        sequence at a time so a single bad record cannot take down its neighbours.
        """
        temp_dir = Path(temp_dir)
        singles: List[ChunkSpec] = []
        handle = None

        with open(spec.query_fa, 'r') as fh:
            for line in fh:
                if line.startswith('>'):
                    if handle is not None:
                        handle.close()
                        handle = None
                    blast_id = line[1:].strip()
                    rec = spec.members.get(blast_id)
                    if rec is None:
                        continue
                    n = len(singles) + 1
                    fa = temp_dir / f"chunk_{spec.index}_{n}.fa"
                    handle = open(fa, 'w')
                    handle.write(line)
                    singles.append(ChunkSpec(
                        index=f"{spec.index}.{n}",
                        query_fa=fa,
                        tabular=temp_dir / f"chunk_{spec.index}_{n}.tab",
                        members={blast_id: rec},
                    ))
                elif handle is not None:
                    handle.write(line)

        if handle is not None:
            handle.close()

        return singles

    # ------------------------------------------------------------------
    # Step 3 - run blastn on a chunk
    # ------------------------------------------------------------------
    def _run_blast_chunk(self, spec: ChunkSpec) -> None:
        """
        Run blastn over one chunk FASTA. Raises subprocess.CalledProcessError on
        a non-zero exit.
        """
        cmd = [
            'blastn',
            '-query', str(spec.query_fa),
            '-db', str(self.db_path),
            '-out', str(spec.tabular),
            '-outfmt', f'6 {BLAST_OUTFMT_COLS}'
        ] + self.blast_options.split()

        logger.debug(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    # ------------------------------------------------------------------
    # Step 4 - demultiplex a chunk's results back to per-sequence TSVs
    # ------------------------------------------------------------------
    @staticmethod
    def _pident_key(row: str) -> float:
        """
        Sort key placing higher percent identity first. Rows whose pident cannot
        be parsed sort last, matching the previous per-sequence sort behaviour.
        Used with a stable sort, so BLAST's own ordering is preserved among hits
        of equal percent identity - downstream taxonomy picks the first matching
        hit, so tie order is significant.
        """
        parts = row.rstrip('\n').split('\t')
        try:
            return -float(parts[2])
        except (ValueError, IndexError):
            return 0.0

    @staticmethod
    def demux_chunk_tabular(tabular: Path, members: Dict[str, QueryRec]) -> Tuple[int, int]:
        """
        Split one chunk's BLAST tabular into the per-sequence TSVs.

        Rows are grouped by their synthetic query ID, column 0 is rewritten to the
        real query ID, and each group is written pident-descending under the standard
        header. Members that returned no rows get a header-only file, which is what
        a no-hit query produced previously and what the summary CSV expects to find.

        Files are written to a '.part' path and renamed into place, so an interrupted
        run cannot leave a truncated TSV that a later run mistakes for complete.

        Args:
            tabular: Chunk output written by blastn
            members: Synthetic query ID -> QueryRec for this chunk

        Returns:
            Tuple of (queries_with_hits, queries_without_hits)

        Raises:
            DemuxError: if BLAST reported a query ID that is not a chunk member.
        """
        groups: Dict[str, List[str]] = {}

        with open(tabular, 'r') as fh:
            for line in fh:
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                parts = line.split('\t')
                # Rows with fewer than 3 columns carry no usable pident and were
                # discarded by the previous sort step too.
                if len(parts) < 3:
                    continue
                rec = members.get(parts[0])
                if rec is None:
                    raise DemuxError(
                        f"BLAST reported query id {parts[0]!r} which is not a member of "
                        f"{tabular}; refusing to discard its hits"
                    )
                parts[0] = rec.qseqid
                groups.setdefault(rec.stem, []).append('\t'.join(parts) + '\n')

        with_hits = 0
        without_hits = 0
        for rec in members.values():
            rows = groups.get(rec.stem, [])
            rows.sort(key=BLASTRunner._pident_key)
            part_path = Path(str(rec.out_path) + '.part')
            with open(part_path, 'w') as out:
                out.write(BLAST_TSV_HEADER)
                out.writelines(rows)
            os.replace(part_path, rec.out_path)
            if rows:
                with_hits += 1
            else:
                without_hits += 1

        return with_hits, without_hits

    # ------------------------------------------------------------------
    # Step 5 - chunk orchestration
    # ------------------------------------------------------------------
    def _process_chunk(self, spec: ChunkSpec, temp_dir: Path) -> ChunkResult:
        """
        Run one chunk and demultiplex its output. On a blastn failure, retry the
        chunk's sequences individually so one bad record does not lose the rest.
        """
        start = time.monotonic()
        try:
            self._run_blast_chunk(spec)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or '')[:2000]
            logger.error(f"✗ chunk {spec.index} ({len(spec.members)} queries) "
                         f"FAILED rc={exc.returncode}: {stderr}")
            logger.error(f"  members: "
                         f"{', '.join(r.qseqid for r in spec.members.values())}")
            return self._retry_chunk_individually(spec, temp_dir, start)
        except Exception as exc:
            logger.error(f"✗ chunk {spec.index} ({len(spec.members)} queries) "
                         f"raised {type(exc).__name__}: {exc}")
            return self._retry_chunk_individually(spec, temp_dir, start)

        try:
            with_hits, without_hits = self.demux_chunk_tabular(spec.tabular, spec.members)
        except DemuxError as exc:
            # Retrying will not help - the output does not match the input we sent.
            logger.error(f"✗ chunk {spec.index}: {exc}")
            return ChunkResult(processed=0, with_hits=0, without_hits=0,
                               failed=[r.qseqid for r in spec.members.values()],
                               seconds=time.monotonic() - start)
        finally:
            # Keep peak temp usage to one live tabular per worker.
            try:
                os.remove(spec.tabular)
            except OSError:
                pass

        return ChunkResult(processed=len(spec.members), with_hits=with_hits,
                           without_hits=without_hits, failed=[],
                           seconds=time.monotonic() - start)

    def _retry_chunk_individually(self, spec: ChunkSpec, temp_dir: Path,
                                  start: float) -> ChunkResult:
        """Re-run a failed chunk's sequences one at a time to isolate the culprit."""
        singles = self.split_chunk_fasta(spec, temp_dir)
        logger.info(f"[chunk {spec.index}] Retrying {len(singles)} sequences individually...")

        processed = 0
        with_hits = 0
        without_hits = 0
        failed: List[str] = []

        for single in singles:
            rec = next(iter(single.members.values()))
            try:
                self._run_blast_chunk(single)
                hits, no_hits = self.demux_chunk_tabular(single.tabular, single.members)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or '')[:1000]
                logger.error(f"[chunk {spec.index}] ✗ {rec.qseqid} failed on individual "
                             f"retry rc={exc.returncode}: {stderr}")
                failed.append(rec.qseqid)
                continue
            except Exception as exc:
                logger.error(f"[chunk {spec.index}] ✗ {rec.qseqid} failed on individual "
                             f"retry: {type(exc).__name__}: {exc}")
                failed.append(rec.qseqid)
                continue
            finally:
                for path in (single.query_fa, single.tabular):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            processed += 1
            with_hits += hits
            without_hits += no_hits

        logger.info(f"[chunk {spec.index}] Recovered {processed}/{len(singles)} sequences")
        return ChunkResult(processed=processed, with_hits=with_hits,
                           without_hits=without_hits, failed=failed,
                           seconds=time.monotonic() - start)

    def process_fasta_chunked(self, fasta_file: Path, output_dir: Path,
                              prefix: Optional[str] = None) -> Tuple[int, int, int]:
        """
        BLAST one (multi-)FASTA in chunks of CHUNK_SIZE sequences.

        Args:
            fasta_file: Path to the input FASTA
            output_dir: Directory to write the per-sequence TSVs into
            prefix: Optional filename prefix (used for single-sequence inputs)

        Returns:
            Tuple of (processed_count, skipped_count, failed_count)
        """
        fasta_file = Path(fasta_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scan = self.collect_pending(fasta_file, output_dir, prefix)
        # Populate shared state here, in the main thread, before any worker starts.
        self.processed_sequences.update(scan.headers)
        self.qseqid_to_stem.update(scan.qseqid_to_stem)

        logger.info(f"Scanned {scan.records} records in {fasta_file.name}: "
                    f"{len(scan.pending)} pending, {scan.resumed} already complete (resume), "
                    f"{scan.skipped} skipped (duplicate/unusable)")

        if not scan.pending:
            logger.info("Nothing to BLAST - all outputs already present or no usable records")
            return 0, scan.resumed, 0

        temp_dir = Path(tempfile.mkdtemp())
        try:
            chunks = self.write_chunks(fasta_file, scan.pending, CHUNK_SIZE, temp_dir)
            final_size = len(chunks[-1].members) if chunks else 0
            logger.info(f"Chunking {len(scan.pending)} pending sequences into "
                        f"{len(chunks)} chunk(s) of {CHUNK_SIZE} (final chunk: {final_size})")

            processed = 0
            failed: List[str] = []
            workers = max(1, min(self.num_processes, len(chunks)))
            logger.info(f"Running up to {workers} concurrent blastn chunk(s)")

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self._process_chunk, spec, temp_dir): spec
                           for spec in chunks}
                for i, future in enumerate(as_completed(futures), 1):
                    spec = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.error(f"[{i}/{len(chunks)}] ✗ chunk {spec.index} raised "
                                     f"{type(exc).__name__}: {exc}")
                        failed.extend(r.qseqid for r in spec.members.values())
                        continue
                    processed += result.processed
                    failed.extend(result.failed)
                    glyph = '✗' if result.failed else '✓'
                    suffix = f", {len(result.failed)} failed" if result.failed else ''
                    logger.info(f"[{i}/{len(chunks)}] {glyph} chunk {spec.index} "
                                f"({len(spec.members)} queries) - {result.with_hits} with hits, "
                                f"{result.without_hits} no hits{suffix} - {result.seconds:.1f}s")

            self.failed_sequences.extend(failed)
            logger.info(f"Completed {fasta_file.name}: {processed} processed, "
                        f"{scan.resumed} skipped, {len(failed)} failed - "
                        f"{len(chunks)} blastn invocation(s)")
            return processed, scan.resumed, len(failed)
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)

    def process_directory_parallel(self, input_dir: Path) -> None:
        """
        Process all FASTA files in a directory.

        Args:
            input_dir: Directory containing FASTA files
        """
        # Find all FASTA files
        fasta_patterns = ['*.fasta', '*.fa']
        fasta_files = []
        for pattern in fasta_patterns:
            fasta_files.extend(input_dir.glob(pattern))

        if not fasta_files:
            raise ValueError(f"No FASTA files found in {input_dir}")

        total_files = len(fasta_files)
        logger.info(f"Found {total_files} FASTA files to process")
        logger.info("Starting BLAST searches...")

        dir_processed_count = 0
        dir_skipped_count = 0

        for i, fasta_file in enumerate(fasta_files, 1):
            filename = fasta_file.name
            base_name = fasta_file.stem

            logger.info(f"[{i}/{total_files}] Processing file: {filename}")

            # Count sequences in file
            with open(fasta_file, 'r') as f:
                seq_count = sum(1 for line in f if line.startswith('>'))

            if seq_count == 0:
                logger.error(f"  ✗ Error: No FASTA sequences found in {fasta_file}")
                continue
            elif seq_count == 1:
                logger.info("  Single sequence FASTA file")
                processed, skipped, _ = self.process_fasta_chunked(
                    fasta_file, self.output_dir, prefix=base_name)
            else:
                logger.info(f"  Multi-FASTA file with {seq_count} sequences")

                # Create subdirectory for this file's results
                file_output_dir = self.output_dir / base_name
                file_output_dir.mkdir(exist_ok=True)

                processed, skipped, _ = self.process_fasta_chunked(
                    fasta_file, file_output_dir)

            dir_processed_count += processed
            dir_skipped_count += skipped

        logger.info(f"Directory processing completed: {dir_processed_count} sequences processed, {dir_skipped_count} sequences skipped")

    def _generate_summary_csv(self) -> None:
        """
        Generate summary CSV file from all TSV files in the output directory.
        Only includes the first 100 hits per sample in the CSV output.
        Includes sequences with no BLAST hits and gapopen data.
        """
        if not self.output_csv:
            return

        logger.info("Generating summary CSV file...")

        # Find all TSV files recursively in output directory
        tsv_files = list(self.output_dir.rglob("*.tsv"))

        if not tsv_files:
            logger.warning("No TSV files found for CSV summary generation")
            return

        # Limit max hits in CSV to 100
        max_csv_hits = 100

        # Extract max_target_seqs from blast_options (for processing TSV files)
        max_hits = 1000  # default
        if 'max_target_seqs' in self.blast_options:
            match = re.search(r'max_target_seqs\s+(\d+)', self.blast_options)
            if match:
                max_hits = int(match.group(1))

        # Create column headers - now includes hitN_gaps, limited to 100 hits
        headers = ['seq_id', 'original_header']
        for i in range(1, max_csv_hits + 1):
            headers.extend([f'hit{i}', f'hit{i}_pident', f'hit{i}_length', f'hit{i}_mismatch', f'hit{i}_gaps', f'hit{i}_evalue'])

        # Dictionary to store results for each sequence
        sequence_results = {}

        # Initialize all sequences with empty results
        for seq_id, original_header in self.processed_sequences.items():
            sequence_results[seq_id] = {
                'original_header': original_header,
                'hits': []
            }

        # Process TSV files to populate hits
        for tsv_file in tsv_files:
            try:
                # Read TSV file
                with open(tsv_file, 'r') as f:
                    lines = f.readlines()

                if not lines:
                    # Empty file - find corresponding sequence ID from filename
                    tsv_filename = tsv_file.stem
                    # Try to match with processed sequences
                    for seq_id in self.processed_sequences:
                        if seq_id in tsv_filename or tsv_filename.endswith(seq_id):
                            logger.debug(f"No hits found for sequence: {seq_id}")
                            break
                    continue

                # Parse TSV data (skip header if present)
                current_qseqid = None
                current_hits = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Skip header line
                    parts = line.split('\t')
                    if parts[0] == 'qseqid':
                        continue

                    if len(parts) < 13:  # Ensure we have all required columns
                        continue

                    qseqid = parts[0]
                    sseqid = parts[1]
                    pident = parts[2]
                    length = parts[3]
                    mismatch = parts[4]
                    gapopen = parts[5]  # Extract gapopen
                    evalue = parts[10]

                    # If this is a new query sequence, process the previous one
                    if current_qseqid is not None and qseqid != current_qseqid:
                        matching_seq_id = self._resolve_seq_id(current_qseqid, sequence_results)

                        if matching_seq_id and matching_seq_id in sequence_results:
                            # Limit to max_hits when saving from the file, but will be further limited in CSV output
                            sequence_results[matching_seq_id]['hits'] = current_hits[:max_hits]

                        current_hits = []

                    current_qseqid = qseqid
                    current_hits.append([sseqid, pident, length, mismatch, gapopen, evalue])

                # Process the last query sequence
                if current_qseqid is not None:
                    matching_seq_id = self._resolve_seq_id(current_qseqid, sequence_results)

                    if matching_seq_id and matching_seq_id in sequence_results:
                        # Limit to max_hits when saving from the file, but will be further limited in CSV output
                        sequence_results[matching_seq_id]['hits'] = current_hits[:max_hits]

            except Exception as e:
                logger.warning(f"Error processing TSV file {tsv_file}: {str(e)}")
                continue

        # Count sequences with and without hits correctly
        sequences_with_hits = len([seq for seq in sequence_results.values() if seq['hits']])
        sequences_without_hits = len([seq for seq in sequence_results.values() if not seq['hits']])

        # Prepare CSV data - limiting to max_csv_hits
        csv_data = []
        for seq_id, result in sequence_results.items():
            row_data = [seq_id, result['original_header']]

            # Add hit data (up to max_csv_hits) - includes gapopen
            for i in range(max_csv_hits):
                if i < len(result['hits']):
                    row_data.extend(result['hits'][i])
                else:
                    row_data.extend(['', '', '', '', '', ''])  # Empty values for missing hits (6 fields)

            csv_data.append(row_data)

        # Write CSV file
        try:
            csv_path = self.output_dir / self.output_csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerows(csv_data)

            logger.info(f"✓ Summary CSV file created: {csv_path}")
            logger.info(f"  - Total sequences: {len(sequence_results)}")
            logger.info(f"  - Sequences with hits: {sequences_with_hits}")
            logger.info(f"  - Sequences without hits: {sequences_without_hits}")
            logger.info(f"  - Included up to {max_csv_hits} hits per query (out of maximum {max_hits})")

        except Exception as e:
            logger.error(f"Failed to create summary CSV file: {str(e)}")

    def _resolve_seq_id(self, qseqid: str, sequence_results: Dict) -> Optional[str]:
        """
        Map a BLAST qseqid back to the key used in the summary CSV.

        BLAST reports only the first whitespace token of a header, whereas the CSV
        is keyed on the sanitized *full* header, so for '>SEQ1 some description'
        the two differ ('SEQ1' vs 'SEQ1_some_description'). The index built while
        scanning the input resolves this; the sanitize fallback covers TSVs found
        on disk from an earlier run whose input was not scanned this time.
        """
        seq_id = self.qseqid_to_stem.get(qseqid)
        if seq_id is not None and seq_id in sequence_results:
            return seq_id
        fallback = self.sanitize_header(qseqid)
        return fallback if fallback in sequence_results else None

    def process_input(self, input_path: Path) -> None:
        """
        Process input (either directory or single file).

        Args:
            input_path: Path to input directory or file
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input path {input_path} does not exist")

        # Display configuration
        logger.info("===== BLAST Configuration =====")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"BLAST options: {self.blast_options}")
        logger.info(f"Sequences per chunk: {CHUNK_SIZE}")
        logger.info(f"Concurrent chunks: {self.num_processes}")
        if self.output_csv:
            logger.info(f"Summary CSV: {self.output_csv}")
        logger.info("===============================")

        if input_path.is_dir():
            logger.info("Input is a directory. Processing all FASTA files...")
            self.process_directory_parallel(input_path)
        else:
            logger.info("Input is a single file...")

            # Validate file extension (with warning)
            if input_path.suffix not in ['.fa', '.fasta']:
                logger.warning("Input file does not have .fa or .fasta extension. Continuing anyway...")

            filename = input_path.name
            base_name = input_path.stem

            # Count sequences
            with open(input_path, 'r') as f:
                seq_count = sum(1 for line in f if line.startswith('>'))

            if seq_count == 0:
                raise ValueError(f"No FASTA sequences found in {input_path}")
            elif seq_count == 1:
                logger.info(f"Processing single sequence FASTA file: {filename}")
                self.process_fasta_chunked(input_path, self.output_dir, prefix=base_name)
            else:
                logger.info(f"Processing multi-FASTA file: {filename} with {seq_count} sequences")
                self.process_fasta_chunked(input_path, self.output_dir)

        # A sequence with no output file is not the same as a sequence with no
        # hits: writing the summary CSV anyway would present a BLAST failure as a
        # genuine no-match to downstream taxonomy assignment. Fail loudly instead
        # so the caller can retry - the TSVs that did succeed stay on disk, so a
        # retry only redoes the failures.
        if self.failed_sequences:
            logger.error(f"{len(self.failed_sequences)} sequence(s) failed BLAST and have no output:")
            for qseqid in self.failed_sequences:
                logger.error(f"  - {qseqid}")
            if not self.allow_partial_failures:
                logger.error("Not writing summary CSV. Re-run to retry only the failed sequences.")
                sys.exit(1)
            logger.warning("--allow-partial-failures set: continuing with an incomplete result set")

        # Generate summary CSV if requested
        if self.output_csv:
            self._generate_summary_csv()

        logger.info(f"BLAST searches completed. Results saved to {self.output_dir}")


def main():
    """Main function to run the BLAST script."""
    parser = argparse.ArgumentParser(
        description="Run BLASTn on FASTA files in parallel, in chunks of "
                    f"{CHUNK_SIZE} sequences per blastn invocation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect database in directory (works if only one database present)
  python tv_local_blast.py -i sequences.fasta -d /path/to/databases/ -o results/

  # Specify exact database path
  python tv_local_blast.py -i sequences.fasta -d /path/to/databases/nt -o results/

  # Create database from FASTA file (or use existing if present)
  python tv_local_blast.py -i queries.fasta -d reference.fasta -o results/

  # Process directory with custom options and generate summary CSV
  python tv_local_blast.py -i /fasta_dir/ -d /db_path/nr -o results/ -p 16 --output-csv summary.csv

  # Custom BLAST parameters
  python tv_local_blast.py -i input.fasta -d db/mydb -o out/ --blast-opts "-evalue 1e-10 -max_target_seqs 10"
        """
    )

    # Required arguments
    parser.add_argument('-i', '--input', required=True, type=Path,
                        help='Input FASTA file or directory containing FASTA files')
    parser.add_argument('-d', '--database', required=True, type=Path,
                        help='Path to BLAST database directory (auto-detects databases), specific database path, or FASTA file to create database from')
    parser.add_argument('-o', '--output', required=True, type=Path,
                        help='Output directory for BLAST results')

    # Optional arguments
    parser.add_argument('-p', '--processes', type=int, default=None,
                        help=f'Number of concurrent blastn chunks (default: {cpu_count()})')
    parser.add_argument('--blast-opts', default='-evalue 1e-5 -max_target_seqs 100 -num_threads 1',
                        help='Additional BLAST options (default: "-evalue 1e-5 -max_target_seqs 100 -num_threads 1")')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Generate summary CSV file with specified filename (e.g. "summary.csv")')
    parser.add_argument('--allow-partial-failures', action='store_true',
                        help='Warn instead of failing when some sequences could not be BLASTed')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Create BLAST runner
        blast_runner = BLASTRunner(
            db_path=str(args.database),
            output_dir=str(args.output),
            blast_options=args.blast_opts,
            num_processes=args.processes,
            output_csv=args.output_csv,
            allow_partial_failures=args.allow_partial_failures
        )

        # Process input
        blast_runner.process_input(args.input)

    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
