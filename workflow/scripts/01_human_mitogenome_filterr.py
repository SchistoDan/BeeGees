#!/usr/bin/env python3
"""
Human Mitogenome Mapping Filter - Memory Efficient Version
Removes sequences that map to human mitochondrial or nuclear genome using minimap2, bwa-mem, or bwa-aln

Key improvements:
- Incremental CSV writing (no result accumulation in memory)
- Live logging with flush to ensure real-time output
- Memory-efficient processing
- Optional output of removed sequences (mapped to human)
"""

import os
import sys
import csv
import argparse
import subprocess
import tempfile
import shutil
import gzip
import glob
from pathlib import Path
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Set, Optional
from datetime import datetime

def log_message(message: str, log_file=None, stdout=False):
    """Log message to file and optionally stdout"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_msg = f"[{timestamp}] {message}"
    
    if log_file:
        log_file.write(formatted_msg + '\n')
        log_file.flush()
    
    if stdout:
        print(formatted_msg, flush=True)

def check_aligner(aligner):
    """Check if the specified aligner is available"""
    try:
        if aligner == 'minimap2':
            result = subprocess.run(['minimap2', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            return result.returncode == 0
        elif aligner in ['bwa-mem', 'bwa-aln']:
            result = subprocess.run(['bwa'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            return result.returncode == 0 or result.returncode == 1  # bwa returns 1 when run without args
        else:
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def build_bwa_index(reference_fasta: str) -> bool:
    """
    Build BWA index for reference genome if it doesn't exist
    
    Returns:
        True if index exists or was built successfully, False otherwise
    """
    # Check if index files already exist
    index_extensions = ['.amb', '.ann', '.bwt', '.pac', '.sa']
    index_exists = all(os.path.exists(reference_fasta + ext) for ext in index_extensions)
    
    if index_exists:
        return True
    
    print(f"Building BWA index for {reference_fasta}...", flush=True)
    try:
        result = subprocess.run(
            ['bwa', 'index', reference_fasta],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for indexing
        )
        
        if result.returncode != 0:
            print(f"Error building BWA index: {result.stderr}", file=sys.stderr, flush=True)
            return False
        
        print(f"BWA index built successfully", flush=True)
        return True
        
    except subprocess.TimeoutExpired:
        print(f"Error: BWA indexing timed out", file=sys.stderr, flush=True)
        return False
    except Exception as e:
        print(f"Error building BWA index: {str(e)}", file=sys.stderr, flush=True)
        return False

def find_fastq_files(input_dir: str) -> List[str]:
    """
    Find concatenated or merged FASTQ files in the input directory
    
    Looks for files matching:
    - *_concat_trimmed.fastq[.gz]
    - *_merged.fastq[.gz]
    - *_concat_trimmed.fq[.gz]
    - *_merged.fq[.gz]
    
    Returns:
        List of file paths
    """
    patterns = [
        '*_concat_trimmed.fastq.gz',
        '*_concat_trimmed.fastq',
        '*_concat_trimmed.fq.gz',
        '*_concat_trimmed.fq',
        '*_merged.fastq.gz',
        '*_merged.fastq',
        '*_merged.fq.gz',
        '*_merged.fq',
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(input_dir, '**', pattern), recursive=True))
    
    return sorted(list(set(files)))

def is_gzipped(file_path: str) -> bool:
    """Check if a file is gzipped"""
    return file_path.endswith('.gz')

def open_fastq(file_path: str, mode: str = 'rt'):
    """Open FASTQ file, handling both gzipped and uncompressed"""
    if is_gzipped(file_path):
        return gzip.open(file_path, mode)
    else:
        return open(file_path, mode)

def degap_sequences(records: List[SeqRecord]) -> List[SeqRecord]:
    """Remove gap characters from sequences"""
    degapped_records = []
    for record in records:
        degapped_seq = str(record.seq).replace('-', '').replace('.', '')
        if len(degapped_seq) > 0:
            new_record = SeqRecord(
                Seq(degapped_seq),
                id=record.id,
                description=record.description
            )
            degapped_records.append(new_record)
    return degapped_records

def run_aligner(query_file: str, reference_fasta: str, aligner: str = 'minimap2', 
                threads: int = 1, is_fastq: bool = False) -> Set[str]:
    """
    Run alignment tool and return set of query sequence IDs that mapped.
    Uses streaming to avoid loading entire SAM output into memory.
    
    Args:
        query_file: Path to query sequences (FASTA or FASTQ)
        reference_fasta: Path to reference genome
        aligner: Alignment tool to use ('minimap2', 'bwa-mem', or 'bwa-aln')
        threads: Number of threads for aligner
        is_fastq: Whether input is FASTQ format (vs FASTA)
    
    Returns:
        Set of sequence IDs that mapped to reference
    """
    mapped_ids = set()
    
    try:
        if aligner == 'minimap2':
            # Run minimap2 with short read preset
            cmd = [
                'minimap2',
                '-x', 'sr',
                '-a',
                '-t', str(threads),
                reference_fasta,
                query_file
            ]
            
            # Stream output line by line instead of loading all into memory
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Parse SAM output line by line as it comes
            for line in process.stdout:
                if line.startswith('@'):
                    continue
                if not line.strip():
                    continue
                
                fields = line.split('\t')
                if len(fields) < 3:
                    continue
                
                query_id = fields[0]
                flag = int(fields[1])
                ref_name = fields[2]
                
                # Check if read is mapped (flag bit 0x4 is NOT set)
                is_unmapped = flag & 4
                
                if not is_unmapped and ref_name != '*':
                    mapped_ids.add(query_id)
            
            # Wait for process to complete
            process.wait(timeout=300)
            
            if process.returncode != 0:
                stderr = process.stderr.read()
                print(f"Warning: minimap2 returned non-zero exit code: {process.returncode}", file=sys.stderr, flush=True)
                if stderr:
                    print(f"minimap2 stderr: {stderr}", file=sys.stderr, flush=True)
        
        elif aligner == 'bwa-mem':
            # Run bwa mem
            cmd = [
                'bwa', 'mem',
                '-t', str(threads),
                reference_fasta,
                query_file
            ]
            
            # Stream output line by line
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Parse SAM output line by line
            for line in process.stdout:
                if line.startswith('@'):
                    continue
                if not line.strip():
                    continue
                
                fields = line.split('\t')
                if len(fields) < 3:
                    continue
                
                query_id = fields[0]
                flag = int(fields[1])
                ref_name = fields[2]
                
                is_unmapped = flag & 4
                
                if not is_unmapped and ref_name != '*':
                    mapped_ids.add(query_id)
            
            process.wait(timeout=300)
            
            if process.returncode != 0:
                stderr = process.stderr.read()
                print(f"Warning: bwa mem returned non-zero exit code: {process.returncode}", file=sys.stderr, flush=True)
                if stderr:
                    print(f"bwa mem stderr: {stderr}", file=sys.stderr, flush=True)
        
        elif aligner == 'bwa-aln':
            # Run bwa aln (two-step process: aln -> samse)
            # Step 1: bwa aln
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sai', delete=False) as sai_file:
                sai_path = sai_file.name
            
            try:
                aln_cmd = [
                    'bwa', 'aln',
                    '-t', str(threads),
                    '-n', '0.01',
                    '-l', '1024',
                    '-o', '2',
                    reference_fasta,
                    query_file
                ]
                
                with open(sai_path, 'w') as sai_out:
                    aln_result = subprocess.run(
                        aln_cmd,
                        stdout=sai_out,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=300
                    )
                
                if aln_result.returncode != 0:
                    print(f"Warning: bwa aln returned non-zero exit code: {aln_result.returncode}", file=sys.stderr, flush=True)
                    if aln_result.stderr:
                        print(f"bwa aln stderr: {aln_result.stderr}", file=sys.stderr, flush=True)
                
                # Step 2: bwa samse - stream output
                samse_cmd = [
                    'bwa', 'samse',
                    reference_fasta,
                    sai_path,
                    query_file
                ]
                
                process = subprocess.Popen(
                    samse_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                # Parse SAM output line by line
                for line in process.stdout:
                    if line.startswith('@'):
                        continue
                    if not line.strip():
                        continue
                    
                    fields = line.split('\t')
                    if len(fields) < 3:
                        continue
                    
                    query_id = fields[0]
                    flag = int(fields[1])
                    ref_name = fields[2]
                    
                    is_unmapped = flag & 4
                    
                    if not is_unmapped and ref_name != '*':
                        mapped_ids.add(query_id)
                
                process.wait(timeout=300)
                
                if process.returncode != 0:
                    stderr = process.stderr.read()
                    print(f"Warning: bwa samse returned non-zero exit code: {process.returncode}", file=sys.stderr, flush=True)
                    if stderr:
                        print(f"bwa samse stderr: {stderr}", file=sys.stderr, flush=True)
            
            finally:
                # Clean up SAI file
                if os.path.exists(sai_path):
                    os.unlink(sai_path)
        
    except subprocess.TimeoutExpired:
        print(f"Warning: {aligner} timed out for {query_file}", file=sys.stderr, flush=True)
        if 'process' in locals():
            process.kill()
    except Exception as e:
        print(f"Error running {aligner}: {str(e)}", file=sys.stderr, flush=True)
    
    return mapped_ids

def process_fasta_file(file_path: str, reference_genome: str, output_dir: str, 
                       aligner: str = 'minimap2', threads_per_file: int = 1,
                       save_removed: bool = False, removed_dir: str = None) -> Dict:
    """Process a single FASTA alignment file for human genome mapping and filtering"""
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if '_align' in base_name:
            base_name = base_name.replace('_align', '')
        
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            return {
                'file_path': file_path,
                'base_name': base_name,
                'status': 'skipped',
                'reason': 'empty_file',
                'input_count': 0,
                'kept_count': 0,
                'removed_count': 0,
                'mapped_count': 0
            }
        
        # Read sequences
        try:
            records = list(SeqIO.parse(file_path, "fasta"))
        except Exception as e:
            return {
                'file_path': file_path,
                'base_name': base_name,
                'status': 'error',
                'reason': f'parse_error: {str(e)}',
                'input_count': 0,
                'kept_count': 0,
                'removed_count': 0,
                'mapped_count': 0
            }
        
        if not records:
            return {
                'file_path': file_path,
                'base_name': base_name,
                'status': 'skipped',
                'reason': 'no_sequences',
                'input_count': 0,
                'kept_count': 0,
                'removed_count': 0,
                'mapped_count': 0
            }
        
        input_count = len(records)
        
        # Degap sequences
        degapped_records = degap_sequences(records)
        
        if not degapped_records:
            return {
                'file_path': file_path,
                'base_name': base_name,
                'status': 'skipped',
                'reason': 'all_sequences_empty_after_degapping',
                'input_count': input_count,
                'kept_count': 0,
                'removed_count': input_count,
                'mapped_count': 0
            }
        
        # Create temporary file for degapped sequences
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as temp_fasta:
            temp_fasta_path = temp_fasta.name
            SeqIO.write(degapped_records, temp_fasta, "fasta")
        
        try:
            # Run aligner to find mapped sequences
            mapped_ids = run_aligner(temp_fasta_path, reference_genome, aligner, threads_per_file, is_fastq=False)
            
            # Filter sequences - keep only those that did NOT map
            kept_records = []
            removed_records = []
            mapped_count = 0
            
            for record in records:
                sequence_no_gaps = str(record.seq).replace('-', '').replace('.', '')
                
                if len(sequence_no_gaps) == 0:
                    # Empty sequence after degapping - remove
                    removed_records.append(record)
                elif record.id in mapped_ids:
                    # Sequence mapped to human genome - remove
                    removed_records.append(record)
                    mapped_count += 1
                else:
                    # Sequence did not map - keep it
                    kept_records.append(record)
            
            removed_count = len(removed_records)
            
            # Write filtered sequences (with original gaps/alignment)
            output_file = None
            if kept_records:
                output_file = os.path.join(output_dir, f"{base_name}_human_filtered.fasta")
                with open(output_file, 'w') as handle:
                    SeqIO.write(kept_records, handle, "fasta")
            
            # Write removed sequences if requested
            removed_file = None
            if save_removed and removed_records and removed_dir:
                removed_file = os.path.join(removed_dir, f"{base_name}_removed.fasta")
                with open(removed_file, 'w') as handle:
                    SeqIO.write(removed_records, handle, "fasta")
            
            return {
                'file_path': file_path,
                'base_name': base_name,
                'status': 'success',
                'reason': 'processed',
                'input_count': input_count,
                'kept_count': len(kept_records),
                'removed_count': removed_count,
                'mapped_count': mapped_count,
                'output_file': output_file,
                'removed_file': removed_file
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_fasta_path):
                os.unlink(temp_fasta_path)
        
    except Exception as e:
        return {
            'file_path': file_path,
            'base_name': base_name,
            'status': 'error',
            'reason': f'processing_error: {str(e)}',
            'input_count': 0,
            'kept_count': 0,
            'removed_count': 0,
            'mapped_count': 0
        }

def process_fastq_file(file_path: str, reference_genome: str, output_dir: str, 
                       aligner: str = 'minimap2', threads_per_file: int = 1,
                       save_removed: bool = False, removed_dir: str = None) -> Dict:
    """
    Process a single FASTQ file for human genome mapping and filtering using streaming approach.
    This version does NOT load the entire file into memory for filtering.
    """
    try:
        base_name = Path(file_path).stem
        # Remove .gz extension if present
        if base_name.endswith('.gz'):
            base_name = Path(base_name).stem
        
        # Determine output file name based on input pattern
        is_gzipped_input = is_gzipped(file_path)
        if '_concat_trimmed' in base_name:
            output_base = base_name.replace('_concat_trimmed', '_concat_trimmed-human_filtered')
            removed_base = base_name.replace('_concat_trimmed', '_concat_trimmed_removed')
        elif '_merged' in base_name:
            output_base = base_name.replace('_merged', '_merged-human_filtered')
            removed_base = base_name.replace('_merged', '_merged_removed')
        else:
            output_base = base_name + '-human_filtered'
            removed_base = base_name + '_removed'
        
        # Determine file extension
        if '.fq' in file_path:
            output_ext = '.fq.gz' if is_gzipped_input else '.fq'
        else:
            output_ext = '.fastq.gz' if is_gzipped_input else '.fastq'
        
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            return {
                'file_path': file_path,
                'base_name': base_name,
                'status': 'skipped',
                'reason': 'empty_file',
                'input_count': 0,
                'kept_count': 0,
                'removed_count': 0,
                'mapped_count': 0
            }
        
        # Run aligner directly on FASTQ file to get mapped IDs
        mapped_ids = run_aligner(file_path, reference_genome, aligner, threads_per_file, is_fastq=True)
        
        # Now stream through the file and filter based on mapped_ids
        input_count = 0
        kept_count = 0
        mapped_count = 0
        
        output_file = os.path.join(output_dir, output_base + output_ext)
        removed_file = None
        if save_removed and removed_dir:
            removed_file = os.path.join(removed_dir, removed_base + output_ext)
        
        try:
            # Open output file for writing
            if is_gzipped_input:
                output_handle = gzip.open(output_file, 'wt')
                removed_handle = gzip.open(removed_file, 'wt') if removed_file else None
            else:
                output_handle = open(output_file, 'w')
                removed_handle = open(removed_file, 'w') if removed_file else None
            
            try:
                # Stream through input file, writing kept sequences directly to output
                with open_fastq(file_path, 'rt') as input_handle:
                    for record in SeqIO.parse(input_handle, "fastq"):
                        input_count += 1
                        
                        if record.id in mapped_ids:
                            # Sequence mapped to human genome - remove it
                            mapped_count += 1
                            # Write to removed file if requested
                            if removed_handle:
                                SeqIO.write(record, removed_handle, "fastq")
                        else:
                            # Sequence did not map - write directly to output
                            SeqIO.write(record, output_handle, "fastq")
                            kept_count += 1
                        
                        # Clear record from memory after processing
                        del record
                
            finally:
                output_handle.close()
                if removed_handle:
                    removed_handle.close()
            
            removed_count = input_count - kept_count
            
            # Check if any sequences were kept
            if kept_count == 0:
                # Remove empty output file
                if os.path.exists(output_file):
                    os.unlink(output_file)
                output_file = None
            
            # Check if any sequences were removed
            if save_removed and removed_count == 0:
                # Remove empty removed file
                if removed_file and os.path.exists(removed_file):
                    os.unlink(removed_file)
                removed_file = None
            
        except Exception as e:
            # Clean up output files on error
            if os.path.exists(output_file):
                os.unlink(output_file)
            if removed_file and os.path.exists(removed_file):
                os.unlink(removed_file)
            raise e
        
        # Sanity check
        if input_count == 0:
            return {
                'file_path': file_path,
                'base_name': base_name,
                'status': 'skipped',
                'reason': 'no_sequences',
                'input_count': 0,
                'kept_count': 0,
                'removed_count': 0,
                'mapped_count': 0
            }
        
        return {
            'file_path': file_path,
            'base_name': base_name,
            'status': 'success',
            'reason': 'processed',
            'input_count': input_count,
            'kept_count': kept_count,
            'removed_count': removed_count,
            'mapped_count': mapped_count,
            'output_file': output_file,
            'removed_file': removed_file
        }
        
    except Exception as e:
        return {
            'file_path': file_path,
            'base_name': base_name,
            'status': 'error',
            'reason': f'processing_error: {str(e)}',
            'input_count': 0,
            'kept_count': 0,
            'removed_count': 0,
            'mapped_count': 0
        }

def main():
    # Record start time
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(
        description='Filter sequences that map to human genome using minimap2, bwa-mem, or bwa-aln',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process FASTA alignment files
  %(prog)s --input-log aligned_files.txt --human-genome human_mt.fasta \\
           --output-dir filtered/ --filtered-files-list filtered_files.txt \\
           --metrics-csv metrics.csv --threads 4

  # Process raw FASTQ reads
  %(prog)s --input-reads reads_dir/ --human-genome human_mt.fasta \\
           --output-dir filtered/ --filtered-files-list filtered_files.txt \\
           --metrics-csv metrics.csv --threads 4 --aligner bwa-mem

  # Process and save removed sequences
  %(prog)s --input-reads reads_dir/ --human-genome human_mt.fasta \\
           --output-dir filtered/ --filtered-files-list filtered_files.txt \\
           --metrics-csv metrics.csv --threads 4 --removed

Requirements:
  - minimap2, bwa, or both must be installed and available in PATH (depending on --aligner choice)
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-log', 
                            help='File containing list of FASTA alignment files (one per line)')
    input_group.add_argument('--input-reads', 
                            help='Directory containing FASTQ read files (*_concat_trimmed.fastq[.gz] or *_merged.fastq[.gz])')
    
    parser.add_argument('--human-genome', required=True, 
                       help='FASTA file containing human mitochondrial/nuclear genome sequences')
    parser.add_argument('--output-dir', required=True, 
                       help='Output directory for filtered files')
    parser.add_argument('--filtered-files-list', required=True, 
                       help='Output file listing successfully filtered files')
    parser.add_argument('--metrics-csv', required=True, 
                       help='Output CSV file with filtering metrics')
    parser.add_argument('--aligner', default='bwa-aln', 
                       choices=['minimap2', 'bwa-mem', 'bwa-aln'],
                       help='Alignment tool to use (default: bwa-aln)')
    parser.add_argument('--threads', type=int, default=1, 
                       help='Number of parallel files to process (default: 1)')
    parser.add_argument('--threads-per-file', type=int, default=4,
                       help='Number of threads for aligner per file (default: 4)')
    parser.add_argument('--removed', action='store_true',
                       help='Output removed sequences (mapped to human) to removed/ subdirectory')
    
    args = parser.parse_args()
    
    # Check if aligner is available
    if not check_aligner(args.aligner):
        print(f"ERROR: {args.aligner} is not installed or not in PATH", file=sys.stderr, flush=True)
        if args.aligner == 'minimap2':
            print("Please install minimap2: https://github.com/lh3/minimap2", file=sys.stderr, flush=True)
        else:
            print("Please install bwa: https://github.com/lh3/bwa", file=sys.stderr, flush=True)
        sys.exit(1)
    
    # Check if human genome file exists
    if not os.path.exists(args.human_genome):
        print(f"ERROR: Human genome file not found: {args.human_genome}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    # Build BWA index if using BWA and index doesn't exist
    if args.aligner in ['bwa-mem', 'bwa-aln']:
        if not build_bwa_index(args.human_genome):
            print(f"ERROR: Failed to build BWA index for {args.human_genome}", file=sys.stderr, flush=True)
            sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create removed directory if requested
    removed_dir = None
    if args.removed:
        removed_dir = os.path.join(args.output_dir, 'removed')
        os.makedirs(removed_dir, exist_ok=True)
        print(f"Removed sequences will be saved to: {removed_dir}", flush=True)
    
    # Determine input mode and get input files
    input_files = []
    is_fastq_mode = False
    
    if args.input_log:
        # FASTA mode - read from log file
        with open(args.input_log, 'r') as f:
            input_files = [line.strip() for line in f if line.strip()]
        
        # Verify input files exist
        missing_files = [f for f in input_files if not os.path.exists(f)]
        if missing_files:
            print(f"WARNING: {len(missing_files)} input files not found:", file=sys.stderr, flush=True)
            for f in missing_files[:5]:  # Show first 5
                print(f"  - {f}", file=sys.stderr, flush=True)
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more", file=sys.stderr, flush=True)
        
        input_files = [f for f in input_files if os.path.exists(f)]
        is_fastq_mode = False
        
    elif args.input_reads:
        # FASTQ mode - find files in directory
        if not os.path.isdir(args.input_reads):
            print(f"ERROR: Input reads directory not found: {args.input_reads}", file=sys.stderr, flush=True)
            sys.exit(1)
        
        input_files = find_fastq_files(args.input_reads)
        
        if not input_files:
            print(f"ERROR: No FASTQ files found in {args.input_reads}", file=sys.stderr, flush=True)
            print("Looking for files matching: *_concat_trimmed.fastq[.gz] or *_merged.fastq[.gz]", file=sys.stderr, flush=True)
            sys.exit(1)
        
        is_fastq_mode = True
        print(f"Found {len(input_files)} FASTQ files in {args.input_reads}", flush=True)
    
    if not input_files:
        print("ERROR: No valid input files found", file=sys.stderr, flush=True)
        sys.exit(1)
    
    print(f"Processing {len(input_files)} files using {args.aligner}", flush=True)
    print(f"Input mode: {'FASTQ reads' if is_fastq_mode else 'FASTA alignments'}", flush=True)
    print(f"Reference genome: {args.human_genome}", flush=True)
    print(f"Parallel files: {args.threads}, Threads per file: {args.threads_per_file}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Open CSV file for incremental writing (summary rows only)
    csv_file = open(args.metrics_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['file_path', 'sequence_id', 'removal_reason', 
                        'mapped_to_human', 'step_name', 'input_count', 'kept_count', 'removed_count'])
    csv_file.flush()
    
    # Open filtered files list for incremental writing
    filtered_list_file = open(args.filtered_files_list, 'w')
    
    # Track summary statistics without storing full results
    summary_stats = {
        'total_input': 0,
        'total_kept': 0,
        'total_removed': 0,
        'successful_count': 0,
        'skipped_count': 0,
        'error_count': 0,
        'mapped_count': 0,
        'successful_files': [],
        'removed_files_count': 0
    }
    
    # Process files in parallel
    process_func = process_fastq_file if is_fastq_mode else process_fasta_file
    
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        future_to_file = {
            executor.submit(process_func, file_path, args.human_genome, 
                          args.output_dir, args.aligner, args.threads_per_file,
                          args.removed, removed_dir): file_path
            for file_path in input_files
        }
        
        completed = 0
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            completed += 1
            try:
                result = future.result()
                
                # Write summary row to CSV (no individual read rows)
                csv_writer.writerow([
                    result['file_path'],
                    result['base_name'],
                    result['reason'],
                    result.get('mapped_count', 0),
                    'human_mitogenome_filter',
                    result['input_count'],
                    result['kept_count'],
                    result['removed_count']
                ])
                
                csv_file.flush()  # Force write to disk
                
                # Update summary statistics (aggregates only, no full result storage)
                summary_stats['total_input'] += result['input_count']
                summary_stats['total_kept'] += result['kept_count']
                summary_stats['total_removed'] += result['removed_count']
                
                if result['status'] == 'success':
                    summary_stats['successful_count'] += 1
                    summary_stats['mapped_count'] += result.get('mapped_count', 0)
                    # Write to filtered files list immediately if output exists
                    if result.get('output_file') and os.path.exists(result['output_file']):
                        filtered_list_file.write(result['output_file'] + '\n')
                        filtered_list_file.flush()
                        summary_stats['successful_files'].append(result['output_file'])
                    
                    # Track removed files
                    if result.get('removed_file') and os.path.exists(result['removed_file']):
                        summary_stats['removed_files_count'] += 1
                    
                    print(f"[{completed}/{len(input_files)}] ✓ {result['base_name']}: "
                          f"{result['kept_count']}/{result['input_count']} sequences kept, "
                          f"{result['removed_count']} mapped to human genome", flush=True)
                
                elif result['status'] == 'skipped':
                    summary_stats['skipped_count'] += 1
                    print(f"[{completed}/{len(input_files)}] ⚠ {result['base_name']}: "
                          f"skipped ({result['reason']})", flush=True)
                
                else:
                    summary_stats['error_count'] += 1
                    print(f"[{completed}/{len(input_files)}] ✗ {result['base_name']}: "
                          f"error ({result['reason']})", flush=True)
                
                # Clear result from memory (Python will garbage collect)
                del result
                    
            except Exception as e:
                summary_stats['error_count'] += 1
                print(f"[{completed}/{len(input_files)}] ✗ {file_path}: "
                      f"processing failed - {str(e)}", flush=True)
                
                # Write error to CSV
                base_name = os.path.basename(file_path)
                csv_writer.writerow([
                    file_path,
                    base_name,
                    f'executor_error: {str(e)}',
                    0,
                    'human_mitogenome_filter',
                    0, 0, 0
                ])
                csv_file.flush()
    
    # Close files
    csv_file.close()
    filtered_list_file.close()
    
    # Calculate run time
    end_time = datetime.now()
    run_time = end_time - start_time
    
    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"Human Mitogenome Mapping Filter Summary:", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Total run time: {run_time}", flush=True)
    print(f"Input mode: {'FASTQ reads' if is_fastq_mode else 'FASTA alignments'}", flush=True)
    print(f"Aligner: {args.aligner}", flush=True)
    print(f"Files processed successfully: {summary_stats['successful_count']}/{len(input_files)}", flush=True)
    print(f"Files skipped: {summary_stats['skipped_count']}", flush=True)
    print(f"Files with errors: {summary_stats['error_count']}", flush=True)
    print(f"Total input sequences: {summary_stats['total_input']}", flush=True)
    print(f"Sequences kept (not mapped): {summary_stats['total_kept']}", flush=True)
    print(f"Sequences removed (mapped to human): {summary_stats['mapped_count']}", flush=True)
    print(f"Total removed: {summary_stats['total_removed']}", flush=True)
    print(f"Filtered files written: {len(summary_stats['successful_files'])}", flush=True)
    if args.removed:
        print(f"Removed sequence files written: {summary_stats['removed_files_count']}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Write summary to file
    summary_file = os.path.join(args.output_dir, 'filtering_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Human Mitogenome Mapping Filter Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total run time: {run_time}\n")
        f.write(f"Input mode: {'FASTQ reads' if is_fastq_mode else 'FASTA alignments'}\n")
        f.write(f"Aligner: {args.aligner}\n")
        f.write(f"Reference genome: {args.human_genome}\n")
        f.write(f"Files processed successfully: {summary_stats['successful_count']}/{len(input_files)}\n")
        f.write(f"Files skipped: {summary_stats['skipped_count']}\n")
        f.write(f"Files with errors: {summary_stats['error_count']}\n")
        f.write(f"Total input sequences: {summary_stats['total_input']}\n")
        f.write(f"Sequences kept (not mapped): {summary_stats['total_kept']}\n")
        f.write(f"Sequences removed (mapped to human): {summary_stats['mapped_count']}\n")
        f.write(f"Total removed: {summary_stats['total_removed']}\n")
        f.write(f"Filtered files written: {len(summary_stats['successful_files'])}\n")
        if args.removed:
            f.write(f"Removed sequence files written: {summary_stats['removed_files_count']}\n")
    
    print(f"\nSummary written to: {summary_file}", flush=True)
    print(f"Metrics CSV written to: {args.metrics_csv}", flush=True)
    print(f"Filtered files list written to: {args.filtered_files_list}", flush=True)
    if args.removed:
        print(f"Removed sequences directory: {removed_dir}", flush=True)

if __name__ == "__main__":
    main()
