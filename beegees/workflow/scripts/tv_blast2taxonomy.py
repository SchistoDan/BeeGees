#!/usr/bin/env python3
"""
tv_blast2taxonomy.py

Validates taxonomic assignments by comparing BLAST hits against expected taxonomy using hierarchical 
taxonomic matching. Supports both BOLDistilled (COI) and generic semicolon-delimited taxonomy formats (e.g., for rbcL). 
Designed for metabarcoding studies requiring taxonomic validation of DNA barcode sequences.

PROCESS OVERVIEW
================
1. Parses BLAST results (CSV), expected taxonomy mappings, BLAST database taxonomy (TSV), and input sequences
2. For each sequence: applies quality filters (see below) -> finds first matching hit -> performs taxonomic validation
3. Uses hierarchical matching strategy (order/family/genus/species only, exact string matching)
4. Selects first hit with matching taxonomy after quality filtering
5. Outputs validation results and filtered sequences
6. Selects best sequence per Process_ID based on match quality metrics

FILTERING
==================
Applied in sequence to BLAST hits:
1. Minimum percent identity threshold (--min-pident)
2. Minimum alignment length threshold (--min-length) 
3. First hit with taxonomic match is selected (no top-N limit)

SELECTION CRITERIA
==================
Best sequence per Process_ID selected based on (in priority order):
1. Must have match_taxonomy == "YES"
2. Lowest matched_rank (species > genus > family > order)
3. Highest pident
4. Lowest mismatch
5. Lowest gaps
6. Lowest evalue
7. Highest length
8. Highest s value (from seq_id)
9. Highest r value (from seq_id)
10. Has "fcleaner" (from seq_id)

COMMAND LINE
======================
REQUIRED:
  --input-blast-csv FILE     # BLAST results CSV with hit columns (hit1, hit1_pident, etc.) output by tv_local_blast.py
  --input-taxonomy-file FILE # BOLD taxonomy TSV 
  --input-exp-taxonomy FILE  # Expected taxonomy CSV (Process ID + taxonomic ranks)
  --input-fasta FILE         # Multi-FASTA sequences with Process ID linkable headers
  --output-csv FILE          # Validation results CSV
  --output-fasta FILE        # Filtered FASTA with matching sequences only

OPTIONAL:
  --taxval-rank RANK         # Primary validation rank: order, family, genus, species (default: family)
  --min-pident FLOAT         # Minimum percent identity threshold (default: 0.0)
  --min-length INT           # Minimum alignment length threshold (default: 0)
  --log FILE                 # Log file path (optional)

VALIDATION STRATEGY
===================
- Restricts matching to order, family, genus, species ranks only
- Uses exact string comparison between BLAST hit taxonomy and expected lineage
- Accepts matches at ANY rank within expected lineage
- Selects first hit (by BLAST order) that matches after quality filtering
- Among matching hits, prioritizes by: percent identity → length → mismatches → e-value

OUTPUT
======
- CSV: Validation outcomes, matched ranks, hit statistics, obs_taxonomy field, gaps, full lineage, selection status
- FASTA: Only sequences with successful taxonomic matches
- obs_taxonomy: Shows top 10 hits with "sseqid: Taxonomy (rank)" format

AUTHORS
=======
Dan Parsons @NHMUK
License: MIT
Version: 2.7
"""

import argparse
import csv
import logging
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def setup_logging(log_file: str = None):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)
    
def parse_fasta(fasta_file: str, logger) -> Dict[str, str]:
    """Parse FASTA file and return dictionary of seq_id -> sequence."""
    sequences = {}
    current_seq = ""
    current_id = ""
    
    try:
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:
                        sequences[current_id] = current_seq
                    current_id = line[1:]  # Remove '>' character
                    current_seq = ""
                else:
                    current_seq += line
            
            # Don't forget the last sequence
            if current_id:
                sequences[current_id] = current_seq
                
        logger.info(f"Parsed {len(sequences)} sequences from {fasta_file}")
        return sequences
        
    except FileNotFoundError:
        logger.error(f"FASTA file not found: {fasta_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing FASTA file: {e}")
        sys.exit(1)

def parse_blast_csv(blast_file: str, logger) -> Dict[str, List[Dict]]:
    """Parse BLAST CSV file and return dictionary of seq_id -> list of ALL hits."""
    blast_data = {}
    
    try:
        with open(blast_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                seq_id = row['seq_id']
                hits = []
                
                # Extract ALL hit information (not just hit1-hit20)
                i = 1
                while True:
                    hit_col = f'hit{i}'
                    pident_col = f'hit{i}_pident'
                    length_col = f'hit{i}_length'
                    mismatch_col = f'hit{i}_mismatch'
                    evalue_col = f'hit{i}_evalue'
                    description_col = f'hit{i}_description'
                    gaps_col = f'hit{i}_gaps'
                    
                    # Check if this hit column exists and has data
                    if hit_col not in row or not row[hit_col] or row[hit_col].lower() == 'null':
                        break
                    
                    # Helper function to safely convert values
                    def safe_float(value):
                        if not value or value.lower() == 'null':
                            return 0.0
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return 0.0
                    
                    def safe_int(value):
                        if not value or value.lower() == 'null':
                            return 0
                        try:
                            return int(value)
                        except (ValueError, TypeError):
                            return 0
                    
                    hits.append({
                        'hit_id': row[hit_col],
                        'pident': safe_float(row.get(pident_col, '')),
                        'length': safe_int(row.get(length_col, '')),
                        'mismatch': safe_int(row.get(mismatch_col, '')),
                        'evalue': safe_float(row.get(evalue_col, '')),
                        'description': row.get(description_col, ''),
                        'gaps': safe_int(row.get(gaps_col, '')),
                        'hit_num': i
                    })
                    
                    i += 1
                
                blast_data[seq_id] = hits
                
        total_hits = sum(len(hits) for hits in blast_data.values())
        logger.info(f"Parsed {total_hits} BLAST hits for {len(blast_data)} sequences from {blast_file}")
        return blast_data
        
    except FileNotFoundError:
        logger.error(f"BLAST file not found: {blast_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing BLAST file: {e}")
        sys.exit(1)
        
def parse_returned_taxonomy_tsv(bold_file: str, logger) -> Dict[str, Dict]:
    """
    Parse taxonomy TSV file and return dictionary of ID -> taxonomy info.
    Auto-detects format:
    - BOLDistilled format: separate columns (bin, kingdom, phylum, etc.) (for COI)
    - Generic format: Feature ID and semicolon-delimited Taxon column (for rbcL)
    """
    taxonomy_data = {}
    
    try:
        with open(bold_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            # Auto-detect format based on columns
            fieldnames = [field.lower() for field in reader.fieldnames]
            
            # Check if this is BOLDistilled format (has separate taxonomy columns)
            is_boldistilled = 'bin' in fieldnames and 'kingdom' in fieldnames
            
            # Check if this is generic semicolon-delimited format
            is_generic = 'feature id' in fieldnames and 'taxon' in fieldnames
            
            if is_boldistilled:
                logger.info("Detected BOLDistilled format (separate taxonomy columns)")
                for row in reader:
                    bin_id = row['bin']
                    taxonomy_data[bin_id] = {
                        'kingdom': row.get('kingdom', ''),
                        'phylum': row.get('phylum', ''),
                        'class': row.get('class', ''),
                        'order': row.get('order', ''),
                        'family': row.get('family', ''),
                        'subfamily': row.get('subfamily', ''),
                        'tribe': row.get('tribe', ''),
                        'genus': row.get('genus', ''),
                        'species': row.get('species', ''),
                        'subspecies': row.get('subspecies', '')
                    }
                    
            elif is_generic:
                logger.info("Detected generic format (semicolon-delimited taxonomy)")
                
                # Get original case column names
                feature_id_col = None
                taxon_col = None
                for field in reader.fieldnames:
                    if field.lower() == 'feature id':
                        feature_id_col = field
                    elif field.lower() == 'taxon':
                        taxon_col = field
                
                for row in reader:
                    feature_id = row[feature_id_col]
                    taxon_string = row[taxon_col]
                    
                    # Parse semicolon-delimited taxonomy
                    taxonomy = {
                        'kingdom': '',
                        'phylum': '',
                        'class': '',
                        'order': '',
                        'family': '',
                        'subfamily': '',
                        'tribe': '',
                        'genus': '',
                        'species': '',
                        'subspecies': ''
                    }
                    
                    for rank_entry in taxon_string.split(';'):
                        rank_entry = rank_entry.strip()
                        if '__' in rank_entry:
                            prefix, value = rank_entry.split('__', 1)
                            
                            # Map prefixes to rank names
                            rank_map = {
                                'k': 'kingdom',
                                'p': 'phylum',
                                'c': 'class',
                                'o': 'order',
                                'f': 'family',
                                'g': 'genus',
                                's': 'species'
                            }
                            
                            if prefix in rank_map:
                                taxonomy[rank_map[prefix]] = value
                    
                    taxonomy_data[feature_id] = taxonomy
            
            else:
                logger.error("Unable to detect taxonomy file format")
                logger.error(f"Expected either BOLDistilled format (bin, kingdom, phylum, ...) "
                           f"or generic format (Feature ID, Taxon)")
                logger.error(f"Found columns: {', '.join(reader.fieldnames)}")
                sys.exit(1)
                
        logger.info(f"Parsed taxonomy data for {len(taxonomy_data)} entries from {bold_file}")
        return taxonomy_data
        
    except FileNotFoundError:
        logger.error(f"Taxonomy file not found: {bold_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing taxonomy file: {e}")
        sys.exit(1)

def parse_input_taxonomy_csv(taxonomy_data: str, logger) -> Dict[str, Dict]:
    """Parse taxonomy CSV file and return dictionary of Process ID -> taxonomy info."""
    exp_taxonomy = {}
    
    try:
        with open(taxonomy_data, 'r') as f:
            reader = csv.DictReader(f)
            
            # Normalize column names to lowercase for case-insensitive matching
            normalized_fieldnames = {field.lower(): field for field in reader.fieldnames}
            
            # Check for Process ID column (accept either 'Process ID' or 'ID')
            process_id_col = None
            if 'process id' in normalized_fieldnames:
                process_id_col = normalized_fieldnames['process id']
            elif 'id' in normalized_fieldnames:
                process_id_col = normalized_fieldnames['id']
            else:
                logger.error("Taxonomy file must have either 'Process ID' or 'ID' column")
                sys.exit(1)
            
            logger.info(f"Using '{process_id_col}' column as Process ID")
            
            for row in reader:
                process_id = row[process_id_col]
                
                # Helper function to get value with case-insensitive key lookup
                def get_taxonomy_value(rank_name):
                    rank_lower = rank_name.lower()
                    if rank_lower in normalized_fieldnames:
                        return row.get(normalized_fieldnames[rank_lower], '')
                    return ''
                
                exp_taxonomy[process_id] = {
                    'phylum': get_taxonomy_value('phylum'),
                    'class': get_taxonomy_value('class'),
                    'order': get_taxonomy_value('order'),
                    'family': get_taxonomy_value('family'),
                    'genus': get_taxonomy_value('genus'),
                    'species': get_taxonomy_value('species')
                }
                
        logger.info(f"Parsed expected taxonomy for {len(exp_taxonomy)} Process IDs from {taxonomy_data}")
        return exp_taxonomy
        
    except FileNotFoundError:
        logger.error(f"Taxonomy file not found: {taxonomy_data}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing taxonomy file: {e}")
        sys.exit(1)

def build_taxonomy_lineage(exp_taxonomy: Dict[str, str]) -> str:
    """Build full taxonomic lineage string in format: phylum;class;order;family;genus;species"""
    ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    lineage_parts = []
    
    for rank in ranks:
        value = exp_taxonomy.get(rank, '').strip()
        lineage_parts.append(value)
    
    return ';'.join(lineage_parts)
    
def extract_hit_id(hit_id: str) -> Optional[str]:
    """
    Extract Feature ID from hit ID.
    Handles both BOLDistilled format (piped: "something|BOLD:AAA0001") 
    and GenBank format (direct accession: "AB000317.1").
    """
    # For BOLDistilled format: extract after the last pipe
    if '|' in hit_id:
        return hit_id.split('|')[-1]
    # For GenBank/generic format: use the ID directly
    return hit_id

def find_process_id(seq_id: str, exp_taxonomy: Dict[str, Dict]) -> Optional[str]:
    """Find matching Process ID by partial match of seq_id."""
    # Try exact match first
    if seq_id in exp_taxonomy:
        return seq_id
    
    # Try partial matches - look for Process IDs that are substrings of seq_id
    for process_id in exp_taxonomy.keys():
        if process_id in seq_id:
            return process_id
    
    return None

def get_expected_lineage(expected_exp_taxonomy: Dict[str, str]) -> Dict[str, str]:
    """Extract the complete expected taxonomic lineage (restricted to allowed ranks: order, family, genus, species only)."""
    return {
        'order': expected_exp_taxonomy.get('order', '').strip(),
        'family': expected_exp_taxonomy.get('family', '').strip(),
        'genus': expected_exp_taxonomy.get('genus', '').strip(),
        'species': expected_exp_taxonomy.get('species', '').strip()
    }

def find_expected_taxonomy(expected_exp_taxonomy: Dict, taxval_rank: str, logger, seq_id: str = None, process_id: str = None) -> Tuple[str, str]:
    """Find expected taxonomy, traversing up ranks if necessary (restricted to order, family, genus, species only)."""
    # Define taxonomic hierarchy (from most specific to least specific)
    rank_hierarchy = ['species', 'genus', 'family', 'order']
    
    # Find the starting position in the hierarchy
    try:
        start_idx = rank_hierarchy.index(taxval_rank)
    except ValueError:
        logger.error(f"Invalid taxonomic rank: {taxval_rank}")
        return "", ""
    
    # Try to find taxonomy at the specified rank and progressively higher ranks
    for current_rank in rank_hierarchy[start_idx:]:
        expected_rank_value = expected_exp_taxonomy.get(current_rank, '').strip()
        
        if expected_rank_value:
            if current_rank != taxval_rank:
                logger.info(f"Expected taxonomy found at higher rank '{current_rank}' for seq_id: {seq_id}, Process ID: {process_id}")
            return expected_rank_value, current_rank
    
    # No taxonomy found at any rank
    logger.warning(f"No expected taxonomy found at any rank for seq_id: {seq_id}, Process ID: {process_id}")
    return "", ""

def check_hit_matches_lineage(hit_taxonomy: Dict[str, str], expected_lineage: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Check if a hit's taxonomy matches the expected lineage at any rank.
    Returns (matched_taxonomy, matched_rank) or (None, None)
    Uses exact string matching only at order, family, genus, species ranks.
    """
    # Define rank hierarchy (most specific to least specific)
    rank_hierarchy = ['species', 'genus', 'family', 'order']
    
    # Check each rank in observed taxonomy against expected lineage
    for rank in rank_hierarchy:
        observed_at_rank = hit_taxonomy.get(rank, '').strip()
        expected_at_rank = expected_lineage.get(rank, '').strip()
        
        if observed_at_rank and expected_at_rank:
            # Exact string match only
            if observed_at_rank == expected_at_rank:
                return observed_at_rank, rank
    
    return None, None
    
def find_first_matching_hit(hits: List[Dict], taxonomy_data: Dict[str, Dict], expected_lineage: Dict[str, str], min_pident: float, min_length: int, logger) -> Optional[Dict]:
    """
    Find the first hit (by BLAST order) that matches the expected lineage after quality filtering.
    Returns the matching hit with taxonomy info added, or None if no matches found.
    """
    identity_filtered_count = 0
    length_filtered_count = 0
    
    logger.info(f"Searching through {len(hits)} hits for first taxonomic match")
    
    # Iterate through hits in BLAST order
    for hit in hits:
        # Apply identity filter
        if hit['pident'] < min_pident:
            identity_filtered_count += 1
            continue
        
        # Apply length filter
        if hit['length'] < min_length:
            length_filtered_count += 1
            continue
        
        # Extract feature/sequence ID
        feature_id = extract_hit_id(hit['hit_id'])
        if not feature_id:
            logger.debug(f"Could not extract feature ID from {hit['hit_id']}")
            continue

        if feature_id not in taxonomy_data:
            logger.warning(f"Feature ID {feature_id} not found in taxonomy database")
            continue

        hit_taxonomy = taxonomy_data[feature_id]
        
        # Check if this hit matches the expected lineage
        matched_taxonomy, matched_rank = check_hit_matches_lineage(hit_taxonomy, expected_lineage)
        
        if matched_taxonomy:
            # Found a match! Add taxonomy info and return
            hit_with_taxonomy = hit.copy()
            hit_with_taxonomy['taxonomy_match'] = matched_taxonomy
            hit_with_taxonomy['matched_rank'] = matched_rank
            hit_with_taxonomy['feature_id'] = feature_id  # previously 'feature_id'
            
            logger.info(f"Found first matching hit: {hit['hit_id']} (hit #{hit['hit_num']})")
            logger.info(f"  Match: {matched_taxonomy} at {matched_rank} rank")
            logger.info(f"  Quality: {hit['pident']}% identity, {hit['length']}bp length, {hit['gaps']} gaps")
            
            if identity_filtered_count > 0:
                logger.info(f"  Filtered {identity_filtered_count} hits below {min_pident}% identity before finding match")
            if length_filtered_count > 0:
                logger.info(f"  Filtered {length_filtered_count} hits below {min_length}bp length before finding match")
            
            return hit_with_taxonomy
    
    # No matching hits found
    logger.info(f"No taxonomic matches found after checking all hits")
    if identity_filtered_count > 0:
        logger.info(f"  Filtered out {identity_filtered_count} hits below {min_pident}% identity threshold")
    if length_filtered_count > 0:
        logger.info(f"  Filtered out {length_filtered_count} hits below {min_length}bp length threshold")
    
    return None

def sort_hits_by_quality(hits: List[Dict]) -> List[Dict]:
    """Sort hits by quality (percent identity, length, mismatch, e-value)."""
    def hit_quality_key(hit):
        return (
            hit['pident'],           # Higher is better
            hit['length'],           # Higher is better  
            -hit['mismatch'],        # Lower is better (negative for sorting)
            -hit['evalue']           # Lower is better (negative for sorting)
        )
    
    return sorted(hits, key=hit_quality_key, reverse=True)
    
def get_top_10_taxonomies(hits: List[Dict], taxonomy_data: Dict[str, Dict], min_pident: float, min_length: int, logger) -> str:
    """
    Extract taxonomies from top 10 hits (after quality filtering and sorting) and format for obs_taxonomy field.
    Format: "sseqid: Taxonomy (rank); sseqid: Taxonomy (rank); ..."
    """
    # Define allowed ranks
    allowed_ranks = ['species', 'genus', 'family', 'order']
    
    # Apply quality filters
    filtered_hits = []
    for hit in hits:
        if hit['pident'] >= min_pident and hit['length'] >= min_length:
            filtered_hits.append(hit)
    
    # Sort by quality
    sorted_hits = sort_hits_by_quality(filtered_hits)
    
    # Limit to top 10
    top_hits = sorted_hits[:10]
    
    obs_taxonomy_formatted = []
    
    for hit in top_hits:
        feature_id = extract_hit_id(hit['hit_id'])
        if not feature_id:
            continue
    
        if feature_id not in taxonomy_data:
            continue
    
        hit_taxonomy = taxonomy_data[feature_id]
        
        # Extract taxonomic ranks for this hit (only allowed ranks)
        # Collect all taxonomies for this hit
        hit_taxonomies = []
        for rank in allowed_ranks:
            hit_rank_value = hit_taxonomy.get(rank, '').strip()
            if hit_rank_value:
                hit_taxonomies.append(f"{hit_rank_value} ({rank})")
        
        # Format with sseqid prefix
        if hit_taxonomies:
            sseqid = hit['hit_id']
            taxonomy_string = ", ".join(hit_taxonomies)
            formatted_entry = f"{sseqid}: {taxonomy_string}"
            obs_taxonomy_formatted.append(formatted_entry)
    
    result = "; ".join(obs_taxonomy_formatted)
    logger.info(f"Generated obs_taxonomy from {len(top_hits)} hits (after filtering from {len(hits)} total hits)")
    
    return result

def extract_seq_id_values(seq_id: str) -> Tuple[Optional[float], Optional[int], bool]:
    """
    Extract r value, s value, and fcleaner status from seq_id.
    Returns: (r_value, s_value, has_fcleaner)
    """
    # Extract r value (e.g., r_1, r_1.3, r_1.5)
    r_match = re.search(r'_r_([\d.]+)', seq_id)
    r_value = float(r_match.group(1)) if r_match else None
    
    # Extract s value (e.g., s_50, s_100)
    s_match = re.search(r'_s_(\d+)', seq_id)
    s_value = int(s_match.group(1)) if s_match else None
    
    # Check for fcleaner
    has_fcleaner = 'fcleaner' in seq_id
    
    return r_value, s_value, has_fcleaner
    
def select_best_sequences(results: List[Dict], logger) -> List[Dict]:
    """
    Select the best sequence for each Process_ID based on quality criteria.
    Adds 'selected' field to each result: 'YES' for best, 'NO' for others.
    
    Selection criteria (in priority order):
    1. Must have match_taxonomy == "YES" (i.e. correct taxa at order-level or below)
    2. Lowest matched_rank (species > genus > family > order)
    3. Lowest gaps
    4. Lowest mismatch
    5. Highest pident
    6. Lowest evalue
    7. Highest length
    8. Highest s value
    9. Highest r value
    10. Has "fcleaner" in seq_id
    """
    # Group results by Process_ID
    process_groups = {}
    for result in results:
        process_id = result['Process_ID']
        if process_id not in process_groups:
            process_groups[process_id] = []
        process_groups[process_id].append(result)
    
    logger.info(f"Selecting best sequences for {len(process_groups)} Process IDs")
    
    # For each Process_ID, select the best sequence
    for process_id, group_results in process_groups.items():
        # Filter to only matching sequences
        matching_results = [r for r in group_results if r['match_taxonomy'] == 'YES']
        
        if not matching_results:
            # No matches for this Process_ID - mark all as not selected
            logger.info(f"Process ID {process_id}: None of the BLAST hits matched the expected taxonomy at any allowed rank (order, family, genus, or species), or they were filtered out")
            for result in group_results:
                result['selected'] = 'NO'
            continue
        
        # Define rank priority (lower number = better/more specific)
        rank_priority = {'species': 1, 'genus': 2, 'family': 3, 'order': 4, 'null': 999}
        
        # Sort matching results by all criteria
        def selection_key(result):
            # Extract seq_id values
            r_value, s_value, has_fcleaner = extract_seq_id_values(result['seq_id'])
            
            # Handle missing values
            r_val = r_value if r_value is not None else -1
            s_val = s_value if s_value is not None else -1
            
            return (
                rank_priority.get(result['matched_rank'], 999),  # Lower is better (more specific)
                result['gaps'],                                  # Lower is better
                result['mismatch'],                              # Lower is better
                -result['pident'],                               # Negative for higher is better
                result['evalue'],                                # Lower is better
                -result['length'],                               # Negative for higher is better
                -s_val,                                          # Negative for higher is better
                -r_val,                                          # Negative for higher is better
                not has_fcleaner                                 # False (has fcleaner) sorts before True
            )
        
        matching_results.sort(key=selection_key)
        
        # The first result after sorting is the best
        best_result = matching_results[0]
        best_result['selected'] = 'YES'
        
        logger.info(f"Process ID {process_id}: Selected {best_result['seq_id']}")
        logger.info(f"  Criteria: matched_rank={best_result['matched_rank']}, gaps={best_result['gaps']}, "
                   f"mismatch={best_result['mismatch']}, pident={best_result['pident']}, "
                   f"evalue={best_result['evalue']}, length={best_result['length']}")
        
        # Mark all others as not selected
        for result in group_results:
            if result is not best_result:
                result['selected'] = 'NO'
    
    return results

def write_output_csv(results: List[Dict], output_file: str, logger):
    """Write results to output CSV file."""
    try:
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['seq_id', 'Process_ID', 'taxval_rank', 'expected_taxonomy', 'expected_taxonomy_rank', 
                         'obs_taxonomy', 'match_taxonomy', 'matched_rank', 'top_matching_hit', 
                         'pident', 'length', 'mismatch', 'gaps', 'evalue', 'selected']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow(result)
                
        logger.info(f"Results written to {output_file}")
        
    except Exception as e:
        logger.error(f"Error writing output CSV: {e}")
        sys.exit(1)

def write_output_fasta(matching_sequences: Dict[str, str], output_file: str, logger):
    """Write matching sequences to output FASTA file."""
    try:
        with open(output_file, 'w') as f:
            for seq_id, sequence in matching_sequences.items():
                f.write(f">{seq_id}\n{sequence}\n")
                
        logger.info(f"Matching sequences written to {output_file} ({len(matching_sequences)} sequences)")
        
    except Exception as e:
        logger.error(f"Error writing output FASTA: {e}")
        sys.exit(1)
        
def main():
    parser = argparse.ArgumentParser(
        description="Validate taxonomic assignments from BLAST results using a custom database"
    )
    parser.add_argument('--input-blast-csv', required=True, 
                       help='Input BLAST results CSV file')
    parser.add_argument('--input-taxonomy-file', required=True,
                       help='Input taxonomy TSV file (BOLDistilled format or generic Feature ID/Taxon format)')
    parser.add_argument('--input-exp-taxonomy', required=True,
                       help='Input expected taxonomy CSV file')
    parser.add_argument('--input-fasta', required=True,
                       help='Input multi-FASTA file with sequences')
    parser.add_argument('--output-csv', required=True,
                       help='Output results CSV file')
    parser.add_argument('--output-fasta', required=True,
                       help='Output FASTA file with matching sequences')
    parser.add_argument('--taxval-rank', default='family',
                       choices=['order', 'family', 'genus', 'species'],
                       help='Taxonomic rank to validate at (default: family)')
    parser.add_argument('--min-pident', type=float, default=0.0,
                       help='Minimum percent identity threshold for hits to be considered (default: 0.0, no filtering)')
    parser.add_argument('--min-length', type=int, default=0,
                       help='Minimum alignment length threshold for hits to be considered (default: 0, no filtering)')
    parser.add_argument('--log', 
                       help='Log file path (optional, defaults to stdout only)')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log)
    
    logger.info("Starting BLAST taxonomy validation using BOLD database")
    logger.info(f"Validation rank: {args.taxval_rank}")
    logger.info(f"Minimum percent identity: {args.min_pident}%")
    logger.info(f"Minimum alignment length: {args.min_length}bp")
    logger.info("Rank restrictions: Only matches at order, family, genus, and species levels allowed")
    logger.info("Hit selection: First hit with matching taxonomy (by BLAST order) after quality filtering")
    logger.info("obs_taxonomy output: Top 10 hits by quality with sseqid information")
    logger.info("Sequence selection: Best sequence per Process_ID based on quality criteria")
    logger.info(f"Taxonomy database: {args.input_taxonomy_file}")
    logger.info(f"BLAST input (CSV file): {args.input_blast_csv}")
    
    if args.log:
        logger.info(f"Log file: {args.log}")
    
    # Parse input files
    sequences = parse_fasta(args.input_fasta, logger)
    blast_data = parse_blast_csv(args.input_blast_csv, logger)
    taxonomy_data = parse_returned_taxonomy_tsv(args.input_taxonomy_file, logger)
    exp_taxonomy = parse_input_taxonomy_csv(args.input_exp_taxonomy, logger)
    
    results = []
    matching_sequences = {}
    
    # Process each sequence
    for seq_id, hits in blast_data.items():
        logger.info(f"Processing sequence: {seq_id} ({len(hits)} hits)")
        
        # Find corresponding Process ID
        process_id = find_process_id(seq_id, exp_taxonomy)
        if not process_id:
            logger.error(f"No matching Process ID found for sequence: {seq_id}")
            sys.exit(1)
        
        # Get complete expected taxonomic lineage (restricted to allowed ranks)
        expected_lineage = get_expected_lineage(exp_taxonomy[process_id])
        logger.info(f"Expected lineage (restricted): {expected_lineage}")
        
        # Build full taxonomy lineage for output
        full_lineage = build_taxonomy_lineage(exp_taxonomy[process_id])
        
        # Find expected taxonomy at specified rank (with traversal if needed)
        expected_taxonomy, expected_taxonomy_rank = find_expected_taxonomy(
            exp_taxonomy[process_id], args.taxval_rank, logger, seq_id, process_id)
        
        # Find first matching hit after quality filtering
        matching_hit = find_first_matching_hit(
            hits, taxonomy_data, expected_lineage, args.min_pident, args.min_length, logger)
        
        # Generate obs_taxonomy string from top 10 hits by quality
        obs_taxonomy_str = get_top_10_taxonomies(hits, taxonomy_data, args.min_pident, args.min_length, logger)
        
        # Determine if there's a match
        if matching_hit:
            match_taxonomy = "YES"
            matched_rank = matching_hit.get('matched_rank', 'unknown')
            matched_taxonomy = matching_hit.get('taxonomy_match', 'unknown')
            best_hit = matching_hit
            logger.info(f"Match found: {matched_taxonomy} at {matched_rank} rank")
        else:
            match_taxonomy = "NO"
            matched_rank = "null"
            # For non-matching cases, show the top overall hit after filtering for reference
            filtered_hits = [h for h in hits if h['pident'] >= args.min_pident and h['length'] >= args.min_length]
            if filtered_hits:
                sorted_filtered = sort_hits_by_quality(filtered_hits)
                best_hit = sorted_filtered[0]
                logger.info(f"No taxonomic matches found. Showing top overall hit by quality: {best_hit['hit_id']}")
            else:
                best_hit = None
                logger.info(f"No hits passed quality filters")
        
        # Log the processing results for this sample
        logger.info(f"Sample processing results:")
        logger.info(f"  taxval_rank: {args.taxval_rank}")
        logger.info(f"  expected_taxonomy (lineage): {full_lineage}")
        logger.info(f"  expected_taxonomy_rank: {expected_taxonomy_rank}")
        logger.info(f"  obs_taxonomy: {obs_taxonomy_str[:200]}..." if len(obs_taxonomy_str) > 200 else f"  obs_taxonomy: {obs_taxonomy_str}")
        logger.info(f"  match_taxonomy: {match_taxonomy}")
        logger.info(f"  matched_rank: {matched_rank}")
        
        # Create result dictionary
        result = {
            'seq_id': seq_id,
            'Process_ID': process_id,
            'taxval_rank': args.taxval_rank,
            'expected_taxonomy': full_lineage,
            'expected_taxonomy_rank': expected_taxonomy_rank,
            'obs_taxonomy': obs_taxonomy_str,
            'match_taxonomy': match_taxonomy,
            'matched_rank': matched_rank,
            'top_matching_hit': best_hit['hit_id'] if best_hit else 'No hits',
            'pident': best_hit['pident'] if best_hit else '',
            'length': best_hit['length'] if best_hit else '',
            'mismatch': best_hit['mismatch'] if best_hit else '',
            'gaps': best_hit['gaps'] if best_hit else '',
            'evalue': best_hit['evalue'] if best_hit else ''
        }
        
        # Add to matching sequences for FASTA output if there was a match
        results.append(result)
    
    # Select best sequences per Process_ID
    results = select_best_sequences(results, logger)
    
    # Collect sequences marked as selected for FASTA output
    for result in results:
        if result.get('selected') == 'YES':
            seq_id = result['seq_id']
            if seq_id in sequences:
                matching_sequences[seq_id] = sequences[seq_id]
            else:
                logger.warning(f"Selected sequence {seq_id} not found in input FASTA")
    
    # Write output files
    write_output_csv(results, args.output_csv, logger)
    write_output_fasta(matching_sequences, args.output_fasta, logger)
    
    # Summary statistics
    total_sequences = len(results)
    matched_sequences = len(matching_sequences)
    selected_sequences = sum(1 for r in results if r.get('selected') == 'YES')
    logger.info(f"Summary: {matched_sequences}/{total_sequences} sequences had matches")
    logger.info(f"Summary: {selected_sequences} sequences selected as best per Process_ID")
    logger.info("BLAST taxonomy validation completed successfully")

if __name__ == "__main__":
    main()
