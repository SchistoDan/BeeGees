#!/usr/bin/env python3
"""
Determine barcode outcomes from a CSV file based on 'selected' column values.

Outcomes:
    PASS: At least one row for the ID has selected == 'YES'
    PARTIAL: No 'YES', but at least one row has selected == 'NO'
    FAIL: No rows with 'YES' or 'NO' in selected column
"""

import argparse
import csv
import logging


def setup_logging(log_path: str) -> logging.Logger:
    """Configure logging to file."""
    logger = logging.getLogger("barcode_outcome")
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(log_path, mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def determine_outcome(rows: list) -> str:
    """
    Determine barcode outcome based on selected column values.
    
    Args:
        rows: List of row dictionaries for a single ID
        
    Returns:
        'PASS', 'PARTIAL', or 'FAIL'
    """
    selected_values = [row['selected'].strip() for row in rows]
    
    if 'YES' in selected_values:
        return 'PASS'
    elif 'NO' in selected_values:
        return 'PARTIAL'
    else:
        return 'FAIL'


def lookup_observed_taxonomy(top_matching_hit: str, obs_taxonomy: str) -> str:
    """
    Look up the top_matching_hit in obs_taxonomy and return the matching entry.
    
    Args:
        top_matching_hit: The hit ID to search for
        obs_taxonomy: The obs_taxonomy string to search in
        
    Returns:
        Matching taxonomy entry or 'NA' if not found
    """
    if not top_matching_hit or top_matching_hit.strip() == '':
        return 'NA'
    
    if not obs_taxonomy or obs_taxonomy.strip() == '':
        return 'NA'
    
    hit_id = top_matching_hit.strip()
    
    # Split obs_taxonomy by semicolon and find the entry containing the hit ID
    for entry in obs_taxonomy.split(';'):
        entry = entry.strip()
        if entry.startswith(hit_id):
            return entry
    
    return 'NA'


def get_taxonomy_info(rows: list, outcome: str) -> tuple:
    """
    Extract expected_taxonomy and observed_taxonomy for an ID.
    
    Args:
        rows: List of row dictionaries for a single ID
        outcome: The barcode outcome (PASS, PARTIAL, or FAIL)
        
    Returns:
        Tuple of (expected_taxonomy, observed_taxonomy)
    """
    # expected_taxonomy - get from first row that has a value
    expected_taxonomy = ''
    for row in rows:
        val = row.get('expected_taxonomy', '').strip()
        if val:
            expected_taxonomy = val
            break
    
    if outcome == 'PASS':
        # Find the row where selected == 'YES'
        for row in rows:
            if row['selected'].strip() == 'YES':
                top_hit = row.get('top_matching_hit', '').strip()
                obs_tax = row.get('obs_taxonomy', '').strip()
                observed_taxonomy = lookup_observed_taxonomy(top_hit, obs_tax)
                return expected_taxonomy, observed_taxonomy
        # Fallback (shouldn't happen for PASS)
        return expected_taxonomy, 'NA'
    
    else:
        # PARTIAL or FAIL: collect all top_matching_hits and look them up
        observed_taxonomies = []
        seen_hits = set()
        
        for row in rows:
            top_hit = row.get('top_matching_hit', '').strip()
            obs_tax = row.get('obs_taxonomy', '').strip()
            
            if top_hit and top_hit not in seen_hits:
                seen_hits.add(top_hit)
                result = lookup_observed_taxonomy(top_hit, obs_tax)
                if result != 'NA':
                    observed_taxonomies.append(result)
        
        if observed_taxonomies:
            observed_taxonomy = '; '.join(observed_taxonomies)
        else:
            observed_taxonomy = 'NA'
        
        return expected_taxonomy, observed_taxonomy


def process_csv(input_path: str, logger: logging.Logger) -> list:
    """
    Process input CSV and collect data per ID.
    
    Args:
        input_path: Path to input CSV file
        logger: Logger instance
        
    Returns:
        List of tuples: (ID, outcome, expected_taxonomy, observed_taxonomy)
    """
    id_rows = {}
    row_count = 0
    
    logger.info(f"Reading input file: {input_path}")
    
    with open(input_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        
        required_cols = ['ID', 'selected', 'expected_taxonomy', 'top_matching_hit', 'obs_taxonomy']
        for col in required_cols:
            if col not in reader.fieldnames:
                raise ValueError(f"Input CSV must contain '{col}' column")
        
        for row in reader:
            row_count += 1
            sample_id = row['ID']
            
            if sample_id not in id_rows:
                id_rows[sample_id] = []
            
            id_rows[sample_id].append(row)
    
    logger.info(f"Processed {row_count} rows")
    logger.info(f"Found {len(id_rows)} unique IDs")
    
    # Determine outcomes and taxonomy info
    results = []
    for sample_id in sorted(id_rows.keys()):
        rows = id_rows[sample_id]
        outcome = determine_outcome(rows)
        expected_tax, observed_tax = get_taxonomy_info(rows, outcome)
        results.append((sample_id, outcome, expected_tax, observed_tax))
    
    return results


def write_output(results: list, output_path: str, logger: logging.Logger) -> None:
    """
    Write results to TSV file.
    
    Args:
        results: List of tuples (ID, outcome, expected_taxonomy, observed_taxonomy)
        output_path: Path to output TSV file
        logger: Logger instance
    """
    pass_count = sum(1 for r in results if r[1] == 'PASS')
    partial_count = sum(1 for r in results if r[1] == 'PARTIAL')
    fail_count = sum(1 for r in results if r[1] == 'FAIL')
    
    logger.info(f"Writing output to: {output_path}")
    logger.info(f"Outcome summary: PASS={pass_count}, PARTIAL={partial_count}, FAIL={fail_count}")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['ID', 'barcode_outcome', 'expected_taxonomy', 'observed_taxonomy'])
        
        for sample_id, outcome, expected_tax, observed_tax in results:
            writer.writerow([sample_id, outcome, expected_tax, observed_tax])
    
    logger.info("Done")


def main():
    parser = argparse.ArgumentParser(
        description="Determine barcode outcomes from CSV based on 'selected' column values."
    )
    parser.add_argument(
        '-i', '--in', '--input',
        dest='input',
        required=True,
        help='Input CSV file path'
    )
    parser.add_argument(
        '-o', '--out', '--output',
        dest='output',
        required=True,
        help='Output TSV file path'
    )
    parser.add_argument(
        '--log',
        required=True,
        help='Log file path'
    )
    
    args = parser.parse_args()
    
    logger = setup_logging(args.log)
    
    try:
        results = process_csv(args.input, logger)
        write_output(results, args.output, logger)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
