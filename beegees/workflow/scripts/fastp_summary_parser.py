#!/usr/bin/env python3

import argparse
import json
import csv
import os
import glob
import sys

def parse_fastp_json(json_path):
    """Parse fastp JSON file and extract specified statistics."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        stats = {}
        
        # Before filtering stats
        bf = data['summary']['before_filtering']
        stats.update({
            'before_total_reads': bf['total_reads'],
            'before_total_bases': bf['total_bases'],
            'before_q20_bases': bf['q20_bases'],
            'before_q30_bases': bf['q30_bases'],
            'before_q20_rate': bf['q20_rate'],
            'before_q30_rate': bf['q30_rate'],
            'before_gc_content': bf['gc_content']
        })
        
        # After filtering stats
        af = data['summary']['after_filtering']
        stats.update({
            'after_total_reads': af['total_reads'],
            'after_total_bases': af['total_bases'],
            'after_q20_bases': af['q20_bases'],
            'after_q30_bases': af['q30_bases'],
            'after_q20_rate': af['q20_rate'],
            'after_q30_rate': af['q30_rate'],
            'after_gc_content': af['gc_content']
        })
        
        # Filtering result stats
        fr = data['filtering_result']
        stats.update({
            'passed_filter_reads': fr['passed_filter_reads'],
            'low_quality_reads': fr['low_quality_reads'],
            'too_many_N_reads': fr['too_many_N_reads'],
            'too_short_reads': fr['too_short_reads'],
            'too_long_reads': fr['too_long_reads']
        })
        
        # Duplication rate (present for both single- and paired-end)
        stats['duplication_rate'] = data.get('duplication', {}).get('rate', '')
        
        # Insert size peak (paired-end only; single-end fastp emits no
        # 'insert_size' block, so default to empty rather than dropping the sample)
        stats['insert_size_peak'] = data.get('insert_size', {}).get('peak', '')
        
        return stats
        
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing {json_path}: {e}", file=sys.stderr)
        return None

def find_fastp_files(trimmed_data_paths):
    """Find all fastp JSON files in the specified directories.
    
    Supports both:
    1. Nested structure: trimmed_data/sample_name/sample_name.json
    2. Flat structure: directory/*.json (sample name from filename)
    """
    fastp_files = []
    
    for trimmed_data_path in trimmed_data_paths:
        if not os.path.exists(trimmed_data_path):
            print(f"Warning: Path does not exist: {trimmed_data_path}", file=sys.stderr)
            continue
        
        # First, look for JSON files directly in the directory (flat structure)
        json_pattern = os.path.join(trimmed_data_path, "*.json")
        direct_json_files = glob.glob(json_pattern)
        
        if direct_json_files:
            # Flat structure found
            for json_file in direct_json_files:
                sample_name = os.path.splitext(os.path.basename(json_file))[0]
                fastp_files.append((sample_name, json_file))
            print(f"Found {len(direct_json_files)} JSON files in flat structure at {trimmed_data_path}")
        else:
            # Look for subdirectories (nested structure)
            try:
                subdirs = [d for d in os.listdir(trimmed_data_path) 
                          if os.path.isdir(os.path.join(trimmed_data_path, d))]
                
                nested_found = 0
                for subdir in subdirs:
                    sample_name = subdir
                    json_pattern = os.path.join(trimmed_data_path, subdir, f"{sample_name}*.json")
                    
                    # Use glob to handle any potential wildcards, though we expect exact match
                    matching_files = glob.glob(json_pattern)
                    
                    if matching_files:
                        fastp_files.append((sample_name, matching_files[0]))
                        nested_found += 1
                    else:
                        print(f"Warning: No fastp JSON found for sample {sample_name} in {json_pattern}", file=sys.stderr)
                
                if nested_found > 0:
                    print(f"Found {nested_found} JSON files in nested structure at {trimmed_data_path}")
                elif subdirs:
                    print(f"Warning: Found subdirectories but no matching JSON files in nested structure at {trimmed_data_path}", file=sys.stderr)
                else:
                    print(f"Warning: No JSON files or subdirectories found at {trimmed_data_path}", file=sys.stderr)
                    
            except PermissionError:
                print(f"Warning: Permission denied accessing {trimmed_data_path}", file=sys.stderr)
    
    return fastp_files

def main():
    parser = argparse.ArgumentParser(
        description="Parse fastp JSON reports and compile statistics into a CSV file. "
                   "Supports both nested (trimmed_data/sample/sample.json) and flat (directory/*.json) structures."
    )
    parser.add_argument(
        '-i', '--input', 
        nargs='+', 
        required=True,
        help='One or more paths to directories containing fastp JSON files'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    
    # Find all fastp JSON files
    fastp_files = find_fastp_files(args.input)
    
    if not fastp_files:
        print("Error: No fastp JSON files found in any of the specified directories", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(fastp_files)} fastp JSON files to process")
    
    # Define CSV header
    header = [
        'sample_name',
        'before_total_reads', 'before_total_bases', 'before_q20_bases', 'before_q30_bases',
        'before_q20_rate', 'before_q30_rate', 'before_gc_content',
        'after_total_reads', 'after_total_bases', 'after_q20_bases', 'after_q30_bases', 
        'after_q20_rate', 'after_q30_rate', 'after_gc_content',
        'passed_filter_reads', 'low_quality_reads', 'too_many_N_reads', 
        'too_short_reads', 'too_long_reads',
        'duplication_rate', 'insert_size_peak'
    ]
    
    # Process files and write CSV
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        processed_count = 0
        for sample_name, json_path in fastp_files:
            stats = parse_fastp_json(json_path)
            
            if stats is not None:
                row = [sample_name] + [stats.get(col, 'NA') for col in header[1:]]
                writer.writerow(row)
                processed_count += 1
                print(f"Processed: {sample_name}")
            else:
                print(f"Skipped: {sample_name} (parsing failed)")
    
    print(f"\nCompleted! Processed {processed_count}/{len(fastp_files)} samples")
    print(f"Output written to: {args.output}")

if __name__ == "__main__":
    main()
