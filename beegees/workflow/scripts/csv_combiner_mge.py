#!/usr/bin/env python3
"""
CSV Combiner - combine multiple CSV files into a single file.

Pure concatenator:
- Preserves the header structure from the first CSV file.
- Aligns data from subsequent files to match the first file's column order,
  filling missing columns with empty values (with a warning).

NOTE: This script no longer rewrites the 'mge_params' column. Mode provenance
(_concat / _merge / _se and their _fcleaner_* variants) is now written directly
into the consensus FASTA headers by rename_headers.py and 05_consensus_generator.py,
and is derived from those headers by compile_barcoding_stats.py. The combiner
therefore only needs to stitch the per-mode stats CSVs together verbatim.

Usage:
    python csv_combiner_mge.py -i input1.csv input2.csv ... -o output.csv
"""

import argparse
import csv
import os
import sys


def combine_csv_files(input_files, output_file):
    if not input_files:
        print("Error: No input files provided.")
        return False

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Error: Input file '{file_path}' does not exist.")
            return False

    try:
        # Header from first file
        with open(input_files[0], 'r', newline='') as infile:
            reader = csv.reader(infile)
            try:
                first_file_header = next(reader)
            except StopIteration:
                print(f"Error: First file '{input_files[0]}' is empty or has no header.")
                return False

        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(first_file_header)

            for i, input_file in enumerate(input_files):
                with open(input_file, 'r', newline='') as infile:
                    reader = csv.reader(infile)

                    try:
                        current_header = next(reader)
                    except StopIteration:
                        if i == 0:
                            continue
                        print(f"Warning: File '{input_file}' is empty. Skipping.")
                        continue

                    # First file: header already written, just stream its rows.
                    if i == 0:
                        for row in reader:
                            writer.writerow(row)
                        continue

                    # Subsequent files: map columns onto the first file's order.
                    column_mapping = []
                    for target_col in first_file_header:
                        if target_col in current_header:
                            column_mapping.append(current_header.index(target_col))
                        else:
                            print(f"Warning: Column '{target_col}' not found in file "
                                  f"'{input_file}'. Filling with empty values.")
                            column_mapping.append(None)

                    for row in reader:
                        new_row = []
                        for map_idx in column_mapping:
                            value = ""
                            if map_idx is not None and map_idx < len(row):
                                value = row[map_idx]
                            new_row.append(value)
                        writer.writerow(new_row)

        print(f"Successfully combined {len(input_files)} CSV file(s) into '{output_file}'.")
        return True

    except Exception as e:
        print(f"Error: Failed to combine CSV files: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Combine multiple CSV files into a single file.')
    parser.add_argument('-i', '--input', nargs='+', required=True,
                        help='Input CSV files to combine')
    parser.add_argument('-o', '--output', required=True,
                        help='Output CSV file path')

    args = parser.parse_args()
    success = combine_csv_files(args.input, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
