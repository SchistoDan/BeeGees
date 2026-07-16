#!/usr/bin/env python3
"""
Count consensus sequences in a combined multi-FASTA by preprocessing mode and
MGE parameter combination.

Under the explicit-suffix scheme, every consensus header ends in one of six
mode tags:

    pre-clean : _concat | _merge | _se
    post-clean: _fcleaner_concat | _fcleaner_merge | _fcleaner_se

(Historically concat was left unmarked; a trailing-unmarked header is still
counted as "concat" for backward compatibility with older combined FASTAs.)
"""

import re
import argparse
from collections import defaultdict
from datetime import datetime

# Ordered most-specific first so the longer fcleaner tags win over bare tags.
# Each entry maps a header-ending tag to its category label.
CATEGORY_TAGS = [
    ("_fcleaner_concat", "fcleaner_concat"),
    ("_fcleaner_merge",  "fcleaner_merge"),
    ("_fcleaner_se",     "fcleaner_se"),
    ("_concat",          "concat"),
    ("_merge",           "merge"),
    ("_se",              "se"),
]

# Category display order for reporting.
CATEGORIES = ["concat", "merge", "se",
              "fcleaner_concat", "fcleaner_merge", "fcleaner_se"]

# MGE parameter combinations checked within each category.
R_VALUES = ['r_1', 'r_1.3', 'r_1.5']
S_VALUES = ['s_50', 's_100']

# Valid sequence characters (nucleotides, N, gap, MGE pad).
VALID_SEQ_CHARS = re.compile(r'^[GCATNgcatn\-~]+$')


def categorise_header(header):
    """Return the mode category for a FASTA header.

    Matching is done on the header ending so that a 'se'/'merge'/'concat'
    substring occurring elsewhere (e.g. inside a sample ID) cannot cause a
    misclassification. Falls back to 'concat' for legacy unmarked headers.
    """
    h = header.rstrip()
    for tag, label in CATEGORY_TAGS:
        if h.endswith(tag):
            return label
    return "concat"


def analyse_fasta(fasta_file, log_file=None, tsv_file=None):
    start_time = datetime.now()

    total_headers = 0
    empty_sequences = 0
    non_empty_sequences = 0

    category_counts = {c: 0 for c in CATEGORIES}
    # keyed by (param_pattern, category)
    parameter_counts = defaultdict(int)

    def tally(header, sequence):
        """Update counters for one (header, sequence) record."""
        nonlocal non_empty_sequences, empty_sequences
        if not sequence:
            empty_sequences += 1
            return
        non_empty_sequences += 1
        category = categorise_header(header)
        category_counts[category] += 1
        for r_val in R_VALUES:
            for s_val in S_VALUES:
                param_pattern = f"{r_val}_{s_val}"
                if param_pattern in header:
                    parameter_counts[(param_pattern, category)] += 1

    current_header = None
    current_sequence = ""

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                total_headers += 1
                if current_header is not None:
                    tally(current_header, current_sequence)
                current_header = line
                current_sequence = ""
            elif VALID_SEQ_CHARS.match(line):
                current_sequence += line
            else:
                print(f"Warning: Line with invalid characters found after "
                      f"header {current_header}: {line[:50]}...")

        # Final record
        if current_header is not None:
            tally(current_header, current_sequence)

    end_time = datetime.now()

    # ---- Build results ----
    results = []
    results.append(f"Total sequences: {total_headers}")
    results.append(f"Empty sequences: {empty_sequences}")
    results.append(f"Non-empty sequences: {non_empty_sequences}")
    results.append("")
    results.append("Non-empty sequences breakdown by mode type:")
    for c in CATEGORIES:
        results.append(f"  '{c}' sequences: {category_counts[c]}")

    results.append("\nParameter Type Counts:")
    for (param_pattern, category) in sorted(parameter_counts.keys()):
        label = f"{param_pattern} [{category}]"
        results.append(f"{label}: {parameter_counts[(param_pattern, category)]}")

    for line in results:
        print(line)

    if log_file:
        with open(log_file, 'w') as f:
            f.write("=" * 70 + '\n')
            f.write("Script: barcode_consensus_count.py\n")
            f.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {fasta_file}\n")
            f.write("=" * 70 + '\n\n')
            for line in results:
                f.write(line + '\n')
            f.write('\n')
            f.write("=" * 70 + '\n')
            f.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {end_time - start_time}\n")
            f.write("=" * 70 + '\n')
        print(f"\nResults have been written to: {log_file}")

    if tsv_file:
        with open(tsv_file, 'w') as f:
            f.write("Metric\tCount\n")
            f.write(f"Total sequences\t{total_headers}\n")
            f.write(f"Empty sequences\t{empty_sequences}\n")
            f.write(f"Non-empty sequences\t{non_empty_sequences}\n")
            f.write("\n")

            f.write("Mode Type\tCount\n")
            for c in CATEGORIES:
                f.write(f"{c} sequences\t{category_counts[c]}\n")
            f.write("\n")

            f.write("Parameter Type\tCategory\tCount\n")
            for (param_pattern, category) in sorted(parameter_counts.keys()):
                f.write(f"{param_pattern}\t{category}\t"
                        f"{parameter_counts[(param_pattern, category)]}\n")
        print(f"Results have been written to TSV: {tsv_file}")


def main():
    parser = argparse.ArgumentParser(description='analyse FASTA file for sequence parameters')
    parser.add_argument('--input', required=True, help='Input FASTA file path')
    parser.add_argument('--log-file', help='Output log file path (optional)')
    parser.add_argument('--tsv', help='Output TSV file path (optional)')

    args = parser.parse_args()

    analyse_fasta(args.input, args.log_file, args.tsv)


if __name__ == "__main__":
    main()
