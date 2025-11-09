#!/usr/bin/env python3
"""
data_dedup.py

Usage:
  # All rows
  python3 data_dedup.py all --csv-file path/to/file.csv

  # Only rows where finalized == 'done' AND there are exactly 2 reviewer names
  python3 data_dedup.py --mode ready --csv-file path/to/file.csv

  # Extras
  python3 data_dedup.py --mode ready --csv-file path/to/file.csv --column Keywords --author "Author Name" --output unique_entry_counts.json

Notes:
- Counts are case-insensitive but preserve the first-seen casing as JSON keys.
- Splits on commas, semicolons, pipes, slashes, and newlines.
- If --no-sort is not set, keys are sorted alphabetically (case-insensitive) in the output JSON.
- The 'ready' mode includes a row only if:
    finalized column equals 'done' (case-insensitive), AND
    the 'reviewer' column contains exactly two non-empty names.
"""

import argparse
import csv
import json
import re
from collections import OrderedDict, Counter

SPLIT_RE = re.compile(r'[,\;\|/\\\n]+')  # for Keywords and general splitting
REVIEWER_SPLIT_RE = re.compile(r'(?:\band\b|[,\;\|/\\\n]+)', re.IGNORECASE)  # split reviewer names


def get_cell(row, target_col):
    """Case-insensitive access to a column in a DictReader row."""
    target = target_col.strip().lower()
    for k, v in row.items():
        if k is not None and k.strip().lower() == target:
            return v or ""
    return ""


def split_keywords(cell):
    """Split a keywords cell into cleaned tokens."""
    tokens = []
    for tok in SPLIT_RE.split(cell):
        t = tok.strip().strip('"').strip("'")
        if t:
            # collapse internal whitespace
            t = " ".join(t.split())
            tokens.append(t)
    return tokens


def parse_reviewer_names(cell):
    """Split reviewers cell into cleaned name tokens."""
    names = []
    for tok in REVIEWER_SPLIT_RE.split(cell or ""):
        t = tok.strip().strip('"').strip("'")
        if t:
            t = " ".join(t.split())
            names.append(t)
    return names


def row_is_ready(row):
    """Return True iff finalized == 'done' and exactly two reviewer names are present."""
    finalized = get_cell(row, "finalized?").strip().lower()
    reviewers_cell = get_cell(row, "reviewer")
    names = parse_reviewer_names(reviewers_cell)
    return finalized == "done" and len(names) == 2


def main():
    ap = argparse.ArgumentParser(
        description="Extract and count keywords from a CSV, output JSON {keyword: count}."
    )
    ap.add_argument(
        "--mode",
        choices=["all", "ready"],
        default="ready",
        help="Row filter: 'all' (no filter) or 'ready' (finalized=='done' and exactly 2 reviewers). Default: all",
    )
    ap.add_argument("--csv-file", required=True, help="Path to the CSV file")
    ap.add_argument("--column", default="Keywords", help="Column name to extract (default: Keywords)")
    ap.add_argument("--author", default=None, help="If set, only rows where Author equals this value are used")
    ap.add_argument("--output", "-o", default=None, help="Optional path to write JSON mapping {keyword: count}")
    ap.add_argument("--no-sort", action="store_true", help="Preserve first-seen order instead of sorting keys")
    args = ap.parse_args()

    # Count occurrences (case-insensitive), but remember first-seen display casing
    counts = Counter()                 # normalized -> count
    display_for_norm = OrderedDict()   # normalized -> first-seen display

    # Use utf-8-sig to gracefully handle BOMs
    with open(args.csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Author filter (if any)
            if args.author is not None:
                author_cell = get_cell(row, "Author")
                if author_cell.strip() != args.author.strip():
                    continue

            # Mode filter
            if args.mode == "ready" and not row_is_ready(row):
                continue

            # Extract keywords
            cell = get_cell(row, args.column)
            if not cell:
                continue

            for kw in split_keywords(cell):
                norm = kw.casefold()
                if norm not in display_for_norm:
                    display_for_norm[norm] = kw  # preserve first-seen casing
                counts[norm] += 1

    # Prepare ordered output mapping: {display_keyword: count}
    ordered_norms = list(display_for_norm.keys())
    if not args.no_sort:
        ordered_norms.sort()  # already normalized, so this is effectively case-insensitive

    out = {display_for_norm[n]: counts[n] for n in ordered_norms}

    # Print JSON to stdout
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # Optionally write JSON to a file
    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_f:
            json.dump(out, out_f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
