#!/usr/bin/env python3
"""
Script to extract experiments from CSV and prepare JSON configuration
Usage: python3 prepare_experiments.py --author "Author Name" --csv file.csv --output-dir custom_dir
Or: python3 prepare_experiments.py "Author Name" file.csv custom_dir
"""

import csv
import json
import sys
import os
import argparse
from pathlib import Path


def prepare_experiments(author, csv_file, output_dir=None):
    """Extract experiments for a specific author and create JSON config"""

    # Video generation parameters (defaults)
    defaults = {
        "provider": "wan",
        "model": "wan2.5-t2v-preview",
        "seconds": 8,
        "width": 1280,
        "height": 720,
        "extra": {
            "prompt_extend": True,
            "audio": False,
            "negative_prompt": "low quality, artifacts"
        }
    }

    # Create output directory
    if output_dir is None:
        output_dir = Path("out") / author.replace(" ", "_")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / "experiments.json"

    print("=" * 60)
    print("Preparing Experiments")
    print(f"Author: {author}")
    print(f"Reading from CSV: {csv_file}")
    print(f"Output directory: {output_dir}")
    print(f"Output JSON: {output_json}")
    print("=" * 60)
    print()

    # Check if CSV exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return 1

    tasks = []
    counter = 1

    # Read CSV file
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Get author from row
            row_author = row.get('Author', '').strip()
            prompt = row.get('Prompts', '').strip()

            # Skip if not matching author or no prompt
            if row_author != author or not prompt:
                continue

            # Create experiment folder
            exp_folder = output_dir / f"{counter:03d}"
            exp_folder.mkdir(parents=True, exist_ok=True)

            # Get fields
            title = row.get('Example Title', '').strip()
            expected_phenomenon = row.get('Expected phenomenon', '').strip()
            fields = row.get('Fields', '').strip()
            keywords = row.get('Keywords', '').strip()
            source = row.get('Source', '').strip()

            # Write info file
            info_file = exp_folder / "info.txt"
            with open(info_file, 'w', encoding='utf-8') as info_f:
                info_f.write(f"Title: {title}\n")
                info_f.write(f"Author: {author}\n")
                info_f.write(f"Fields: {fields}\n")
                info_f.write(f"Keywords: {keywords}\n")
                info_f.write(f"Source: {source}\n")
                info_f.write(f"\n=== Prompt ===\n")
                info_f.write(f"{prompt}\n")
                info_f.write(f"\n=== Expected Phenomenon ===\n")
                info_f.write(f"{expected_phenomenon}\n")

            # Add task
            video_file = str(exp_folder / "video.mp4")
            tasks.append({
                "id": counter,
                "prompt": prompt,
                "output_path": video_file
            })

            print(f"Prepared experiment #{counter}: {title}")
            counter += 1

    # Check if any tasks were found
    if not tasks:
        print()
        print(f"Warning: No experiments found for author: {author}")
        return 1

    # Write JSON configuration
    config = {
        "defaults": defaults,
        "tasks": tasks
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"✓ Successfully prepared {len(tasks)} experiments")
    print(f"JSON configuration: {output_json}")
    print("=" * 60)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract experiments from CSV and prepare JSON configuration for video generation"
    )
    parser.add_argument(
        "--author",
        default="Yujie Zhao",
        help="Author name to filter experiments (default: Yujie Zhao)"
    )
    parser.add_argument(
        "--csv",
        default="test.csv",
        help="Path to CSV file (default: test.csv)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: out/<author_name>)"
    )

    # Also support positional arguments for backward compatibility
    parser.add_argument(
        "author_pos",
        nargs="?",
        help="Author name (positional, overrides --author)"
    )
    parser.add_argument(
        "csv_pos",
        nargs="?",
        help="CSV file path (positional, overrides --csv)"
    )
    parser.add_argument(
        "output_dir_pos",
        nargs="?",
        help="Output directory (positional, overrides --output-dir)"
    )

    args = parser.parse_args()

    # Use positional args if provided, otherwise use named args
    author = args.author_pos if args.author_pos else args.author
    csv_file = args.csv_pos if args.csv_pos else args.csv
    output_dir = args.output_dir_pos if args.output_dir_pos else args.output_dir

    exit_code = prepare_experiments(author, csv_file, output_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
