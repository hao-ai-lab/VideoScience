#!/usr/bin/env python3
"""
Script to extract experiments from CSV and prepare JSON configuration for video generation and evaluation.

- Filtering by multiple authors (comma-separated)
- Only include rows that are "ready": finalized? == 'done' AND exactly two reviewers
- Name each prompt directory from a CSV Unique ID (e.g., vid_005) or sequentially with --name-by seq

Usage:
  python3 prepare_experiments.py --csv file.csv --author "Alice,Bob" --mode all --name-by uid --output-dir out/crosseval
"""

import csv
import json
import sys
import os
import re
import argparse
from pathlib import Path

# -------- helper functions -------- #

REVIEWER_SPLIT_RE = re.compile(r'(?:\band\b|[,\;\|/\\\n]+)', re.IGNORECASE)

def get_cell(row, target_col: str) -> str:
    """Case-insensitive access to a column in a DictReader row."""
    t = target_col.strip().lower()
    for k, v in row.items():
        if k is not None and k.strip().lower() == t:
            return v
    return ""

def parse_reviewer_names(cell: str) -> list[str]:
    names = []
    for tok in REVIEWER_SPLIT_RE.split(cell or ""):
        t = tok.strip().strip('"').strip("'")
        if t:
            names.append(" ".join(t.split()))
    return names

def row_is_ready(row) -> bool:
    """finalized? == 'done' AND exactly two reviewer names."""
    finalized = get_cell(row, "finalized?").strip().lower()
    names = parse_reviewer_names(get_cell(row, "reviewer"))
    return finalized == "done" and len(names) == 2

# naming
def slugify(s: str) -> str:
    s = re.sub(r'[^A-Za-z0-9_-]+', '-', s).strip('-')
    return s or "uid"

def uid_to_folder(uid: str, prefix: str = "vid_", pad: int = 3) -> str:
    uid = (uid or "").strip()
    if uid.isdigit():
        return f"{prefix}{int(uid):0{pad}d}"
    if not uid:
        return f"{prefix}uid"
    return f"{prefix}{slugify(uid)}"

def parse_authors(author_arg: str | None) -> set[str]:
    if not author_arg:
        return set()
    return {a.strip() for a in author_arg.split(",") if a.strip()}

# ============= main function ============= #

def prepare_experiments(authors_csv: set[str],
                        csv_file: str,
                        output_dir: str | None = None,
                        id_column: str = "Unique ID",
                        name_by: str = "uid",  # 'uid' or 'seq'
                        prefix: str = "vid_",
                        pad: int = 3,
                        mode: str = "ready"):   # 'ready' or 'all'
    """Extract experiments for given authors and create JSON config for all models."""

    MODELS = [
        {
            "name": "veo3gen",
            "provider": "veo3gen",
            "model": "veo3-quality",
            "seconds": 8,
            "width": 1920,
            "height": 1080,
            "extra": {
                "modelVersion": "3.1",
                "aspect_ratio": "16:9",
                "resolution": "1080p",
                "negative_prompt": "cartoon, low quality",
                "enhancePrompt": False,
                "audio": False,
            }
        },
        {
            "name": "luma_ray2",
            "provider": "ray",
            "model": "ray-2",
            "seconds": 9,
            "width": 1280,
            "height": 720,
            "extra": {"aspect_ratio": "16:9", "resolution": "720p", "loop": False}
        },
        {
            "name": "wan2.5",
            "provider": "wan",
            "model": "wan2.5-t2v-preview",
            "seconds": 10,
            "width": 1280,
            "height": 720,
            "extra": {"prompt_extend": False, "audio": False}
        },
        {
            "name": "sora2",
            "provider": "sora-openai",
            "model": "sora-2",
            "seconds": 8,
            "width": 1280,
            "height": 720,
            "extra": {}
        },
        {
            "name": "hailuo2.3",
            "provider": "replicate",
            "model": "minimax/hailuo-2.3",
            "seconds": 10,
            "width": 1280,
            "height": 720,
            "extra": {"prompt_optimizer": False}
        },
        {
            "name": "seedance1pro",
            "provider": "replicate",
            "model": "bytedance/seedance-1-pro",
            "seconds": 10,
            "width": 1280,
            "height": 720,
            "extra": {"prompt_optimizer": False}
        },
        {
            "name": "kling2.5",
            "provider": "replicate",
            "model": "kwaivgi/kling-v2.5-turbo-pro",
            "seconds": 10,
            "width": 1280,
            "height": 720,
            "extra": {"prompt_optimizer": False}
        },
    ]

    # Output dir
    if output_dir is None:
        base = "multi_author" if len(authors_csv) != 1 else next(iter(authors_csv)).replace(" ", "_")
        output_dir = Path("out") / base
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / "experiments.json"

    print("=" * 60)
    print("Preparing Experiments")
    print(f"Authors: {', '.join(sorted(authors_csv)) if authors_csv else '<ALL>'}")
    print(f"Mode   : {mode}")
    print(f"Reading from CSV: {csv_file}")
    print(f"Output directory: {output_dir}")
    print(f"Output JSON: {output_json}")
    print("=" * 60)
    print()

    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return 1

    # Pass 1: read and filter rows
    rows = []
    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # author filter first
            row_author = get_cell(row, "Author").strip()
            if authors_csv and row_author not in authors_csv:
                continue
            # ready-only filter
            if mode == "ready" and not row_is_ready(row):
                continue
            # require prompt text
            prompt = get_cell(row, "Prompts").strip()
            if not prompt:
                continue
            rows.append(row)

    if not rows:
        print()
        print(f"Warning: No experiments found (authors={', '.join(sorted(authors_csv)) or '<ALL>'}, mode={mode}).")
        return 1

    tasks = []
    task_id = 1
    prompt_counter = 1
    seen_dirnames = set()

    # Pass 2: create per-prompt directories (by Unique ID or by sequence)
    for row in rows:
        prompt = get_cell(row, "Prompts").strip()
        title = get_cell(row, "Example Title").strip()
        expected_phenomenon = get_cell(row, "Expected phenomenon").strip()
        fields = get_cell(row, "Fields").strip()
        keywords = get_cell(row, "Keywords").strip()
        source = get_cell(row, "Source").strip()

        # Decide folder name
        if name_by == "uid":
            uid_val = get_cell(row, id_column).strip()
            if not uid_val:
                # try common fallbacks
                for candidate in ["Unique ID", "UniqueID", "UID", "Id", "ID"]:
                    uid_val = get_cell(row, candidate).strip()
                    if uid_val:
                        break
            dirname = uid_to_folder(uid_val, prefix=prefix, pad=pad) if uid_val else f"{prompt_counter:03d}"
        else:
            dirname = f"{prompt_counter:03d}"

        if dirname in seen_dirnames:
            raise SystemExit(
                f"Duplicate folder name generated: {dirname}. "
                f"Use a different prefix/pad or switch to --name-by seq."
            )
        seen_dirnames.add(dirname)

        exp_folder = output_dir / dirname
        exp_folder.mkdir(parents=True, exist_ok=True)

        # Shared info file
        info_file = exp_folder / "info.txt"
        with open(info_file, "w", encoding="utf-8") as info_f:
            info_f.write(f"Title: {title}\n")
            info_f.write(f"Author: {get_cell(row, 'Author').strip()}\n")
            info_f.write(f"Fields: {fields}\n")
            info_f.write(f"Keywords: {keywords}\n")
            info_f.write(f"Source: {source}\n")
            info_f.write(f"\n=== Prompt ===\n{prompt}\n")
            info_f.write(f"\n=== Expected Phenomenon ===\n{expected_phenomenon}\n")

        # Per-model subfolders + tasks
        for model_config in MODELS:
            model_folder = exp_folder / model_config["name"]
            model_folder.mkdir(parents=True, exist_ok=True)

            model_info = model_folder / "info.txt"
            with open(model_info, "w", encoding="utf-8") as info_f:
                info_f.write(f"Model: {model_config['name']}\n")
                info_f.write(f"Provider: {model_config['provider']}\n")
                info_f.write(f"Model ID: {model_config['model']}\n")
                info_f.write(f"Title: {title}\n")
                info_f.write(f"Author: {get_cell(row, 'Author').strip()}\n")
                info_f.write(f"\n=== Prompt ===\n{prompt}\n")
                info_f.write(f"\n=== Expected Phenomenon ===\n{expected_phenomenon}\n")

            video_file = str(model_folder / "video.mp4")
            timeout = 3000 if model_config["provider"] == "sora-openai" else 1200

            tasks.append({
                "id": task_id,
                "prompt": prompt,
                "provider": model_config["provider"],
                "model": model_config["model"],
                "seconds": model_config["seconds"],
                "width": model_config["width"],
                "height": model_config["height"],
                "extra": model_config["extra"],
                "output_path": video_file,
                "timeout_s": timeout
            })
            task_id += 1

        print(f"Prepared prompt {dirname} with {len(MODELS)} models: {title if title else (prompt[:50]+'...')}")
        prompt_counter += 1

    if not tasks:
        print("\nWarning: No tasks were produced (unexpected).")
        return 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"tasks": tasks}, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"✓ Successfully prepared {len(tasks)} tasks "
          f"({prompt_counter-1} prompts × {len(MODELS)} models)")
    print(f"JSON configuration: {output_json}")
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract experiments from CSV and prepare JSON configuration for video generation"
    )
    # required CSV path; author filter optional
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--author",
                        help="Author or comma-separated authors (filter). Omit to include all authors.")

    parser.add_argument("--output-dir", help="Output directory")

    # options
    parser.add_argument("--id-column", default="Unique ID",
                        help="CSV column for the unique id (default: 'Unique ID')")
    parser.add_argument("--name-by", choices=["uid", "seq"], default="uid",
                        help="Use 'uid' to name prompt dirs from the CSV unique id, "
                             "or 'seq' for 001, 002, ... (default: uid)")
    parser.add_argument("--prefix", default="vid_", help="Prefix when naming by uid (default: vid_)")
    parser.add_argument("--pad", type=int, default=3, help="Zero-pad width for numeric IDs (default: 3)")
    parser.add_argument("--mode", choices=["ready", "all"], default="ready",
                        help="Row filter: 'ready' requires finalized?=='done' and exactly 2 reviewers (default).")

    args = parser.parse_args()

    authors = parse_authors(args.author)
    exit_code = prepare_experiments(
        authors_csv=authors,
        csv_file=args.csv,
        output_dir=args.output_dir,
        id_column=args.id_column,
        name_by=args.name_by,
        prefix=args.prefix,
        pad=args.pad,
        mode=args.mode,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
