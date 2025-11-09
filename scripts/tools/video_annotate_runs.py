"""
Rename all files named 'video.mp4' found recursively so they become
'video_run_<x>.mp4', where x is the smallest unused positive integer
in that directory.

Usage:
  python3 video_annotate_runs.py [ROOT_DIR] [--dry-run]

Examples:
  python3 video_annotate_runs.py --root /path/to/library --dry-run
"""

import argparse
import os
import sys

def find_targets(root: str):
    targets = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # exact match; change to name.lower() == "video.mp4"
        if "video.mp4" in filenames:
            targets.append(os.path.join(dirpath, "video.mp4"))
    return sorted(targets)

def next_available_name(dirpath: str, stem: str, ext: str) -> str:
    x = 1
    while True:
        candidate = os.path.join(dirpath, f"{stem}_run_{x}{ext}")
        if not os.path.exists(candidate):
            return candidate
        x += 1

def main():
    parser = argparse.ArgumentParser(description="Append _run_x to every 'video.mp4' found recursively.")
    parser.add_argument("--root", default=".", help="Root directory to scan (default: current directory).")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without changing files.")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Error: '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    targets = find_targets(root)
    if not targets:
        print("No files named 'video.mp4' found.")
        return

    renamed = 0
    errors = 0

    for old_path in targets:
        dirpath = os.path.dirname(old_path)
        stem, ext = os.path.splitext(os.path.basename(old_path))  # 'video', '.mp4'
        new_path = next_available_name(dirpath, stem, ext)

        rel_old = os.path.relpath(old_path, root)
        rel_new = os.path.relpath(new_path, root)

        if args.dry_run:
            print(f"[DRY-RUN] {rel_old} -> {rel_new}")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed: {rel_old} -> {rel_new}")
        renamed += 1

    if not args.dry_run:
        print(f"\nDone. Renamed {renamed} file(s).", file=sys.stderr)
        if errors:
            print(f"{errors} error(s) occurred.", file=sys.stderr)

if __name__ == "__main__":
    main()
