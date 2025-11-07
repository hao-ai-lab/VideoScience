#!/usr/bin/env python3
"""
Script to extract experiments from CSV and prepare JSON configuration for multi-model evaluation
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
    """Extract experiments for a specific author and create JSON config for all models"""

    # Define all models to test
    # NOTE: All models configured to NOT use prompt extension/augmentation
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
                "enhancePrompt": False,  # Disable prompt enhancement
                "audio": False,  # Disable audio generation
            }
        },
        {
            "name": "luma_ray2",
            "provider": "ray",
            "model": "ray-2",
            "seconds": 9,
            "width": 1280,
            "height": 720,
            "extra": {
                "aspect_ratio": "16:9",
                "resolution": "720p",
                "loop": False,
            }
        },
        {
            "name": "wan2.5",
            "provider": "wan",
            "model": "wan2.5-t2v-preview",
            "seconds": 10,
            "width": 1280,
            "height": 720,
            "extra": {
                "prompt_extend": False,  # Disable prompt extension
                "audio": False,  # Disable audio generation
            }
        },
        {
            "name": "kling2.5",
            "provider": "kling",
            "model": "kling-v2",
            "seconds": 10,
            "width": 1280,
            "height": 720,
            "extra": {
                "aspect_ratio": "16:9",
                "cfg_scale": 0.5,
                # Kling v2: No prompt optimization parameter - uses raw prompt
            }
        },
        {
            "name": "sora2",
            "provider": "sora-openai",
            "model": "sora-2",
            "seconds": 8,  # Sora-2 supports up to 8 seconds
            "width": 1280,
            "height": 720,
            "extra": {
                # Sora-2: Uses raw prompt as-is, no augmentation parameters
            }
        },
        {
            "name": "hailuo2.3",
            "provider": "replicate",
            "model": "minimax/hailuo-2.3",
            "seconds": 10,
            "width": 1280,
            "height": 720,
            "extra": {
                "prompt_optimizer": False,  # Disable Replicate's prompt optimization
            }
        },
        {
            "name": "seedance1pro",
            "provider": "replicate",
            "model": "bytedance/seedance-1-pro",
            "seconds": 10,
            "width": 1280,
            "height": 720,
            "extra": {
                "prompt_optimizer": False,  # Disable Replicate's prompt optimization
            }
        }
    ]

    # Video generation parameters (defaults) - now unused but kept for compatibility
    defaults = {
        "provider": "wan",
        "model": "wan2.5-t2v-preview",
        "seconds": 10,
        "width": 1280,
        "height": 720,
        "extra": {
            "prompt_extend": False,
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
    task_id = 1
    prompt_counter = 1

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

            # Create base experiment folder
            exp_folder = output_dir / f"{prompt_counter:03d}"
            exp_folder.mkdir(parents=True, exist_ok=True)

            # Get fields
            title = row.get('Example Title', '').strip()
            expected_phenomenon = row.get('Expected phenomenon', '').strip()
            fields = row.get('Fields', '').strip()
            keywords = row.get('Keywords', '').strip()
            source = row.get('Source', '').strip()

            # Write shared info file for this prompt
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

            # Create tasks for each model
            for model_config in MODELS:
                # Create model subfolder
                model_folder = exp_folder / model_config["name"]
                model_folder.mkdir(parents=True, exist_ok=True)

                # Write model-specific info file
                model_info = model_folder / "info.txt"
                with open(model_info, 'w', encoding='utf-8') as info_f:
                    info_f.write(f"Model: {model_config['name']}\n")
                    info_f.write(f"Provider: {model_config['provider']}\n")
                    info_f.write(f"Model ID: {model_config['model']}\n")
                    info_f.write(f"Title: {title}\n")
                    info_f.write(f"Author: {author}\n")
                    info_f.write(f"\n=== Prompt ===\n")
                    info_f.write(f"{prompt}\n")
                    info_f.write(f"\n=== Expected Phenomenon ===\n")
                    info_f.write(f"{expected_phenomenon}\n")

                # Add task with model-specific config
                video_file = str(model_folder / "video.mp4")
                
                # Set longer timeout for Sora (needs 3000s)
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

            print(f"Prepared prompt #{prompt_counter} with {len(MODELS)} models: {title if title else prompt[:50]+'...'}")
            prompt_counter += 1

    # Check if any tasks were found
    if not tasks:
        print()
        print(f"Warning: No experiments found for author: {author}")
        return 1

    # Write JSON configuration (no defaults needed since each task has full config)
    config = {
        "tasks": tasks
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"✓ Successfully prepared {len(tasks)} tasks ({prompt_counter-1} prompts × {len(MODELS)} models)")
    print(f"JSON configuration: {output_json}")
    print("=" * 60)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract experiments from CSV and prepare JSON configuration for video generation"
    )
    
    AUTHOR = "Daniel Zhao" # change this to author you want to review
    # Support both positional and named arguments
    parser.add_argument(
        "author",
        nargs='?',
        default=AUTHOR,
        help="Author name to filter experiments"
    )
    parser.add_argument(
        "csv_file",
        nargs='?',
        default="test.csv",
        help="Path to CSV file (default: test.csv)"
    )
    parser.add_argument(
        "output_dir",
        nargs='?',
        default=None,
        help="Output directory (default: out/<author_name>)"
    )
    
    # Also support named arguments for backwards compatibility
    parser.add_argument(
        "--author",
        dest="author_named",
        help="Author name (alternative to positional argument)"
    )
    parser.add_argument(
        "--csv",
        dest="csv_named",
        help="Path to CSV file (alternative to positional argument)"
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir_named",
        help="Output directory (alternative to positional argument)"
    )

    args = parser.parse_args()

    # Prefer named arguments if provided, otherwise use positional
    author = args.author_named if args.author_named else args.author
    csv_file = args.csv_named if args.csv_named else args.csv_file
    output_dir = args.output_dir_named if args.output_dir_named else args.output_dir

    exit_code = prepare_experiments(author, csv_file, output_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
