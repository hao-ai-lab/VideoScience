#!/usr/bin/env python3
"""
Prepare FastVideo experiments for VideoScience evaluation.

This script reads prompts from the VideoScience data file and creates
experiment configurations for FastVideo models.

Supported models:
  - FastWan2.1-T2V-1.3B-Diffusers (distilled, 3-step inference)
  - Wan2.1-T2V-1.3B-Diffusers (base model, standard inference)

Usage:
  python prepare_fastvideo_experiments.py \
      --data-file data/database/data_filtered.jsonl \
      --output-dir out/fastvideo_eval \
      --models fastwan-1.3b \
      --num-runs 3
"""

import argparse
import json
import sys
from pathlib import Path


# FastVideo model configurations
FASTVIDEO_MODELS = {
    "fastwan-1.3b": {
        "name": "fastwan-1.3b",
        "provider": "fastvideo",
        "model": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        "seconds": 5,  # 81 frames at 16fps ≈ 5s
        "width": 832,
        "height": 480,
        "extra": {
            "num_frames": 81,
            "num_inference_steps": 3,  # DMD distilled - 3 steps
            "fps": 16,
            "seed": 1024,
            "negative_prompt": (
                "Bright tones, overexposed, static, blurred details, subtitles, "
                "style, works, paintings, images, static, overall gray, worst quality, "
                "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
                "misshapen limbs, fused fingers, still picture, messy background, "
                "three legs, many people in the background, walking backwards"
            ),
        }
    },
    "wan-1.3b": {
        "name": "wan-1.3b",
        "provider": "fastvideo",
        "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "seconds": 5,
        "width": 832,
        "height": 480,
        "extra": {
            "num_frames": 81,
            "num_inference_steps": 50,  # Standard inference steps
            "fps": 16,
            "seed": 1024,
            "negative_prompt": (
                "Bright tones, overexposed, static, blurred details, subtitles, "
                "style, works, paintings, images, static, overall gray, worst quality, "
                "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
                "misshapen limbs, fused fingers, still picture, messy background, "
                "three legs, many people in the background, walking backwards"
            ),
        }
    },
}


def load_prompts_from_jsonl(data_file: Path) -> list[dict]:
    """Load prompts from VideoScience data file."""
    prompts = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get('prompt') and data.get('vid'):
                    prompts.append({
                        'vid': data['vid'],
                        'prompt': data['prompt'],
                        'expected_phenomenon': data.get('expected phenomenon', ''),
                        'keywords': data.get('keywords', []),
                        'field': data.get('field', ''),
                    })
            except json.JSONDecodeError:
                continue
    return prompts


def prepare_experiments(
    data_file: str,
    output_dir: str,
    models: str,
    num_runs: int,
    skip_existing: bool = True,
):
    """Prepare FastVideo experiments."""
    
    data_path = Path(data_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which models to evaluate
    if models == "both":
        model_configs = [FASTVIDEO_MODELS["fastwan-1.3b"], FASTVIDEO_MODELS["wan-1.3b"]]
    elif models in FASTVIDEO_MODELS:
        model_configs = [FASTVIDEO_MODELS[models]]
    else:
        print(f"Error: Unknown model '{models}'. Choose from: fastwan-1.3b, wan-1.3b, both")
        return 1
    
    print(f"Loading prompts from: {data_file}")
    prompts = load_prompts_from_jsonl(data_path)
    print(f"Loaded {len(prompts)} prompts")
    
    if not prompts:
        print("Error: No prompts found in data file")
        return 1
    
    tasks = []
    skipped_count = 0
    task_id = 1
    
    for prompt_data in prompts:
        vid = prompt_data['vid']
        prompt = prompt_data['prompt']
        
        # Create prompt directory
        prompt_dir = output_path / f"vid_{int(vid):03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        
        # Write info file
        info_file = prompt_dir / "info.txt"
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"Video ID: {vid}\n")
            f.write(f"Field: {prompt_data['field']}\n")
            f.write(f"Keywords: {', '.join(prompt_data['keywords'])}\n")
            f.write(f"\n=== Prompt ===\n{prompt}\n")
            f.write(f"\n=== Expected Phenomenon ===\n{prompt_data['expected_phenomenon']}\n")
        
        # Create tasks for each model and run
        for model_config in model_configs:
            model_name = model_config["name"]
            model_dir = prompt_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            for run in range(1, num_runs + 1):
                if num_runs == 1:
                    video_file = model_dir / "video.mp4"
                else:
                    video_file = model_dir / f"video_run{run}.mp4"
                
                # Skip if video already exists
                if skip_existing and video_file.exists():
                    skipped_count += 1
                    task_id += 1
                    continue
                
                # Copy extra config and update seed for different runs
                extra = dict(model_config["extra"])
                if num_runs > 1:
                    extra["seed"] = model_config["extra"].get("seed", 1024) + run - 1
                
                tasks.append({
                    "id": task_id,
                    "prompt": prompt,
                    "provider": model_config["provider"],
                    "model": model_config["model"],
                    "seconds": model_config["seconds"],
                    "width": model_config["width"],
                    "height": model_config["height"],
                    "extra": extra,
                    "output_path": str(video_file),
                    "timeout_s": 600,  # 10 minutes timeout per video
                })
                task_id += 1
        
        print(f"Prepared vid_{int(vid):03d}: {prompt[:60]}...")
    
    # Write experiments.json
    experiments_json = output_path / "experiments.json"
    with open(experiments_json, 'w', encoding='utf-8') as f:
        json.dump({"tasks": tasks}, f, indent=2, ensure_ascii=False)
    
    total_possible = len(prompts) * len(model_configs) * num_runs
    print()
    print("=" * 60)
    print(f"✓ Successfully prepared {len(tasks)} tasks")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Models: {len(model_configs)}")
    print(f"  Runs per prompt: {num_runs}")
    print(f"  Skipped (already exist): {skipped_count}")
    print(f"  Remaining to generate: {len(tasks)}")
    print(f"JSON configuration: {experiments_json}")
    print("=" * 60)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FastVideo experiments for VideoScience evaluation"
    )
    parser.add_argument(
        "--data-file",
        default="data/database/data_filtered.jsonl",
        help="Path to VideoScience data file (JSONL format)"
    )
    parser.add_argument(
        "--output-dir",
        default="out/fastvideo_eval",
        help="Output directory for experiments"
    )
    parser.add_argument(
        "--models",
        choices=["fastwan-1.3b", "wan-1.3b", "both"],
        default="fastwan-1.3b",
        help="Which FastVideo models to evaluate"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of generation runs per prompt (for variance analysis)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing videos (regenerate all)"
    )
    
    args = parser.parse_args()
    
    exit_code = prepare_experiments(
        data_file=args.data_file,
        output_dir=args.output_dir,
        models=args.models,
        num_runs=args.num_runs,
        skip_existing=not args.no_skip,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
