import argparse
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from backend.api_manager import generate_video


def generate_video_wrapper(task):
    """Wrapper function to call generate_video with task parameters"""
    try:
        print(f"\n[Task {task['id']}] Starting generation...")

        result = generate_video(
            provider=task['provider'],
            model=task['model'],
            prompt=task['prompt'],
            n_seconds=task['seconds'],
            width=task['width'],
            height=task['height'],
            output_path=task['output_path'],
            timeout_s=task['timeout_s'],
            extra=task['extra'],
        )

        print(f"\n[Task {task['id']}]; Completed: {task['output_path']}")
        return {
            'id': task['id'],
            'status': 'success',
            'output_path': task['output_path'],
            'result': result
        }
    except Exception as e:
        print(f"\n[Task {task['id']}]; Failed: {str(e)}")
        return {
            'id': task['id'],
            'status': 'error',
            'output_path': task['output_path'],
            'error': str(e)
        }


def process_tasks_parallel(tasks, max_workers=5):
    """Process multiple video generation tasks in parallel"""
    print(f"\n=== Starting parallel processing of {len(tasks)} tasks ===")
    print(f"Max parallel workers: {max_workers}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(generate_video_wrapper, tasks))

    return results


def main():
    parser = argparse.ArgumentParser(description="Async text-to-video CLI for batch processing.")
    parser.add_argument("--json", required=True, help="Path to JSON file with tasks.")
    parser.add_argument("--workers", type=int, default=5, help="Max parallel workers.")

    args = parser.parse_args()

    # Load tasks from JSON file
    with open(args.json, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Get global defaults
    defaults = config.get('defaults', {})
    tasks_data = config.get('tasks', [])

    # Build task list with merged defaults
    tasks = []
    for i, task_data in enumerate(tasks_data):
        task = {
            'id': task_data.get('id', i + 1),
            'provider': task_data.get('provider', defaults.get('provider', 'wan')),
            'model': task_data.get('model', defaults.get('model', 'wan2.5-t2v-preview')),
            'prompt': task_data['prompt'],  # Required
            'seconds': task_data.get('seconds', defaults.get('seconds', 8)),
            'width': task_data.get('width', defaults.get('width', 1280)),
            'height': task_data.get('height', defaults.get('height', 720)),
            'output_path': task_data['output_path'],  # Required
            'timeout_s': task_data.get('timeout_s', defaults.get('timeout_s', 1200)),
            'extra': task_data.get('extra', defaults.get('extra', {})),
        }
        tasks.append(task)

    # Process tasks in parallel
    results = process_tasks_parallel(tasks, max_workers=args.workers)

    # Print summary
    print("\n" + "="*60)
    print("=== SUMMARY ===")
    print("="*60)
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")

    if error_count > 0:
        print("\nFailed tasks:")
        for r in results:
            if r['status'] == 'error':
                print(f"  Task {r['id']}: {r['error']}")

    print("="*60)


if __name__ == "__main__":
    main()
