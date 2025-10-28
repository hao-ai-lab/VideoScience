import argparse
import json

import os, sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
    
from backend.api_manager import generate_video

def main():
    parser = argparse.ArgumentParser(description="Text-to-video CLI (Replicate, Sora).")
    parser.add_argument("--provider", required=True,
                        help="replicate | sora-azure | sora-openai")
    parser.add_argument("--model", required=True,
                        help="Provider-specific model identifier "
                             "(e.g. 'google/veo-3-fast' on Replicate, 'sora' on Sora)")
    parser.add_argument("--prompt", required=True, help="Text prompt for generation.")
    parser.add_argument("--seconds", type=int, default=5, help="Video duration (seconds).")
    parser.add_argument("--width", type=int, default=720, help="Output width.")
    parser.add_argument("--height", type=int, default=720, help="Output height.")
    parser.add_argument("--out", default="output.mp4", help="Path to save video (if file output).")
    parser.add_argument("--extra", default="{}", help="JSON of extra provider/model params.")
    parser.add_argument("--timeout_s", type=int, default=1200, help="Max wait (polling) seconds.")

    args = parser.parse_args()
    extra = json.loads(args.extra)

    result = generate_video(
        provider=args.provider,
        model=args.model,
        prompt=args.prompt,
        n_seconds=args.seconds,
        width=args.width,
        height=args.height,
        output_path=args.out,
        timeout_s=args.timeout_s,
        extra=extra,
    )

    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
