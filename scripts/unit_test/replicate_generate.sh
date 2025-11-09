#!/usr/bin/env bash
set -euo pipefail

# --- CONFIGURATION ---
PROVIDER="replicate"
#MODEL="google/veo-3"
MODEL="minimax/video-01"
PROMPT="a tiny robot chef flipping pancakes in zero gravity, soft cinematic lighting"
SECONDS=8
WIDTH=1280
HEIGHT=720
#OUTPUT_PATH="out/replicate-google-veo3/replicate_veo3_1280x720_8s.mp4"
OUTPUT_PATH="out/replicate-minimax/replicate_hailuo_1280x720_8s.mp4"
EXTRA='{"guidance":7.5, "seed":12345}'  # model-specific parameters
# ----------------------

echo "Running single_generation_frontend.py with provider=$PROVIDER, model=$MODEL"

python3 single_generation_frontend.py \
  --provider "${PROVIDER}" \
  --model "${MODEL}" \
  --prompt "${PROMPT}" \
  --seconds ${SECONDS} \
  --width ${WIDTH} \
  --height ${HEIGHT} \
  --out ${OUTPUT_PATH} \
  --extra "${EXTRA}"

