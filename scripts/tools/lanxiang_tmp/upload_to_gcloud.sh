#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./upload_model_videos_to_gcs.sh <BASE_DIR> <AUTHOR> <REMOTE_MODEL_NAME> [--dry-run]
# Example:
#   ./upload_model_videos_to_gcs.sh out/crosseval lanxiang sora-2 --dry-run

BASE_DIR="${1:?base dir required (e.g., out/crosseval)}"
AUTHOR="${2:?author required (e.g., lanxiang)}"
REMOTE_MODEL="${3:?remote model required (e.g., sora-2)}"
DRY_RUN="${4:-}"

# local -> remote mapping
declare -A MAP=(
  ["hailuo2.3"]="minimax-hailuo-2.3"
  ["kling2.5"]="kling-v2-5-turbo-pro"
  ["luma_ray2"]="ray-2"
  ["sora2"]="sora-2"
  ["veo3gen"]="veo3-quality"
  ["wan2.5"]="wan2.5-t2v-preview"
  ["seedance1pro"]="bytedance-seedance-1-pro"
)

# figure out the local model directory name for the requested remote model
LOCAL_MODEL=""
for k in "${!MAP[@]}"; do
  [[ "${MAP[$k]}" == "$REMOTE_MODEL" ]] && LOCAL_MODEL="$k" && break
done
[[ -n "$LOCAL_MODEL" ]] || { echo "Unknown model: $REMOTE_MODEL"; exit 1; }

BASE_DIR="$(realpath -m "$BASE_DIR")"
BUCKET="gs://science_compass/evaluation_videos/$REMOTE_MODEL/$AUTHOR"

echo "Base dir : $BASE_DIR"
echo "Author   : $AUTHOR"
echo "Model    : $REMOTE_MODEL  (local: $LOCAL_MODEL)"
echo "Dest     : $BUCKET"
echo

# collect all mp4s in */<LOCAL_MODEL>/*
mapfile -d '' FILES < <(find "$BASE_DIR" -type f -path "*/$LOCAL_MODEL/*" -name "*.mp4" -print0 | sort -z)

[[ ${#FILES[@]} -gt 0 ]] || { echo "No .mp4 files for $LOCAL_MODEL"; exit 0; }

echo "Uploading ${#FILES[@]} file(s):"
for f in "${FILES[@]}"; do
  f="${f%$'\0'}"
  model_dir="$(dirname "$f")"                               # .../vid_XXX/<LOCAL_MODEL>
  prompt_dir="$(basename "$(dirname "$model_dir")")"        # vid_XXX
  base="$(basename "$f")"
  base="${base#video_}"                                     # strip 'video_' if present
  dest="$BUCKET/${prompt_dir}_${base}"                      # vid_XXX_run_3.mp4

  if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "gcloud storage cp \"$f\" \"$dest\" --no-clobber --quiet --dry-run"
  else
    gcloud storage cp "$f" "$dest" --no-clobber --quiet
  fi
done

echo
echo "Done."
