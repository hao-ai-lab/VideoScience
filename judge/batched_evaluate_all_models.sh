#!/usr/bin/env bash
set -Eeuo pipefail

# --------------------------------------------------------------------
# batched_evaluate.sh
# Usage:
#   bash judge/batched_evaluate.sh <csv> <authors> [results_dir] [source_dir] [reference_dir] [checklist_root]
#
# Positional args:
#   csv            - path to experiments CSV
#   authors        - exact author or comma-separated authors filter
#   results_dir    - (optional) default: judge/results/evaluation_videos
#   source_dir     - (optional) default: judge/data/evaluation_videos
#   reference_dir  - (optional) default: judge/data/reference_videos
#   checklist_root - (optional) root dir for generated checklists (default: generated_checklists)
#
# Env vars (optional):
#   PROVIDER             - API provider (default: openai)
#   MODEL                - Judge model id (default: gpt-5-pro)
#   MODE                 - ready | all   (default: ready)
#   MAX_RUNS             - hard cap across all runs per target (0 = no cap; default: 0)
#   JUDGE_SCRIPT         - Path to vlm_as_a_judge.py (default: judge/vlm_as_a_judge.py)
#   PYTHON_BIN           - Python executable (default: python3)
#   LOG_DIR              - Directory for logs (default: logs)
#   CHECKLIST_ROOT       - Root directory for checklists (default: judge/checklist/generated_checklists, if no positional arg)
#   CHECKLIST_SCRIPT     - Path to generate_checklists.py (default: judge/checklist/generate_checklists.py)
#   CHECKLIST_MODEL      - Gemini model name for checklist generation (default: models/gemini-flash-latest)
#   CHECKLIST_MAX_ENTRIES - Max checklists to generate (0 = no limit; default: 0)
#
#   ENABLE_CV            - If set to non-zero, enable CV metrics integration (default: 0 = off)
#   CV_METRICS_SCRIPT    - Path to cv_metrics.py (default: judge/cv_tools/cv_metrics.py)
#   CV_MODULES           - Comma-separated CV modules (default: grounding,bytetrack,raft,clip4clip,lpips)
# --------------------------------------------------------------------

CSV="${1:?CSV path required}"
AUTHOR="${2:?Authors required}"
BASE_OUT_DIR="${3:-judge/results/evaluation_videos_checklist_cv}"
SOURCE_DIR="${4:-judge/data/evaluation_videos}"
REFERENCE_DIR="${5:-judge/data/reference_videos}"
CHECKLIST_ROOT_ARG="${6-}"

PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-gpt-5-pro}"
MODE="${MODE:-ready}"
MAX_RUNS="${MAX_RUNS:-0}"
JUDGE_SCRIPT="${JUDGE_SCRIPT:-judge/vlm_as_a_judge.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG_DIR="${LOG_DIR:-logs}"

# Checklist plumbing
if [[ -n "$CHECKLIST_ROOT_ARG" ]]; then
  CHECKLIST_ROOT="$CHECKLIST_ROOT_ARG"
else
  CHECKLIST_ROOT="${CHECKLIST_ROOT:-judge/checklist/generated_checklists}"
fi
CHECKLIST_SCRIPT="${CHECKLIST_SCRIPT:-judge/checklist/generate_checklists.py}"
CHECKLIST_MODEL="${CHECKLIST_MODEL:-models/gemini-flash-latest}"
CHECKLIST_MAX_ENTRIES="${CHECKLIST_MAX_ENTRIES:-0}"

# Match generate_checklists.py: author.replace(" ", "_")
CHECKLIST_AUTHOR_SUBDIR="${AUTHOR// /_}"
CHECKLIST_DIR="${CHECKLIST_ROOT}/${CHECKLIST_AUTHOR_SUBDIR}"

# CV metrics plumbing (for extract_exp_info --enable-cv)
ENABLE_CV="${ENABLE_CV:-1}"
CV_METRICS_SCRIPT="${CV_METRICS_SCRIPT:-judge/cv_tools/cv_metrics.py}"
#CV_MODULES="${CV_MODULES:-grounding,bytetrack,raft,clip4clip,lpips}"
CV_MODULES="${CV_MODULES:-grounding,bytetrack,raft,clip4clip}"

# Target model folders to scan under SOURCE_DIR/
TGT_MODELS=(
  "bytedance-seedance-1-pro"
  "kling-v2-5-turbo-pro"
  "minimax-hailuo-2.3"
  "ray-2"
  "sora-2"
  "veo3-quality"
  "wan2.5-t2v-preview"
)

slug() {
  # lower, replace non [a-z0-9._-] with '-', collapse dashes, trim
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/-+/-/g; s/^-+//; s/-+$//'
}

AUTHOR_SLUG="$(slug "$AUTHOR")"

mkdir -p "$BASE_OUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKLIST_DIR"

# Export so extract_exp_info and runner scripts can use it (for --checklist_json)
export CHECKLIST_DIR

echo "[batched_evaluate] CSV:              $CSV"
echo "[batched_evaluate] Results dir:      $BASE_OUT_DIR"
echo "[batched_evaluate] Provider:         $PROVIDER"
echo "[batched_evaluate] Judge Model:      $MODEL"
echo "[batched_evaluate] Source dir:       $SOURCE_DIR"
echo "[batched_evaluate] Reference dir:    $REFERENCE_DIR"
echo "[batched_evaluate] Mode:             $MODE"
echo "[batched_evaluate] Author filter:    $AUTHOR"
echo "[batched_evaluate] Max runs cap:     $MAX_RUNS"
echo "[batched_evaluate] Judge script:     $JUDGE_SCRIPT"
echo "[batched_evaluate] Python:           $PYTHON_BIN"
echo "[batched_evaluate] Log dir:          $LOG_DIR"
echo "[batched_evaluate] Checklist script: $CHECKLIST_SCRIPT"
echo "[batched_evaluate] Checklist root:   $CHECKLIST_ROOT"
echo "[batched_evaluate] Checklist dir:    $CHECKLIST_DIR"
echo "[batched_evaluate] Checklist model:  $CHECKLIST_MODEL"
echo "[batched_evaluate] Checklist max:    $CHECKLIST_MAX_ENTRIES"
echo "[batched_evaluate] ENABLE_CV:        $ENABLE_CV"
echo "[batched_evaluate] CV_METRICS_SCRIPT:$CV_METRICS_SCRIPT"
echo "[batched_evaluate] CV_MODULES:       $CV_MODULES"
echo

# --------------------------------------------------------------------
# 1) Generate / refresh checklists for this author from the CSV
# --------------------------------------------------------------------
# (left commented out as in your current version)
# echo "==> [checklists] Generating checklists for author '$AUTHOR' from CSV: $CSV"
# ${PYTHON_BIN} "$CHECKLIST_SCRIPT" \
#   --csv_file "$CSV" \
#   --output_folder "$CHECKLIST_ROOT" \
#   --author "$AUTHOR" \
#   --model "$CHECKLIST_MODEL" \
#   --max_entries "$CHECKLIST_MAX_ENTRIES"
# echo "==> [checklists] Done."
# echo

run_one_target() {
  local target="$1"
  local target_slug
  target_slug="$(slug "$target")"

  local runner="$BASE_OUT_DIR/run_checklist_cv-method_${AUTHOR_SLUG}_target_${target_slug}.sh"
  local log_path="$LOG_DIR/run_checklist_cv-method_${AUTHOR_SLUG}_${target_slug}.log"

  # Build CV-related args for extract_exp_info if ENABLE_CV is set
  local cv_args=()
  if [[ "$ENABLE_CV" != "0" ]]; then
    cv_args+=(--enable-cv)
    cv_args+=(--cv-metrics-script "$CV_METRICS_SCRIPT")
    cv_args+=(--cv-modules "$CV_MODULES")
  fi

  echo "==> [generate] target: $target"
  ${PYTHON_BIN} judge/extract_exp_info.py \
    --csv "$CSV" \
    --source-dir "$SOURCE_DIR" \
    --reference-dir "$REFERENCE_DIR" \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --out-dir "$BASE_OUT_DIR" \
    --script "$JUDGE_SCRIPT" \
    --mode "$MODE" \
    --author "$AUTHOR" \
    --max-runs "$MAX_RUNS" \
    --checklist-dir "$CHECKLIST_DIR" \
    "${cv_args[@]}" \
    --eval_target "$target"

  # The extractor names the runner as:
  #   run_checklist_cv-method_{author}_target_{target_model_name}.sh
  if [[ ! -f "$runner" ]]; then
    # Fallback: find the most recent matching file (handles minor naming/spacing issues)
    candidate="$(ls -1t "$BASE_OUT_DIR"/run_checklist_cv-method_"$AUTHOR_SLUG"_target_"$target_slug".sh 2>/dev/null | head -n1 || true)"
    [[ -n "$candidate" ]] && runner="$candidate"
  fi

  if [[ ! -f "$runner" ]]; then
    echo "[warn] No runner created for target '$target' (runner missing: $runner). Skipping."
    return 0
  fi

  chmod +x "$runner"

  # Quick count
  if command -v grep >/dev/null 2>&1; then
    COUNT="$(grep -E '^[[:space:]]*run[[:space:]]' -c "$runner" || true)"
    echo "    Planned runs for $target: $COUNT"
  fi

  echo "==> [execute] $runner"
  echo "    (log: $log_path)"
  (
    echo "[start] $(date) - $target"
    MAX_RUNS="$MAX_RUNS" bash "$runner"
    echo "[end] $(date) - $target"
  ) >"$log_path" 2>&1 &
  PIDS+=($!)
  LOGS+=("$log_path")
}

declare -a PIDS=()
declare -a LOGS=()

for tgt in "${TGT_MODELS[@]}"; do
  run_one_target "$tgt"
done

# Wait for all background targets
fail=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  log="${LOGS[$i]}"
  if ! wait "$pid"; then
    echo "[error] Target process (pid=$pid) failed. See log: $log"
    fail=1
  else
    echo "[ok] Target process (pid=$pid) completed. Log: $log"
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[batched_evaluate] Completed with errors. Check logs under: $LOG_DIR"
  exit 1
fi

echo
echo "[batched_evaluate] All targets completed successfully. Logs in: $LOG_DIR"
