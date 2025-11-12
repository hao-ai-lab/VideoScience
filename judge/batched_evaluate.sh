#!/usr/bin/env bash
set -Eeuo pipefail

# --------------------------------------------------------------------
# run_from_csv.sh
# Usage:
#   bash judge/run_from_csv.sh <csv> <authors> <eval_target> [results_dir] [source_dir] [reference_dir]
#
# Positional args:
#   csv           - path to experiments CSV
#   authors       - exact author or comma-separated authors filter
#   eval_target   - subdirectory under SOURCE_DIR to scan (e.g., sora2)
#   results_dir   - (optional) default: judge/results/evaluation_videos
#   source_dir    - (optional) default: judge/data/evaluation_videos
#   reference_dir - (optional) default: judge/data/reference_videos
#
# Env vars (optional):
#   PROVIDER       - API provider (default: openai)
#   MODEL          - Model id (default: gpt-5-pro)
#   MODE           - ready | all   (default: ready)
#   MAX_RUNS       - hard cap across all runs (0 = no cap; default: 3)
#   JUDGE_SCRIPT   - Path to vlm_as_a_judge.py (default: judge/vlm_as_a_judge.py)
#   PYTHON         - Python executable (default: python3)
# --------------------------------------------------------------------

CSV="${1:?CSV path required}"
AUTHOR="${2:?Authors required}"
EVAL_TARGET="${3:?Eval target (subdir under SOURCE_DIR) required}"
BASE_OUT_DIR="${4:-judge/results/evaluation_videos}"
SOURCE_DIR="${5:-judge/data/evaluation_videos}"
REFERENCE_DIR="${6:-judge/data/reference_videos}"

PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-gpt-5-pro}"
MODE="${MODE:-ready}"
MAX_RUNS="${MAX_RUNS:-0}"
JUDGE_SCRIPT="${JUDGE_SCRIPT:-judge/vlm_as_a_judge.py}"

if [[ -z "$SOURCE_DIR" ]]; then
  echo "[run_from_csv] ERROR: SOURCE_DIR is required for the new extractor." >&2
  exit 1
fi

mkdir -p "$BASE_OUT_DIR"
RUNNER="$BASE_OUT_DIR/run_all.sh"

echo "[run_from_csv] CSV:            $CSV"
echo "[run_from_csv] Results dir:    $BASE_OUT_DIR"
echo "[run_from_csv] Provider:       $PROVIDER"
echo "[run_from_csv] Model:          $MODEL"
echo "[run_from_csv] Source dir:     $SOURCE_DIR"
echo "[run_from_csv] Eval target:    $EVAL_TARGET"
[[ -n "$REFERENCE_DIR" ]] && echo "[run_from_csv] Reference dir: $REFERENCE_DIR" || true
echo "[run_from_csv] Mode:           $MODE"
[[ -n "$AUTHOR" ]] && echo "[run_from_csv] Author filter:  $AUTHOR" || true
echo "[run_from_csv] Max runs cap:   $MAX_RUNS"
echo "[run_from_csv] Judge script:   $JUDGE_SCRIPT"
echo

# 1) Generate the run script from the CSV (exact-column extraction)
python3 judge/extract_exp_info.py \
  --csv "$CSV" \
  --source-dir "$SOURCE_DIR" \
  ${REFERENCE_DIR:+--reference-dir "$REFERENCE_DIR"} \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --out-dir "$BASE_OUT_DIR" \
  --script "$JUDGE_SCRIPT" \
  --mode "$MODE" \
  ${AUTHOR:+--author "$AUTHOR"} \
  ${MAX_RUNS:+--max-runs "$MAX_RUNS"} \
  "$EVAL_TARGET"

# If the generator placed run_all.sh in CWD, move it into the results dir
if [[ -f "./run_all.sh" ]]; then
  mv -f "./run_all.sh" "$RUNNER"
elif [[ -f "$RUNNER" ]]; then
  : # already written directly to BASE_OUT_DIR
else
  echo "[run_from_csv] ERROR: run_all.sh not found after generation." >&2
  exit 1
fi

chmod +x "$RUNNER"

# 2) Quick count (match lines that start with 'run ' regardless of quoting)
if command -v grep >/dev/null 2>&1; then
  COUNT="$(grep -E '^[[:space:]]*run[[:space:]]' -c "$RUNNER" || true)"
  echo "[run_from_csv] Planned runs: $COUNT"
  echo
fi

# 3) Execute the generated runner (the script enforces MAX_RUNS internally)
echo "[run_from_csv] Executing: $RUNNER"
export MAX_RUNS
bash "$RUNNER"

echo
echo "[run_from_csv] All runs completed."
