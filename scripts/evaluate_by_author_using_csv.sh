#!/bin/bash

################################################################################
# Multi-Model Video Generation Evaluation Script
#
# This script prepares experiments from CSV and runs parallel video generation
# across multiple models: Veo3Gen, Luma Ray 2, Wan 2.5, Kling 2.5, Sora-2, 
# Hailuo-2.3, and Seedance-1-Pro.
#
# CONFIGURATION: Edit the variables below to customize
################################################################################

# API Configuration
# Set your API keys directly here
export VEO3GEN_API_KEY=""    # For Veo3Gen
export LUMA_API_KEY=""  # For Luma Ray 2
export DASHSCOPE_API_KEY=""        # For Wan 2.5
export KLING_ACCESS_KEY=""             # For Kling 2.5
export KLING_SECRET_KEY=""             # For Kling 2.5
export OPENAI_API_KEY=""  # For Sora-2
export REPLICATE_API_TOKEN=""  # For Hailuo-2.3 & Seedance-1-Pro

# Experiment Configuration
AUTHOR="Daniel Zhao" # change this to author you want to review
CSV_FILE="/workspace/ScienceAtlas/test.csv"
OUT_DIR="/workspace/ScienceAtlas/out/crosseval"                                          # Custom output directory (leave empty for default: out/<author_name>)
WORKERS=7                                                                                         # Parallel workers (7 models will be generated per prompt)

################################################################################
# Do not edit below this line unless you know what you're doing
################################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set output directory
if [ -z "$OUT_DIR" ]; then
    OUTPUT_DIR="$PROJECT_ROOT/out/${AUTHOR// /_}"
else
    OUTPUT_DIR="$OUT_DIR"
fi

EXPERIMENTS_JSON="$OUTPUT_DIR/experiments.json"

echo "============================================================"
echo "Video Generation Evaluation Pipeline"
echo "============================================================"
echo "Author: $AUTHOR"
echo "CSV File: $CSV_FILE"
echo "Workers: $WORKERS"
echo "Output Directory: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Step 1: Prepare experiments
echo "Step 1/2: Preparing experiments..."
echo "------------------------------------------------------------"
if [ -z "$OUT_DIR" ]; then
    python3 "$SCRIPT_DIR/prepare_experiments.py" "$AUTHOR" "$CSV_FILE"
else
    python3 "$SCRIPT_DIR/prepare_experiments.py" "$AUTHOR" "$CSV_FILE" "$OUT_DIR"
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Failed to prepare experiments"
    exit 1
fi

echo ""

# Step 2: Run video generation
echo "Step 2/2: Running parallel video generation..."
echo "------------------------------------------------------------"

# Check if experiments.json exists
if [ ! -f "$EXPERIMENTS_JSON" ]; then
    echo "Error: Configuration file not found: $EXPERIMENTS_JSON"
    exit 1
fi

# Count tasks
task_count=$(grep -c '"id":' "$EXPERIMENTS_JSON")
echo "Found $task_count tasks to process"
echo ""

# Run the async frontend
cd "$PROJECT_ROOT"
python3 frontend.py --json "$EXPERIMENTS_JSON" --workers "$WORKERS"

exit_code=$?

echo ""
echo "============================================================"
echo "Pipeline Complete"
echo "============================================================"
echo "Author: $AUTHOR"
echo "Total experiments: $task_count"
if [ $exit_code -eq 0 ]; then
    echo "Status: ✓ All tasks completed successfully"
else
    echo "Status: ✗ Some tasks failed (check output above)"
fi
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

exit $exit_code
