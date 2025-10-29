#!/bin/bash

################################################################################
# Video Generation Evaluation Script
#
# This script prepares experiments from CSV and runs parallel video generation.
#
# CONFIGURATION: Edit the variables below to customize
################################################################################

# API Configuration
# Set your API keys directly here
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"     # For Alibaba/Wan models
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"        # For OpenAI/Sora models
export REPLICATE_API_TOKEN="r8_xxxxxxxxxxxxxxxx"   # 
AUTHOR="Yujie Zhao"
CSV_FILE="test.csv"
WORKERS=

################################################################################
# Do not edit below this line unless you know what you're doing
################################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/out/${AUTHOR// /_}"
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
python3 "$SCRIPT_DIR/prepare_experiments.py" "$AUTHOR" "$CSV_FILE"

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
python3 frontend_async.py --json "$EXPERIMENTS_JSON" --workers "$WORKERS"

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
