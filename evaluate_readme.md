# Video Generation Evaluation Pipeline

Quick start guide: from Notion export to video generation.

## Step 1: Export CSV from Notion

1. Open your Notion database (experiment collection)
2. Click the **`···`** menu (top right of the database)
3. Select **Export**
4. Choose format: **CSV**
5. Enable: **Include subpages** (if needed)
6. Click **Export**
7. Download and save as `{data_file_name}.csv` in the project destinated path

**Required CSV Columns:**
- `Example Title` - Experiment name
- `Author` - Author name (used for filtering)
- `Prompts` - Video generation prompt (required)
- `Expected phenomenon` - Description of expected result
- `Fields` - Scientific field(s)
- `Keywords` - Comma-separated keywords
- `Source` - Reference URL

## Step 2: Configure the Script

Edit `scripts/evaluate_by_author_using_csv.sh` and modify the following configuration:

### Set API Keys (Required)

```bash
# API Configuration - Set your API keys directly here
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"     # For Alibaba/Wan models
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"        # For OpenAI/Sora models
export REPLICATE_API_TOKEN="r8_xxxxxxxxxxxxxxxx"   # For Replicate models
```

### Set Experiment Parameters

```bash
# Author to filter (must match exactly with the Author column in CSV)
AUTHOR="Your Name"

# CSV file path (change if using a different file)
CSV_FILE="{data_file_name}.csv"

# Custom your output directory (leave empty for default: out/<author_name>)
OUT_DIR=""

# Number of parallel workers (3-5 recommended, 1-2 for free tier)
WORKERS=5
```

**Examples:**
- Default: `OUT_DIR=""` → outputs to `out/Author_Name/`
- Custom: `OUT_DIR="my_experiments"` → outputs to `my_experiments/`
- Absolute path: `OUT_DIR="/path/to/results"` → outputs to `/path/to/results/`

## Step 3: Run the Script

```bash
./scripts/evaluate_by_author_using_csv.sh
```

The script will automatically:
- Extract experiments for the specified author from CSV
- Create directory structure and info files
- Generate all videos in parallel

## Results Location

Generated videos and related files are located in:

```
out/
└── Author_Name/              # Author name (spaces replaced with underscores)
    ├── experiments.json      # Task configuration
    ├── 001/
    │   ├── info.txt         # Experiment metadata
    │   └── video.mp4        # Generated video
    ├── 002/
    │   ├── info.txt
    │   └── video.mp4
    └── ...
```
