# ScienceAtlas

## Installation

### Basic Setup

```bash
# Clone the repository
git clone https://github.com/your-org/ScienceAtlas.git
cd ScienceAtlas

# Install dependencies
pip install -r requirements.txt
```

### FastVideo Setup

FastVideo is a video generation provider that supports two modes of operation:

#### Option 1: Remote API Server (Recommended for Production)

If you have a deployed FastVideo API server:

```bash
export FASTVIDEO_API_BASE="http://your-fastvideo-server:8000"
export FASTVIDEO_API_KEY="your-api-key"  # Optional, if authentication is required
```

#### Option 2: Local Inference Mode

For local GPU inference:

```bash
# Install FastVideo package
pip install fastvideo

# Set the model path (will be downloaded on first use)
export FASTVIDEO_MODEL_PATH="FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
```

**Requirements for local inference:**
- CUDA-capable GPU with sufficient VRAM
- PyTorch with CUDA support

## Usage

### launch batched video generation
1. Download csv data file under `data/database/data.csv`.
2. Launch script:
```
bash scripts/batched_generation_using_csv.sh
```

### launch single video generation
1. Launch script:
```
python3 single_generation_frontend.py --provider {provider_name} --model {model_name} --prompt {customized_prompt}
```

### VLM as a judge
```
bash judge/batched_evaluate_all_models.sh
```
