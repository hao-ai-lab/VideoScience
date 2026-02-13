<div align="center"><h1>&nbsp;VideoScience-Bench: Benchmarking Scientific Reasoning in Video Generations</h1></div>




<p align="center">
  <a href="https://arxiv.org/pdf/2512.02942">📄 Paper</a> •
  <a href="https://hao-ai-lab.github.io/blogs/videoscience/">📝 Blog</a> •
  <a href="https://huggingface.co/datasets/lmgame/VideoScienceBench">🤗 Dataset</a> •
  <a href="https://huggingface.co/spaces/lmgame/videoscience-bench">🚀 Demo</a>
</p>

---

## What this repo provides

**VideoScience-Bench** evaluates whether video generation models can go beyond *looking plausible* to *being scientifically correct*.

- **200** curated undergraduate-level scientific scenarios (physics + chemistry)
- **14 topics**, **103 concepts**, and **multi-concept “cascading effects”** in a single prompt
- Evaluation along **5 dimensions** (Prompt Consistency, Phenomenon Congruency, Correct Dynamism, Immutability, Spatio-Temporal Coherence)

**VideoScience-Judge** is a scalable evaluation pipeline that:
1) generates a **prompt-specific checklist**,  
2) selects **key frames / salient moments**,  
3) extracts **CV-grounded evidence** (e.g., tracks, motion, color changes), and  
4) uses a **reasoning-capable VLM** to grade against the checklist.

---

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [VideoScience-Judge vs. Human Annotations](#videoscience-judge-vs-human-annotations)
- [Citation](#citation)
- [License](#license)

---

## Dataset Overview

VideoScience-Bench is curated to stress **scientific reasoning** in video generation: each prompt typically requires **≥2 interacting scientific concepts** to produce the correct phenomenon (not just “everyday plausibility”).

### Topics (14)

**Physics (9):**
- Classical Mechanics
- Optics
- Thermodynamics
- Fluid Mechanics
- Electromagnetism
- Wave
- Energy
- Material Mechanics
- Modern Physics

**Chemistry (5):**
- Redox Reactions
- Liquid Chemistry
- Acid-Base
- Reaction Kinetics
- Material Chemistry

### What each example contains

The prompt suite is lightweight and easy to integrate into any video generation harness.

Common fields (as in the HF release):
- `prompt`: the experimental setup + procedure
- `expected phenomenon`: a concise description of what should happen if the laws are obeyed
- `keywords`: fine-grained scientific concepts involved
- `field`: Physics / Chemistry
- `vid`: instance id

### Loading from Hugging Face

```python
from datasets import load_dataset

ds = load_dataset("lmgame/VideoScienceBench")
data = ds["test"]  # 160 prompts in the HF release
print(data[0]["prompt"])
print(data[0]["expected phenomenon"])
print(data[0]["keywords"])
```

---

## Installation

### Basic Setup

```bash
# Clone the repository
git clone https://github.com/hao-ai-lab/VideoScience.git
cd VideoScience

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

---

## Usage

### 1) Batched video generation (from CSV)

1. Download the CSV data file under `data/database/data.csv`.
2. Launch the script:

```bash
bash scripts/batched_generation_using_csv.sh
```

### 2) Single video generation

```bash
python3 single_generation_frontend.py \
  --provider {provider_name} \
  --model {model_name} \
  --prompt "{your_prompt}"
```

### 3) VLM-as-a-judge evaluation

```bash
bash judge/batched_evaluate_all_models.sh
```

---

## Evaluation Metrics

We evaluate each generated video on **five dimensions** (Likert **1–4**):

- **Prompt Consistency (PCS)**: is the setup/procedure faithful to the prompt?
- **Phenomenon Congruency (PCG)**: does the correct scientific outcome occur?
- **Correct Dynamism (CDN)**: are motions / dynamics physically consistent?
- **Immutability (IMB)**: are static attributes preserved (no flicker/identity drift)?
- **Spatio-Temporal Coherence (STC)**: is the video coherent over time and space?

A commonly used weighted aggregate (paper setting):
- PCG **0.30**, PCS **0.20**, STC **0.20**, CDN **0.15**, IMB **0.15**

---

## VideoScience-Judge vs. Human Annotations

Manual scientific evaluation is expensive. VideoScience-Judge aims to be human expert-aligned while remaining scalable.

### Ranking correlation with expert ratings

We report ranking correlations between automatic metrics and **domain-expert annotations** across 7 evaluated video models.

| Metric | Kendall τ | Spearman ρ |
|---|---:|---:|
| **VSci-Judge** | **0.81** | **0.89** |
| **VSci-Judge (Checklist)** | **0.90** | **0.96** |
| **VSci-Judge (Checklist + CV evidence)** | **0.90** | **0.96** |
| PhyGenEval | 0.52 | 0.61 |
| VideoScore2 | 0.24 | 0.29 |

> Note: adding prompt-specific checklists (and optional CV evidence) makes the judge align **near-perfectly** with expert-ranked model quality on VideoScience-Bench.

### VideoScience-Judge Features

1. **[optional] Checklist generation**: create an evaluative rubric tied to the prompt
2. **[optional] CV-based evidence extraction** (optional but recommended): tracking, motion, attribute changes, key frames
3. **final grading**: VLM-as-a-judge reasons over the checklist + all evidences

---

## Citation

If you use VideoScience in your research, please cite:

```bibtex
@article{hu2025videoscience,
  title={Benchmarking Scientific Understanding and Reasoning for Video Generation using VideoScience-Bench},
  author={Hu, Lanxiang and Shankarampeta, Abhilash and Huang, Yixin and Dai, Zilin and Yu, Haoyang and Zhao, Yujie and Kang, Haoqiang and Zhao, Daniel and Rosing, Tajana and Zhang, Hao},
  journal={arXiv preprint arXiv:2512.02942},
  year={2025}
}
```

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE).
