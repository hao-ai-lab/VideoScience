<p align="center">
  <img src="assets/VideoScience-Logo-v1.png" alt="vsci-bench-logo" width="220" align="center">
</p>


<div align="center"><h1>&nbsp;VideoScience-Bench: Benchmarking Scientific Reasoning in Video Generations</h1></div>


<p align="center">
  <a href="https://arxiv.org/pdf/2512.02942">📄 Paper</a> •
  <a href="https://hao-ai-lab.github.io/blogs/videoscience/">📝 Blog</a> •
  <a href="https://huggingface.co/datasets/lmgame/VideoScienceBench">🤗 Dataset</a> •
  <a href="https://huggingface.co/spaces/lmgame/videoscience-bench">🚀 Demo</a>
</p>

---

## What this repo provides

**VideoScience-Bench** evaluates whether video models can go beyond *looking plausible* to *being scientifically correct*.

- **200** undergraduate-level scientific scenarios (physics + chemistry)
  - 160 for T2V evaluation
  - and 40 for I2V evaluation
- **12 topics**, **103 concepts**, and **multi-concept scientific reasoning required** in a single prompt
- Evaluation along **5 dimensions** (Prompt Consistency, Phenomenon Congruency, Correct Dynamism, Immutability, Spatio-Temporal Coherence)

**VideoScience-Judge** is an auto evaluation pipeline that supports:
1) **Prompt-specific checklist** generation
2) **CV-grounded evidence extraction** (e.g., object detection, object tracking, motion tracking)
3) **Salient key frames selection** where scientific phenomena occur
4) final grading with a **reasoning-capable VLM**

---

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Understand Evaluation Metrics](#evaluation-metrics)
- [VideoScience-Judge Results](#videoscience-judge-results)
- [Citation](#citation)
- [License](#license)

---

## Dataset Overview

VideoScience-Bench is curated to stress **scientific reasoning** in video generation: each prompt typically requires **at least 2 interacting scientific concepts** to produce the correct phenomenon.

### Topics (12)

**Physics (7):**
- Classical Mechanics
- Thermodynamics
- Electromagnetism
- Optics
- Fluid Mechanics
- Material Mechanics
- Modern Physics

**Chemistry (5):**
- Redox Reactions
- Acid-Base
- Reaction Kinetics
- Solution and Phase Chemistry
- Materials and Solid-State Chemistry

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
data = ds["test"]

# sanity check an example with
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

### 1) Batched video generation

1. Download the CSV data file under `data/database/data_filtered.jsonl`.
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
