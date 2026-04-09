# CounterVQA: Evaluating and Improving Counterfactual Reasoning in Vision-Language Models for Video Understanding

<!-- <p align="center">
  <img src="assets/teaser.png" width="95%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/CVPR-2026-blue.svg" alt="CVPR 2026"></a>
  <a href="https://huggingface.co/datasets/xxx/CounterVQA"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow" alt="Dataset"></a>
  <a href="https://huggingface.co/xxx/CFGPT"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow" alt="Model"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p> -->

<p align="center">
  <b>Yuefei Chen<sup>1</sup>, Jiang Liu<sup>2</sup>, Xiaodong Lin<sup>1</sup>, Ruixiang Tang<sup>1†</sup></b><br>
  <sup>1</sup>Rutgers University &nbsp; <sup>2</sup>Advanced Micro Devices<br>
  <sup>†</sup>Corresponding author
</p>

---

## 🔥 News

- **[2026/03]** CounterVQA is accepted to **CVPR 2026**!
- Dataset & Code will be released.


---

## 📖 Overview

**CounterVQA** is the first video benchmark that systematically evaluates counterfactual reasoning in Vision-Language Models (VLMs) using explicit causal graphs.

**CFGPT** (Counterfactual Graph-based Post-Training) is a two-stage framework that distills counterfactual reasoning capability from language to vision, achieving **+12.5%** improvement over the leading 8B-scale VLM.

### Key Contributions

- **CounterVQA Benchmark** — 712 videos, 3,987 QA pairs across three progressive difficulty levels:
  - **Level 1**: Adjacent counterfactual inference (single-hop causal reasoning)
  - **Level 2**: Long-chain counterfactual inference (multi-hop causal chains)
  - **Level 3**: Counterfactual inference with non-existent events (hallucination detection)
- **CFGPT Framework** — Two-stage post-training combining cross-modal causal transfer (SFT) and visual-causal alignment optimization (GRPO with causal graph rewards)
- **Comprehensive Evaluation** — Benchmarking 5 state-of-the-art open- and closed-source VLMs

<p align="center">
  <img src="assets/pipeline.png" width="90%">
</p>

---

## 📊 Main Results

| Model | Level 1 | Level 2 | Level 3 | Avg. |
|-------|---------|---------|---------|------|
| ChatGPT-4o | 9.6 | 8.0 | 1.3 | 6.2 |
| Gemini-2.5-Pro | 41.1 | 40.5 | 42.4 | 41.5 |
| Qwen-2.5-VL-7B | 12.3 | 10.3 | 2.7 | 8.3 |
| Qwen-2.5-VL-32B | 33.0 | 46.2 | 44.0 | 39.4 |
| Qwen-3-VL-8B | 54.6 | 62.1 | 65.3 | 60.1 |
| **Qwen-3-VL-8B + CFGPT (Ours)** | **70.1** | **71.6** | **76.0** | **72.6** |

---

<!-- ## 🛠️ Installation

```bash
git clone https://github.com/AzhenHETU/CounterVQA.git
cd CounterVQA
conda create -n countervqa python=3.10 -y
conda activate countervqa
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.1
- Transformers ≥ 4.40
- [Decord](https://github.com/dmlc/decord) (video loading)
- vLLM (optional, for fast inference)

---

## 📁 Project Structure

```
CounterVQA/
├── data/
│   ├── countervqa/              # Benchmark data
│   │   ├── videos/              # Video files
│   │   ├── causal_graphs/       # Causal graphs for each video
│   │   ├── questions/           # QA pairs (Level 1/2/3)
│   │   └── splits/              # Train/Val/Test splits
│   └── sft_data/                # SFT training data with CoT
├── src/
│   ├── dataset_generation/      # Multi-agent causal graph construction
│   │   ├── observer_agent.py
│   │   ├── verifier_agent.py
│   │   ├── critic_agent.py
│   │   └── synthesizer_agent.py
│   ├── evaluation/              # Benchmark evaluation scripts
│   ├── training/
│   │   ├── sft/                 # Stage I: Cross-modal causal transfer
│   │   └── grpo/                # Stage II: Visual-causal alignment (GRPO)
│   └── rewards/                 # Reward model (Rcausal + Rvisual)
├── configs/                     # Training & evaluation configs
├── scripts/                     # Shell scripts for running experiments
└── README.md
```

---

## 📥 Data Preparation

### Download CounterVQA Benchmark

```bash
# Download from HuggingFace
huggingface-cli download xxx/CounterVQA --local-dir data/countervqa

# Or download manually from:
# https://huggingface.co/datasets/xxx/CounterVQA
```

### Download Ego-Exo4D Videos

CounterVQA is built on [Ego-Exo4D](https://ego-exo4d-data.org/). Follow their instructions to download the source videos, then link them:

```bash
ln -s /path/to/ego-exo4d/videos data/countervqa/videos
```

---

## 🚀 Quick Start

### Evaluate on CounterVQA

```bash
# Evaluate a vanilla model
python scripts/evaluate.py \
    --model Qwen/Qwen3-VL-8B \
    --data_dir data/countervqa \
    --output_dir results/qwen3-vl-8b-vanilla

# Evaluate our CFGPT model
python scripts/evaluate.py \
    --model xxx/CFGPT-Qwen3-VL-8B \
    --data_dir data/countervqa \
    --output_dir results/cfgpt
```

### Evaluate with API-based Models

```bash
# GPT-4o
python scripts/evaluate_api.py \
    --model gpt-4o \
    --api_key $OPENAI_API_KEY \
    --data_dir data/countervqa

# Gemini-2.5-Pro
python scripts/evaluate_api.py \
    --model gemini-2.5-pro \
    --api_key $GOOGLE_API_KEY \
    --data_dir data/countervqa
```

---

## 🏋️ Training CFGPT

### Stage I: Cross-Modal Causal Transfer (SFT)

```bash
bash scripts/train_sft.sh
```

<details>
<summary>Key hyperparameters</summary>

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-5 |
| LoRA rank | 16 |
| Epochs | 3 |
| GPUs | 4× NVIDIA H200 |
| Frames per video | 512 |
| Frame resolution | 144×144 |

</details>

### Stage II: Visual-Causal Alignment (GRPO)

```bash
bash scripts/train_grpo.sh
```

<details>
<summary>Key hyperparameters</summary>

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-5 |
| Temperature | 0.9 |
| Epochs | 3 |
| Samples per question (K) | 4 |
| α (Rcausal weight) | 0.5 |
| β (Rvisual weight) | 0.5 |

</details>

---

## 🔧 Dataset Generation Pipeline

To construct causal graphs and generate counterfactual questions for new videos:

### Step 1: Multi-Agent Causal Graph Construction

```bash
python src/dataset_generation/build_causal_graph.py \
    --video_dir /path/to/videos \
    --annotation_dir /path/to/annotations \
    --output_dir data/causal_graphs
```

### Step 2: Video Filtering via Graph Complexity

```bash
python src/dataset_generation/filter_videos.py \
    --graph_dir data/causal_graphs \
    --ancd_threshold 0.2 \
    --causal_depth_threshold 3 \
    --cnda_threshold 0.12
```

### Step 3: Question Generation & Quality Control

```bash
python src/dataset_generation/generate_questions.py \
    --graph_dir data/causal_graphs \
    --output_dir data/questions
```

---

## 📈 Benchmark Statistics

| | Count |
|---|---|
| Videos | 712 |
| QA Pairs | 3,987 |
| Level 1 (Adjacent) | 38.6% |
| Level 2 (Long-chain) | 26.7% |
| Level 3 (Non-existent) | 34.6% |
| Human-to-Human (H2H) | 39.3% |
| Human-to-Object (H2O) | 60.7% |
| Human Accuracy | 97.4% |

---

## 🤗 Model Zoo

| Model | Base | Training | HF Link | Avg. Acc. |
|-------|------|----------|---------|-----------|
| CFGPT-Qwen3-VL-8B-SFT | Qwen3-VL-8B | SFT only | [🤗 Link](#) | 69.3 |
| CFGPT-Qwen3-VL-8B | Qwen3-VL-8B | SFT + GRPO | [🤗 Link](#) | 72.6 |

---

## 📝 Citation

If you find CounterVQA or CFGPT useful in your research, please cite our paper:

```bibtex
@inproceedings{chen2026countervqa,
  title={Distilling Counterfactual Reasoning from Language to Vision: Causal Graph Guided Post-Training for Video Understanding},
  author={Chen, Yuefei and Liu, Jiang and Lin, Xiaodong and Tang, Ruixiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## 🙏 Acknowledgements

- [Ego-Exo4D](https://ego-exo4d-data.org/) for the source video dataset
- [Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL) for the base VLM
- Computational resources provided through ACCESS (NSF)

--- -->

## License

This project is released under the [Apache 2.0 License](LICENSE).

---

<p align="center">
  <i>If you have any questions, please open an issue or contact <a href="mailto:chen.yuefei@rutgers.edu">chen.yuefei@rutgers.edu</a></i>
</p>
