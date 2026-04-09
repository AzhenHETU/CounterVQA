# CounterVQA: Evaluating and Improving Counterfactual Reasoning in Vision-Language Models for Video Understanding


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



## License

This project is released under the [Apache 2.0 License](LICENSE).

---

<p align="center">
  <i>If you have any questions, please open an issue or contact <a href="mailto:chen.yuefei@rutgers.edu">chen.yuefei@rutgers.edu</a></i>
</p>
