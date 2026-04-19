<div align="center">

<img src="https://img.shields.io/badge/NeurIPS-2025-blue?style=for-the-badge&logo=arxiv&logoColor=white" alt="NeurIPS 2025"/>
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/TRL-0.12+-6B35FF?style=for-the-badge" alt="TRL"/>
<img src="https://img.shields.io/badge/PEFT-LoRA-00B04F?style=for-the-badge" alt="PEFT"/>
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>

<br/><br/>

```

████████╗████████╗██████╗ ██╗
╚══██╔══╝╚══██╔══╝██╔══██╗██║
   ██║      ██║   ██████╔╝██║
   ██║      ██║   ██╔══██╗██║
         ██║      ██║   ██║  ██║███████╗
         ╚═╝      ╚═╝   ╚═╝  ╚═╝╚══════╝

```

# Test-Time Reinforcement Learning

### *Self-evolving LLMs on unlabeled data — no ground truth required*

<br/>

[📄 Paper](https://arxiv.org/abs/2504.16084) &nbsp;·&nbsp;
[🗂️ Official Repo](https://github.com/PRIME-RL/TTRL) &nbsp;·&nbsp;
[📦 Models & Results](https://drive.google.com/drive/folders/15KhUU7fQol3oq0lYpXanSqRQrHLgvLdW?usp=sharing) &nbsp;·&nbsp;.

<br/>

---

</div>

## 🧠 What is TTRL?

**TTRL (Test-Time Reinforcement Learning)** is a method for training Large Language Models using Reinforcement Learning **without any labeled data**. Instead of ground-truth answers, it estimates rewards using **majority voting** across multiple model outputs — then uses those rewards to improve the model via GRPO.

> *"The model lifts itself up by its own bootstraps."*

```
 Unlabeled Question
        │
        ▼
 ┌─────────────┐
 │     LLM     │ ──── generates N completions ────►  { ŷ₁, ŷ₂, ..., ŷₙ }
 └─────────────┘                                              │
        ▲                                                     ▼
        │                                           Majority Vote → ŷ*
        │                                                     │
        └──────── GRPO Update ◄──── Reward r(ŷᵢ, ŷ*) ◄──────┘
```

**Key insight:** Even when the majority-voted label is *wrong*, the "Lucky Hit" phenomenon ensures reward accuracy stays high — making RL stable without any supervision.

<br/>

---

## ✨ Highlights
Original Paper Results

| Metric | Qwen2.5-Math-7B (Base) | Qwen2.5-Math-7B + TTRL | Improvement |
|:-------|:----------------------:|:----------------------:|:-----------:|
| AIME 2024 | 16.7 | 43.3 | **+159.3%** |
| AMC | 38.6 | 67.5 | **+74.9%** |
| MATH-500 | 50.6 | 84.2 | **+66.4%** |

- 🔑 **No labels needed** — trains entirely on unlabeled test data
- 📈 **Surpasses maj@n ceiling** — exceeds its own majority voting upper bound
- 🔁 **Compatible** with GRPO, PPO, and PRIME
- 📦 **Scales** from 1.5B to 32B models
- 🧩 **Drop-in reward function** — just replace your reward fn

<br/>

---

## 🗂️ Project Structure

```
TTRL/
├── 📓 ttrl_final.ipynb          # Main training + evaluation notebook
│
├── 📁 math_grader/              # PRIME's math grading utilities
│   ├── __init__.py              # extract_answer, grade, compute_score
│   ├── math_utils.py            # boxed answer extraction
│   ├── math_normalize.py        # LaTeX normalization
│   ├── grader.py                # symbolic + numeric equivalence
│   └── cluster.py               # TTRLClusterCounter (math-aware majority voting)
│
├── 📁 MATH/                     # MATH dataset (JSON + Parquet)
├── 📁 AIME-TTT/                 # AIME 2024 dataset
│
├── 📁 ttrl-grpo-output-*/       # Training checkpoints
├── 📁 ttrl_plots_*/             # Metric plots (saved every 20 steps)
├── 📁 ttrl_evaluation_*/        # Before/after evaluation results
│
└── 📁 ttrl-final-*/             # Final saved models (LoRA adapters)
```

<br/>

---

## ⚙️ Setup

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/TTRL.git
cd TTRL
```

```bash
pip install torch transformers trl>=0.12.0 datasets accelerate peft
pip install latex2sympy2-extended sympy pyarrow pandas matplotlib
pip install math-verify
```

### 2. Set up math_grader

The `math_grader/` package provides PRIME's math-aware grading stack (symbolic + numeric equivalence). Place all 5 files in a folder called `math_grader/` in your working directory:

```
math_grader/
├── __init__.py
├── math_utils.py
├── math_normalize.py
├── grader.py
└── cluster.py
```

> **Note for Kaggle users:** Input directories are read-only. Copy files to `/kaggle/working/math_grader/` first — see notebook Cell 1 for the patching script.

### 3. Prepare your dataset

Your JSON should follow this format:

```json
[
  {
    "prompt": "Every morning Aya goes for a 9-kilometer walk...",
    "answer": "204",
    "source": "aime",
    "id": "0"
  }
]
```

Convert to Parquet for faster training (optional but recommended):

```python
convert_json_to_parquet(
    json_path    = "your_data.json",
    parquet_path = "your_data.parquet",
    tokenizer_name = "Qwen/Qwen2.5-Math-1.5B",
)
```

<br/>

---

## 🚀 Training

### Quick Start

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

# TTRL reward — no ground truth needed
def majority_voting_reward(completions, **kwargs):
    answers  = [extract_answer(flatten_completion(c)) for c in completions]
    cluster  = TTRLClusterCounter(equivalence_func=math_equivalence)
    cluster.update([a for a in answers if a])
    majority, _ = cluster.most_common(1)[0]
    return [1.0 if math_equivalence(a, majority) else 0.0 for a in answers]

trainer = GRPOTrainer(
    model        = "Qwen/Qwen2.5-Math-1.5B",
    reward_funcs = majority_voting_reward,
    args         = GRPOConfig(output_dir="ttrl-out", num_generations=8, ...),
    train_dataset = dataset,
)
trainer.train()
```

### Memory Profiles

Pick based on your GPU VRAM:

| Profile | VRAM Required | Method | LoRA r |
|:--------|:-------------:|:------:|:------:|
| `bf16` | ≥ 24 GB | Full fp16 | 64 |
| `qlora_8bit` | ~16 GB | 8-bit + LoRA | 32 |
| `qlora_4bit` | ~10 GB | 4-bit NF4 + LoRA | 16 |

```python
MEMORY_PROFILE = "qlora_4bit"   # change this one line
```

### Resume from Checkpoint

```python
# Auto-detect latest checkpoint
checkpoint = find_latest_checkpoint("ttrl-grpo-output/")
trainer.train(resume_from_checkpoint=checkpoint)

# Or pin to specific step
trainer.train(resume_from_checkpoint="ttrl-grpo-output/checkpoint-100")
```

<br/>

---

## 📊 Tracked Metrics

Every training step logs the full TTRL diagnostic suite:

### Core Accuracy
| Metric | Description |
|:-------|:------------|
| `pass@k` | Unbiased estimator — probability ≥1 of k samples is correct |
| `avg@n` | Average correctness across all n completions |
| `maj@n` | Is the majority-voted answer correct? |
| `accuracy` | Greedy decoding correctness |

### TTRL-Specific
| Metric | Description |
|:-------|:------------|
| `consensus_ratio` | `majority_count / N` — self-consistency strength |
| `format_rate` | Fraction of completions with `\boxed{}` |
| `reward_accuracy` | Fraction of rewards matching ground truth (Lucky Hit) |

### RL Stability
| Metric | Description |
|:-------|:------------|
| `reward_mean / var` | Reward signal quality |
| `entropy` | Policy exploration vs convergence |
| `response_length` | Should decrease as model gains confidence |

Plots are auto-saved to `ttrl_plots/` every 20 steps:

```
ttrl_plots/
├── metrics_step00020.png
├── metrics_step00040.png
└── metrics_final.png
```

<br/>

---

## 🧪 Evaluation

Run before/after comparison with full visualization:

```python
# Evaluates base model and TTRL model, generates 9-panel comparison plot
base_summary, base_df = evaluate_model_on_dataset(base_model, ...)
ttrl_summary, ttrl_df = evaluate_model_on_dataset(ttrl_model, ...)
```

Output includes:
- ✅ Grouped bar charts (accuracy, avg@n, maj@n)
- ✅ pass@k curve (k = 1, 4, 8, 16)
- ✅ Per-problem pass@1 comparison
- ✅ Consensus ratio vs pass@1 scatter
- ✅ Response length distribution
- ✅ Colour-coded scorecard table

<br/>

---

## 📦 Models, Checkpoints & Results

All trained model checkpoints, LoRA adapters, evaluation CSVs, and training plots are available here:

<div align="center">

### 🔗 [Google Drive — Models & Results](YOUR_DRIVE_LINK_HERE)

</div>

**Contents of the Drive folder:**

```
📁 Drive/
├── 📁 checkpoints/
│   ├── ttrl-grpo-output-aime-qwen2.5-math-1.5b/
│   │   ├── checkpoint-50/
│   │   ├── checkpoint-100/
│   │   └── checkpoint-150/
│   └── ttrl-grpo-output-math-qwen2.5-math-1.5b/
│       └── checkpoint-50/
│
├── 📁 final-models/
│   ├── ttrl-final-grpo-aime-qwen2.5-math-1.5b/   # LoRA adapter
│   └── ttrl-final-grpo-math-qwen2.5-math-1.5b/
│
├── 📁 evaluation-results/
│   ├── ttrl_evaluation_AIME_Qwen2.5-Math-1.5B/
│   │   ├── base_per_problem.csv
│   │   ├── ttrl_per_problem.csv
│   │   ├── comparison.csv
│   │   └── ttrl_evaluation.png
│   └── ...
│
└── 📁 training-plots/
    ├── ttrl_plots_grpo_aime/
    └── ttrl_plots_grpo_math/
```

<br/>

---

## 🔧 Key Implementation Details

### Why `TTRLClusterCounter` instead of Python's `Counter`?

Plain `Counter` treats `"1/2"`, `"0.5"`, and `"\frac{1}{2}"` as three different answers — splitting the majority vote. `TTRLClusterCounter` uses PRIME's `grade()` function for mathematical equivalence, so all three cluster together correctly.

```python
# ❌ Wrong — splits the vote
Counter(["1/2", "0.5", "\\frac{1}{2}", "1/2"])
# → {"1/2": 2, "0.5": 1, "\\frac{1}{2}": 1}  majority = "1/2" (count 2)

# ✅ Correct — math-aware clustering
TTRLClusterCounter(math_equivalence).update(["1/2", "0.5", "\\frac{1}{2}", "1/2"])
# → all 4 in one cluster  majority = "1/2" (count 4)
```

### The "Lucky Hit" Phenomenon

Even when majority voting produces a *wrong* label, reward accuracy stays high:

```
Sampled:  [1, 1, 2, 2, 2, 4, 5, 6]   True label: 3
Majority: 2  (wrong!)

Rewards from majority label:  [0, 0, 1, 1, 1, 0, 0, 0]
Rewards from true label:      [0, 0, 0, 0, 0, 0, 0, 0]

Reward hit rate: 62.5%  ← most rewards are still correct
```

### Format + Majority Voting Reward

```python
reward = format_score + majority_score
#        └─ 0.5 if \boxed{} present    └─ 1.0 if matches majority
# max = 1.5 per completion
```

The format reward prevents early training collapse when the model hasn't learned to use `\boxed{}` yet.

<br/>

---

## ⚠️ Common Errors & Fixes

| Error | Cause | Fix |
|:------|:------|:----|
| `ConnectionRefusedError` on port 8000 | `use_vllm=True` without server | Remove `use_vllm` or launch `trl vllm-serve` first |
| `TypeError: expected string, got list` | TRL passes completions as message dicts | Use `flatten_completion()` helper |
| `OverflowError: cannot convert inf` | Model generates `inf` early in training | Add `math.isinf()` guard in `normalise()` |
| `CUDA device-side assert` | `num_generations` too high / fp16 mismatch | Use CFG profile values; set `fp16=True` |
| `ImportError: bitsandbytes` | Wrong version | `pip install -U bitsandbytes>=0.46.1` |
| `AcceleratorError` on resume | Config changed between runs | Keep config identical to original run |

<br/>

---

## 📋 Hyperparameters

Critical settings from the paper (§3.1):

```python
temperature          = 0.6      # CRITICAL: 1.0 causes training collapse (paper Fig 11)
learning_rate        = 5e-7     # with cosine schedule
num_generations      = 64       # for voting; downsample 32 for training (paper setup)
                                # use 4-8 for single GPU
max_completion_length = 3072    # for LRMs; 512 for smaller models
episodes             = 10/30/80 # for MATH-500/AMC/AIME (scale with difficulty)
```

<br/>

---

## 📚 Citation

If you use this implementation, please cite the original TTRL paper:

```bibtex
@article{zuo2025ttrl,
  title   = {TTRL: Test-Time Reinforcement Learning},
  author  = {Zuo, Yuxin and Zhang, Kaiyan and Sheng, Li and Qu, Shang
             and Cui, Ganqu and Zhu, Xuekai and Li, Haozhan and Zhang, Yuchen
             and Long, Xinwei and Hua, Ermo and others},
  journal = {arXiv preprint arXiv:2504.16084},
  year    = {2025}
}
```

<br/>

---

## 🙏 Acknowledgements

- [PRIME-RL/TTRL](https://github.com/PRIME-RL/TTRL) — original implementation and math grading utilities
- [TRL](https://github.com/huggingface/trl) — GRPO trainer
- [verl](https://github.com/volcengine/verl) — official TTRL training framework
- [Qwen2.5-Math](https://huggingface.co/Qwen) — backbone models

<br/>

---

<div align="center">

[⬆ Back to top](#test-time-reinforcement-learning)

</div>
