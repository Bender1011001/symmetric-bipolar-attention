# Symmetric Bipolar Attention (SBA): Standalone Reproducibility Suite

This repository isolates **Symmetric Bipolar Attention (SBA)** and **Differential SBA (Diff-SBA)** as standalone attention variants. It provides the mathematical proof-of-concept, optimized FlashAttention implementation vectors, KV-caching verification suites, and matched pre-training baselines.

> This project was split out of `CASCADES--continual-PEFT-for-Local-LLMs`; SBA is an independent attention mechanism and does not depend on that work.

---

## 0. Two Research Threads

This repo contains two distinct lines of work — keep them separate:

1. **From-scratch Softmax-vs-SBA comparison (experimental).**
   Train matched small Transformers from scratch with softmax vs SBA attention
   under identical seeds/hyperparameters and compare. The repository provides
   the comparison code; it does not ship the raw run artifacts needed to audit
   a positive performance claim. The rest of this README describes how to run it.
   Entry points: `run_scratch_tokenized_compare.py`, `run_scratch_compare.py`.

2. **Pretrained softmax → SBA conversion (a negative result).**
   An attempt to convert `google/gemma-4-E4B` to SBA via PEFT/LoRA homotopy
   bridges. **This did not work** — MMLU dropped from 71.2% to 60% (and an ECoT
   variant collapsed to 24%), and diagnostics showed the model learned a
   positional shortcut rather than signed reasoning. It is documented honestly
   in **[`CONVERSION_NEGATIVE_RESULT.md`](CONVERSION_NEGATIVE_RESULT.md)**.
   All `run_*conversion*.py`, `gemma4_*.py`, and `evaluate_mmlu.py` belong to
   this thread.

> **Runnability note:** all imports are root-relative — run scripts from the
> repo root (e.g. `python run_scratch_tokenized_compare.py ...`) or with the
> repo root on `PYTHONPATH`. Datasets, checkpoints, and `results/` are
> gitignored and kept on disk only; regenerate datasets with `build_*_mix.py`.

---

## 1. Math Overview
Standard Softmax attention is constrained to positive-only smoothing operations ($A_{ij} \ge 0$). SBA enables signed attention routing ($A_{ij} \in [-1, 1]$) with row-L1-norm bounding by replacing Softmax with hyperbolic functions:
$$A_{ij} = \frac{\sinh(S_{ij})}{\sum_{k=1}^{T} \cosh(S_{ik})} = \frac{e^{S_{ij}} - e^{-S_{ij}}}{\sum_k (e^{S_{ik}} + e^{-S_{ik}})}$$

To prevent $O(T^2)$ HBM memory materialization overhead, we bridge this formulation to standard hardware-fused attention blocks (SDPA/FlashAttention) by doubling the sequence dimension:
* **Key Concatenation**: $K_{\text{cat}} = [K, -K] \in \mathbb{R}^{2T \times d}$
* **Value Concatenation**: $V_{\text{cat}} = [V, -V] \in \mathbb{R}^{2T \times d}$
* **Attention Execution**: Dispatch directly to `F.scaled_dot_product_attention(Q, K_cat, V_cat, attn_mask=M_cat)`

---

## 2. Installation & Setup

Set up a fresh virtual environment and install the required dependencies (PyTorch, Hugging Face transformers, datasets, tiktoken, numpy):

```bash
# Clone the repository
git clone https://github.com/Bender1011001/symmetric-bipolar-attention.git
cd symmetric-bipolar-attention

# Install dependencies (using uv or pip)
pip install torch transformers datasets tiktoken numpy
```

---

## 3. Two-Command Replication Guide

### Command 1: Build the OpenReason-Code Dataset Mix
This script downloads and compiles a training mixture of GSM8K, Code, and reasoning tokens into a unified JSONL file:

```bash
python build_open_reason_code_mix.py --output data/open_reason_code_mix.jsonl
```

### Command 2: Execute Matched 125M Parameter Training
This command trains a 125M parameter causal Transformer from scratch on the compiled token dataset. It executes standard Softmax and SBA treatment arms sequentially under identical seeds and hyperparameter bounds:

```bash
python run_scratch_tokenized_compare.py \
    --output-dir results/blackwell_125m_overnight \
    --data-files data/open_reason_code_mix.jsonl \
    --train-limit-per-file 135000 \
    --eval-limit-per-file 4000 \
    --n-layer 12 \
    --n-head 12 \
    --n-embd 768 \
    --block-size 1024 \
    --batch-size 16 \
    --grad-accum-steps 4 \
    --steps 15000 \
    --warmup-steps 1000 \
    --lr 6e-4 \
    --weight-decay 0.01 \
    --dtype bf16 \
    --arms both \
    --sba-impl sdpa \
    --save-checkpoints \
    --checkpoint-every 3000
```

---

## 4. Verification & Testing

### A. KV-Cache Mathematical Equivalence Test
To verify that our stateful channel-doubled KV-cache implementation is mathematically identical to a full-sequence forward pass during generation:

```bash
# scratch/ scripts import root modules, so put the repo root on PYTHONPATH
PYTHONPATH=. python scratch/test_diff_sba_cache.py
```
*Expected Output*: Displays passing checks for both Softmax and SBA SDPA, indicating maximum discrepancy bounded under FP32 machine precision ($\approx 1.19 \times 10^{-7}$).

### B. Downstream Task Evaluations
To evaluate the trained checkpoints across GSM8K and arithmetic tasks:

```bash
python evaluate_scratch_tokenized_checkpoint.py \
    --checkpoint results/blackwell_125m_overnight/sba/checkpoints/step15000.pt \
    --output task_eval_step15000.json \
    --max-new-tokens 384 \
    --temperature 0.0
```

### C. Collating Sweep Metrics
To aggregate and format the results across multiple seeds (if running sweeps):

```bash
# Summarize validation loss, peak VRAM, and step latencies
PYTHONPATH=. python scratch/collate_results.py

# Summarize task accuracies
PYTHONPATH=. python scratch/collate_evals.py
```
