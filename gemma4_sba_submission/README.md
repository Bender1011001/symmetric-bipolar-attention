# Gemma 4 Symmetric Bipolar Attention (SBA) Softmax-Bridge Conversion Package

This folder contains the complete, self-contained suite of files required to perform the 3-stage Softmax-to-SBA bridge conversion for the Gemma 4 E4B model (`google/gemma-4-E4B`) on a single GPU.

---

## 1. Directory Structure

* **`patch.py`**: Monkey-patches `Gemma4TextAttention` to replace standard softmax attention with the centered SBA bridge formula, preserving GQA query-head repeating, RoPE embeddings, and attention softcapping. It registers buffers and parameters for the bridge bias ($b_{base}$, $b_{delta}$) and temperature ($\tau$).
* **`run_softmax_bridge_conversion.py`**: The training script that runs the 3-stage conversion. It loads the dataset, sets up PEFT LoRA adapters, freezes all other parameters (including MLPs), and performs the scheduled homotopy training.
* **`gemma4_diagnostics_v3.py`**: A CPU-safe diagnostic script that monkey-patches `_stable_sinh_over_sum_cosh` directly. It runs forward passes on held-out evaluation prompts to log the actual row $L_1$ norm trajectory, BOS attention routing shares ($r_{BOS}$), and the fraction of negative attention weights.
* **`run_training.sh`**: A shell script to launch the optimized training process on the remote host in the background.
* **`run_diagnostics.sh`**: A shell script to run the diagnostic evaluation suite.

---

## 2. Theoretical Formulation

The Symmetric Bipolar Attention (SBA) forward path is formulated as:

$$w_j = \frac{\sinh\left(b + \tau (s_j - \bar{s})\right)}{\sum_k \cosh\left(b + \tau (s_k - \bar{s})\right)}$$

where:
* $s_j$ is the scaled query-key dot-product score for position $j$, softcapped at $50.0$.
* $\bar{s}$ is the mean of the scores over all visible (non-masked) query positions.
* $b$ is the per-head attention bias, consisting of a scheduled base value ($b_{base}$) and a trainable delta parameter ($b_{delta}$).
* $\tau$ is a positive per-head temperature parameter implemented as $\text{softplus}(\log\tau)$.

### Initialization & Softmax Bridging
Strict equivalence to standard centered softmax attention is achieved asymptotically as $b \to +\infty$. In practice, at $b = 4.0$, the bridge functions as a high-fidelity softmax approximation. Stage 0 uses this initialization to bootstrap student parameters before the structural transition begins.

---

## 3. The 3-Stage Homotopy Recipe

The conversion scheduling is structured over three stages to transition the model from positive-only softmax attention to signed bipolar routing while preserving language modeling coherence:

### Stage 0: Preservation (Steps 0 – 500)
* **Parameters**: Bias $b = 4.0$ (constant).
* **Loss Weights**: $\mathcal{L}_{KL} = 1.0$, $\mathcal{L}_{CE} = 0.0$, $\mathcal{L}_{BOS} = 0.0$, $\mathcal{L}_{L1} = 0.0$.
* **Goal**: Student LoRA adapters train purely to mimic the teacher's softmax logits in the high-bias softmax-equivalent regime.

### Stage 1: Transition (Steps 500 – 2000)
* **Parameters**: Bias $b$ is linearly annealed from $4.0 \to 1.5$.
* **Loss Weights**: $\mathcal{L}_{KL} = 1.0$, $\mathcal{L}_{CE} = 0.0$. Structural regularizers ramp up smoothly using a smoothstep schedule to maximums of $\mathcal{L}_{BOS} = 5.0$ and $\mathcal{L}_{L1} = 1.0$.
* **Goal**: Model is forced to break softmax-equivalence. The $L_1$ regularizer pulls row norms down (target = 0.65), and the BOS suppression penalty forces the model to route the attention sink away from fully positive weights, resulting in the learning of signed (negative) attention routing.

### Stage 2: Recovery (Steps 2000 – 5000)
* **Parameters**: Bias $b$ is annealed from $1.5 \to 1.0$.
* **Loss Weights**: Cross-Entropy is activated at weight $\mathcal{L}_{CE} = 1.0$ to act as a language modeling anchor. The KL weight is linearly annealed down from $1.0 \to 0.1$ to allow the student to prioritize text coherence over strict teacher logit matching. Structural weights remain constant at their maximums.
* **Goal**: Model recovers language generation capacity and adjusts to the active signed attention routing regime.

---

## 4. Single-GPU Memory Optimizations

Converting a 7.95 Billion parameter model with a teacher-student setup normally requires over 32 GB of VRAM. To fit the training run on a single 32 GB GPU (e.g., Blackwell or H100), the following optimizations are implemented in `run_softmax_bridge_conversion.py`:

1. **Single-Model Dual-Pass Strategy**: Rather than loading separate teacher and student model instances, only a single model weight copy is loaded in `bfloat16` (~15.9 GB). 
   * *Teacher Pass*: LoRA adapters are disabled (`student.disable_adapter()`) and the attention patch is bypassed (`unpatch_gemma4_sba`) under `torch.no_grad()` to compute the baseline softmax teacher logits.
   * *Student Pass*: LoRA adapters are re-enabled and the SBA patch is re-applied to compute training outputs and gradients.
2. **Sequence Length Limit**: The training context limit is set to $512$ to halve activation memory during backpropagation.
3. **Loss Chunking**: KL divergence and Cross-Entropy loss functions are calculated iteratively in chunks of $32$ tokens to minimize temporary activation memory.

---

## 5. Execution Instructions

### Installation Prerequisites
Ensure `accelerate`, `transformers`, `peft`, and `datasets` are installed:
```bash
pip install torch transformers peft datasets accelerate tqdm
```

### Running Training
Execute the training script using:
```bash
bash run_training.sh
```
This runs the conversion in the background. The log file `gemma4_conversion.log` will record stage steps, loss trajectories, row $L_1$ norms, and active weights.

### Running Diagnostics
Inspect saved checkpoints for true bipolar routing parameters by executing:
```bash
bash run_diagnostics.sh
```
This will process all checkpoints in the output directory sequentially on CPU, saving the detailed report to `results/gemma4_e4b_sba_bridge/gemma4_diagnostics_report.json`.
