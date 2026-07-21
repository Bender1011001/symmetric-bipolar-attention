# Symmetric Bipolar Attention (SBA) Mathematical Handover Report

> [!NOTE]
> This handover document is prepared for the model that designed the original Symmetric Bipolar Attention (SBA) architecture and mathematical formulation. It details the Gemma 4 Softmax-to-SBA Homotopy Conversion experiments, final evaluation results, and a mathematical post-mortem of the orthogonal gradient projection trick.

---

## 1. Project Context & Implementation Overview

The goal of this project is to convert the pre-trained text model `google/gemma-4-E4B` (8.0B parameters) from standard Softmax attention to **Symmetric Bipolar Attention (SBA)**. Because full model fine-tuning of an 8B model is resource-intensive and risks destroying pre-trained capabilities, we restrict parameter updates to parameter-efficient fine-tuning (PEFT) using **LoRA adapters** on the query, key, value, and output projections ($W_q, W_k, W_v, W_o$) inside the attention blocks. The base model weights remain frozen.

The attention routing is modified from standard Softmax:
$$\text{Softmax}(S)_{ij} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}}$$

to the Symmetric Bipolar Attention (SBA) bridge:
$$\text{SBA}(S)_{ij} = \frac{\sinh(b_h + \tau_h (S_{ij} - \mu_i))}{\sum_k \cosh(b_h + \tau_h (S_{ik} - \mu_i))}$$

where:
*   $S_{ij}$ are the scaled dot-product attention logits (with softcapping applied).
*   $\mu_i = \text{mean}_{k \in \text{visible}}(S_{ik})$ is the row-wise mean of visible logits.
*   $b_h$ is the bias parameter per head (initialized to $b_h = 4.0$ to ensure Stage 0 softmax equivalence).
*   $\tau_h$ is the positive temperature parameter per head (initialized to $1.0$).
*   The bridge parameters $b_h$ and $\tau_h$ are trainable alongside the LoRA weights.

### Homotopy Conversion Schedule
The conversion process transitions through three distinct phases:
1.  **Stage 0: Preservation (Steps 0 - 300)**: $b_h$ is held fixed at $4.0$. The SBA bridge is mathematically equivalent to softmax. The LoRA parameters are trained using KL-divergence distillation against the frozen softmax teacher to match the initial representation routing.
2.  **Stage 1: Transition (Steps 300 - 1300)**: $b_h$ is gradually ramped down from $4.0$ to $1.5$. The model transitions from the softmax-equivalent positive-only routing to bipolar attention, starting to utilize negative attention weights.
3.  **Stage 2: Recovery (Steps 1300 - 5000)**: $b_h$ is ramped down from $1.5$ to $1.0$. The distillation loss weight is reduced, and the model is trained with a combined Cross-Entropy language modeling objective and KL-divergence loss to recover its language coherence.

---

## 2. What Was Done: Key Milestone Summaries

1.  **Gemma 4 Custom Patching (`patch.py`)**:
    We implemented a robust monkey-patching wrapper for Hugging Face's `Gemma4TextAttention` that correctly preserves Rotary Position Embeddings (RoPE), Grouped-Query Attention (GQA) head repetition, and Gemma 4's final shared-KV layers, while routing the attention weights through a numerically stabilized centered SBA bridge.
2.  **Single-Model Dual-Pass Optimization (`run_softmax_bridge_conversion.py`)**:
    To train on consumer GPUs with limited VRAM (e.g., 32GB/80GB), loading both teacher (softmax) and student (SBA) model weights concurrently is memory-prohibitive. We implemented a dynamic hot-swapping pass. For every batch:
    *   **Teacher Pass**: Disable LoRA adapters and bypass the SBA patch to run the forward pass on the base softmax configuration and collect logits.
    *   **Student Pass**: Re-enable LoRA adapters and the SBA patch, apply the current bridge parameters, run the forward pass, and compute gradients.
3.  **Numerical Stability Refinement**:
    The SBA bridge is computed in floating-point format. We stabilized the sinh/cosh math against extreme attention logits by factoring out the row-maximum of absolute logits and implementing PyTorch's `expm1` to prevent underflows/overflows.
4.  **B200 Distillation & General Capabilities Anchor**:
    In earlier experiments using 0% general knowledge replay data, the model exhibited severe **representation collapse** during reasoning training—using negative weights as template position indicators rather than logic elements, causing MMLU scores to crash to $23.67\%$.
    To fix this, we generated a **Capabilities Self-Distillation Dataset** (`sba_self_distill.jsonl`) containing 8,000 prompt-response pairs produced directly by the base Gemma 4 model. This serves as a general capability anchor.
5.  **Orthogonal Gradient Projection constraint**:
    We implemented a Frobenius-orthogonal constraint on the parameter updates to force the adapter weights $\Delta W$ to update orthogonally to the pre-trained weights $W_{\text{base}}$ to protect the pre-trained capabilities.

---

## 3. Empirical Results & Performance Diagnostics

Following the 5,000-step distillation-based bridge conversion run on the Blackwell/B200 GPU, we conducted local MMLU evaluations. The evaluation was run on a representative subset of **285 questions across 57 subjects** (5 questions per subject) to assess pre-trained general knowledge preservation:

### Overall MMLU Performance Comparison
| Model / Checkpoint | Overall Accuracy | Total Correct | Total Questions |
| :--- | :---: | :---: | :---: |
| **Gemma 4 Softmax Baseline** | **71.23%** | 203 | 285 |
| **SBA Checkpoint-5000** | **60.00%** | 171 | 285 |
| **Net Absolute Drop (Forgetting)** | **-11.23%** | -32 | — |

*   **Row $L_1$ Norm**: Successfully dropped to a mean of **$0.912$** (breaking the softmax row sum = $1.0$ constraint).
*   **Negative Attention Fraction**: Settled at **$10.2\% - 16.8\%$**, indicating that signed bipolar attention routing was successfully activated.
*   **BOS Attention Sink**: Successfully suppressed (BOS token weight contribution decoupled from $100\%$ down to $<2\%$).
*   **The Forgetting Dilemma**: Despite implementing the first-order orthogonal projection constraint to keep updates orthogonal to the base model weights, the model still suffered an **11.23% absolute accuracy drop** on general knowledge.

---

## 4. Mathematical Critique: Why the Orthogonal Projection Trick Failed

The orthogonal gradient projection constraint was designed to ensure that the cumulative adapter updates $\Delta W = B A$ do not overwrite the representational subspace of the frozen base weights $W_{\text{base}}$ by enforcing:
$$\langle \Delta W, W_{\text{base}} \rangle_F = \text{Tr}(\Delta W^T W_{\text{base}}) = 0$$

At each step, we project the adapter parameter gradients $G_B = \nabla_B \mathcal{L}$ and $G_A = \nabla_A \mathcal{L}$ in Frobenius space:
$$V_B = W_{\text{base}} A^T, \quad V_A = W_{\text{base}}^T B$$
$$I = \langle G_B, V_B \rangle_F + \langle G_A, V_A^T \rangle_F, \quad N = \|V_B\|_F^2 + \|V_A^T\|_F^2$$
$$G_B \leftarrow G_B - \frac{I}{N} V_B, \quad G_A \leftarrow G_A - \frac{I}{N} V_A^T$$

However, this constraint failed to prevent the 11.23% MMLU degradation due to three core mathematical and structural reasons:

### Reason 1: Adam Optimizer's Geometric Warping (Coordinate Variance Division)
The orthogonal projection is applied directly to the *raw gradients* $G_B$ and $G_A$. This guarantees that the gradient direction step $G_{\text{proj}}$ is orthogonal to the constraint plane:
$$\langle G_{\text{proj}}, V \rangle = 0$$

However, modern training utilizes the **AdamW optimizer**, which does not update the weights in the direction of the raw gradient. Instead, Adam maintains exponential moving averages of the first moment $m_t$ (momentum) and the second raw moment $v_t$ (uncentered variance):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) G_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) G_t^2$$

The actual step taken by the optimizer is:
$$\Delta \theta_t = -\frac{\eta}{\sqrt{v_t} + \epsilon} m_t$$

The division by the element-wise standard deviation vector $\sqrt{v_t} + \epsilon$ is a coordinate-wise scaling. Let $D_t = \text{diag}\left(\frac{1}{\sqrt{v_t} + \epsilon}\right)$ be the diagonal scaling matrix. The parameter update vector is:
$$\Delta \theta_t = -\eta D_t m_t$$

Because the scaling factor is non-uniform (different coordinates have different historical variances and update frequencies), **this element-wise scaling completely warps the geometry of the update vector**. Even if the raw gradient $G_t$ is perfectly orthogonal to the base weights, the warped step $\Delta \theta_t$ is not:
$$\langle D_t m_t, V \rangle \neq 0$$

Over 5,000 steps, these non-orthogonal coordinate updates accumulate significant representational drift, slowly writing into the frozen base model weights' subspace and causing catastrophic forgetting.

### Reason 2: Weight Space vs. Activation Space Orthogonality
The projection constraint enforces orthogonality in the **parameter weight space** (Frobenius norm over matrices):
$$\langle \Delta W, W_{\text{base}} \rangle_F = 0$$

However, neural networks do not process parameters; they process **activations**. When an input activation vector $x$ passes through the attention layer, the output is:
$$y = (W_{\text{base}} + \Delta W) x = W_{\text{base}} x + \Delta W x$$

For pre-trained capabilities to be perfectly preserved on general knowledge prompts, the output of the adapter must be zero for activations $x_{\text{general}}$ representing that general knowledge:
$$\Delta W x_{\text{general}} \approx 0$$

This requires $\Delta W$ to be orthogonal to the *activation vectors* of the general domain:
$$\Delta W \perp x_{\text{general}}$$

There is no mathematical guarantee that Frobenius orthogonality in weight space ($\text{Tr}(\Delta W^T W_{\text{base}}) = 0$) translates to orthogonality in activation space ($\Delta W x = 0$). Because the conversion run used $0\%$ general domain replay data during Phase 2 (Recovery), the adapter weights optimized their row-space projection purely for the self-distillation and conversion tasks. As a result, the adapter output $\Delta W x$ acted as a destructive interference vector when multiplied by general-domain activations, changing the routing distributions and causing MMLU decay.

### Reason 3: Mixed Precision Representation Truncation
Gemma 4 training is conducted in mixed precision (`bfloat16`).
*   `bfloat16` utilizes a 7-bit mantissa, yielding roughly $1\%$ relative precision.
*   When calculating the projection scalars $I$ and $N$, the dot products sum over $65,000+$ elements per layer. 
*   Summing thousands of elements in `bfloat16` causes catastrophic truncation errors due to the running sum growing larger than individual elements. While we mitigated this by casting tensors to `float32` before summing, the updated gradients must still be cast back into `bfloat16` to apply the parameter step. This quantization step back to `bfloat16` introduces a small, persistent projection error that violates the orthogonality boundary on every step.

---

## 5. Overview of Included Handoff Files

The following flat files are copied directly into this handoff folder:

1.  **`patch.py`**: The core Symmetric Bipolar Attention (SBA) monkey-patch module for `Gemma4TextAttention`.
2.  **`run_softmax_bridge_conversion.py`**: The 3-stage homotopy conversion pipeline featuring the double-pass teacher-student loading loop and the high-precision orthogonal gradient projection implementation.
3.  **`evaluate_mmlu.py`**: The local evaluation script used to measure baseline and checkpoint MMLU accuracy.
4.  **`gemma4_diagnostics_v4.py`**: Evaluates row $L_1$ norms, signed vs. cosh BOS token share, and negative weight fractions across layers on representative prompts.
5.  **`gemma4_gradient_audit.py`**: Computes L2 norms of gradients on negative attention weights during `<INHIBIT>` tokens to verify active routing.
6.  **`gemma4_heldout_trap_eval.py`**: Runs evaluations on reasoning traps to check if the model triggers negative attention weights outside explicit reasoning tags.
7.  **`generate_self_distill_data.py` & `prepare_reasoning_data.py`**: Scripts used to format and tokenize the self-distillation capability anchor datasets.
8.  **`sba_triton_kernel.py`**: Triton implementation of the Symmetric Bipolar Attention forward and backward passes.
9.  **`math_ops.py`**: Basic helper modules for attention calculations.
10. **`mmlu_base_final_quick.json` & `mmlu_b200_final_quick.json`**: Raw JSON results of the local 285-question diagnostic MMLU evaluations.
11. **`SBA_Ablation_Colab.ipynb`**: Jupyter notebook ready for upload to Google Colab to run MMLU evaluations, diagnostic checks, and ablation experiments.
