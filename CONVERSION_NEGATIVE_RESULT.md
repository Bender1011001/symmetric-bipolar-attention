# Negative Result: Converting a Pretrained Softmax Model to SBA

**Status: did not work.** This document records the Gemma-4 softmax→SBA
conversion experiments (referred to internally as "Thread B" / the homotopy
bridge) and the evidence that the approach, as attempted, fails to preserve
model quality. It is kept as an honest negative result so the approach is not
re-attempted blindly.

The separate **from-scratch Softmax-vs-SBA comparison** (see
[`README.md`](README.md)) is an experimental code path. These conversion
results neither establish nor refute its performance. The local measurement
artifacts cited below are not included in this checkout; the numbers are
reported results, not independently verified by the published artifacts.

---

## 1. What we tried

Goal: take a pretrained softmax-attention model (`google/gemma-4-E4B`) and
convert it to **Symmetric Bipolar Attention** by training only PEFT LoRA
adapters — keeping base weights frozen — so that the model keeps its language
ability while its attention routing becomes signed (`A_ij ∈ [-1, 1]`).

Approaches attempted, in order:

| Variant | Script | Idea |
| :--- | :--- | :--- |
| v1 LoRA | `run_lora_conversion.py`, `run_lora_conversion_v2.py` | Direct LoRA fit of SBA-patched attention |
| v3 homotopy bridge | `run_softmax_bridge_conversion.py`, `run_homotopy_conversion*.py` | 3-stage homotopy: anneal row-L1 and BOS-sink targets from softmax toward SBA |
| Progressive / dual-teacher | `run_progressive_conversion.py`, `run_dual_teacher_conversion.py` | Staged + dual-teacher distillation |
| ECoT distillation | `run_b200_ecot_training.sh` + self-distill data | Distill on Explicit-Chain-of-Thought data with `<INHIBIT>`/`<RECALCULATE>` phases |

Diagnostics and evals: `gemma4_diagnostics_v3.py`, `gemma4_diagnostics_v4.py`,
`gemma4_gradient_audit.py`, `gemma4_heldout_trap_eval.py`, `evaluate_mmlu.py`.

## 2. The evidence it failed

**MMLU (general capability) dropped, never recovered to baseline:**

| Model | MMLU acc | Source |
| :--- | :--- | :--- |
| Base `gemma-4-E4B` (unconverted) | **71.2%** (203/285) | `sba_math_handoff/mmlu_base_final_quick.json` |
| Best converted (distill bridge, ckpt-5000) | **60.0%** (171/285) | `sba_math_handoff/mmlu_b200_final_quick.json` |
| ECoT-tuned checkpoint | **23.67%** (71/300) | `context.md` audit |

**From-scratch bridge variant also collapsed downstream:** the v3 bridge
checkpoint scored **0/6** on the held-out task eval
(`_from_sba_flat/v3_bridge_task_eval.json`).

**Mechanistic diagnostics show the model gamed the objective, not learned signed logic:**
- Row L1 norm was successfully pushed off the softmax constraint (≈0.65–0.88),
  and negative-attention fraction moved — so the *surface* SBA targets were hit.
- But the **held-out trap eval** stayed at chance: **50%** accuracy, failing
  `bat_ball_trap` and `lilypad_trap`, with the inhibition ratio flat at
  **1.0010** (`gemma4_heldout_trap_eval.py` output).
- The **gradient audit** (`R_grad` mean abs ≈ **4.31**) showed update activity
  concentrated on negative weights specifically during `<INHIBIT>` template
  tokens.

## 3. Diagnosis

The model learned a **positional shortcut**: it used the new signed/negative
attention weights as **position markers for the XML/template tokens** in the
ECoT format (`<INHIBIT>`, etc.) rather than as logic-bearing inhibitory
routing. It satisfied the row-L1 / negative-fraction *regularizers* without
acquiring the intended signed-reasoning behavior — hence surface metrics move
while real capability (MMLU) drops and trap accuracy stays at chance.

## 4. Why this is plausible / what would be needed to revisit

- Signed attention is a **different function class** from softmax; LoRA on
  frozen base weights may simply lack the capacity to re-route the residual
  stream into a coherent signed regime without catastrophic interference.
- The regularizer-driven objective is **gameable** — hitting row-L1 / neg-frac
  targets is necessary but nowhere near sufficient for preserved reasoning.

If revisited, the bar should be: an evaluation that **cannot** be satisfied by a
positional shortcut (the trap suite is a start), and at minimum **non-regression
on MMLU** before any claim of successful conversion. Training SBA **from
scratch** (the other thread in this repo) sidesteps the conversion problem
entirely and is the recommended path.

## 5. File map for this thread

- Training: `run_softmax_bridge_conversion.py`, `run_homotopy_conversion*.py`,
  `run_lora_conversion*.py`, `run_progressive_conversion.py`,
  `run_dual_teacher_conversion.py`, `run_b200_*.sh`
- Diagnostics/eval: `gemma4_*.py`, `evaluate_mmlu.py`, `conversion_diagnostics.py`,
  `run_diagnostics.py`, `verify_rbos.py`, `verify_bos_multi_prompt.py`
- Data prep: `generate_self_distill_data.py`, `prepare_reasoning_data.py`
- Preserved results: `_from_sba_flat/v1_lora_*.json`,
  `_from_sba_flat/v3_bridge_*.json`, `sba_math_handoff/mmlu_*.json`
- Status notes: `context.md`, `CASCADES_SBA_B200_HANDOVER.md`
