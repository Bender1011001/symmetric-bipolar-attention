# run_softmax_bridge_conversion.py
# Copyright (c) 2026.
# SPDX-License-Identifier: Apache-2.0
"""
Three-stage Softmax -> Centered SBA bridge conversion for Gemma 4 text models.

This script trains only PEFT LoRA adapters on q_proj/k_proj/v_proj/o_proj plus
small SBA bridge parameters attached by patch.py. All original base model weights,
including MLPs, embeddings, norms, routers, and experts, remain frozen.

Expected use:

  accelerate launch run_softmax_bridge_conversion.py \
    --model_name_or_path google/gemma-4-E4B \
    --dataset_name wikitext --dataset_config wikitext-103-raw-v1 \
    --text_column text \
    --output_dir ./gemma4-e4b-sba-bridge

For local JSONL/TXT data, use --train_file path/to/data.jsonl or .txt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed

from patch import (
    clear_sba_attention_stats,
    clear_sba_padding_mask,
    freeze_non_lora_non_sba_parameters,
    named_sba_bridge_parameters,
    patch_gemma4_sba,
    unpatch_gemma4_sba,
    reset_sba_bridge_deltas,
    save_sba_bridge_state,
    set_sba_bridge_bias,
    set_sba_padding_mask,
    set_sba_structure_loss_enabled,
    sba_structure_loss,
    set_sba_patch_bypass,
)

try:
    from accelerate import Accelerator
except ImportError as exc:
    raise RuntimeError(
        "run_softmax_bridge_conversion.py requires accelerate. Install with: pip install accelerate"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Gemma 4 softmax attention to centered SBA with LoRA.")

    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-4-E4B")
    parser.add_argument("--teacher_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="eager")

    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--append_eos", action="store_true")
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--pad_to_multiple_of", type=int, default=8)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--allow_unscoped_lora_fallback",
        action="store_true",
        help=(
            "Fallback to suffix target_modules=['q_proj','k_proj','v_proj','o_proj'] if exact Gemma text targets "
            "are not found. This may hit vision/audio towers and is disabled by default."
        ),
    )

    parser.add_argument("--attention_softcap", type=float, default=50.0)
    parser.add_argument("--initial_tau", type=float, default=1.0)
    parser.add_argument("--min_tau", type=float, default=0.05)
    parser.add_argument("--max_tau", type=float, default=8.0)
    parser.add_argument("--row_l1_target", type=float, default=0.65)

    parser.add_argument("--stage0_steps", type=int, default=500)
    parser.add_argument("--stage1_steps", type=int, default=1500)
    parser.add_argument("--stage2_steps", type=int, default=3000)
    parser.add_argument("--max_train_steps", type=int, default=None)

    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--stage2_final_kl_weight", type=float, default=0.1)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--rbos_loss_weight", type=float, default=5.0)
    parser.add_argument("--row_l1_loss_weight", type=float, default=1.0)
    parser.add_argument("--loss_chunk_size", type=int, default=128)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--bridge_learning_rate", type=float, default=5.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1.0e-8)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--tf32", action="store_true")

    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument(
        "--save_final_merged_lora",
        action="store_true",
        help="Also save a merged LoRA checkpoint. The SBA patch/state is still required for inference.",
    )

    args = parser.parse_args()

    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Provide either --dataset_name or --train_file.")
    if args.stage0_steps < 0 or args.stage1_steps < 0 or args.stage2_steps < 0:
        raise ValueError("stage step counts must be non-negative")
    if (args.stage0_steps + args.stage1_steps + args.stage2_steps) <= 0 and args.max_train_steps is None:
        raise ValueError("at least one stage must have positive steps")
    if args.max_seq_length < 2:
        raise ValueError("--max_seq_length must be at least 2")
    if args.kl_temperature <= 0.0:
        raise ValueError("--kl_temperature must be positive")
    if args.loss_chunk_size <= 0:
        raise ValueError("--loss_chunk_size must be positive")

    return args


def parse_torch_dtype(dtype_name: str) -> Union[str, torch.dtype]:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"unsupported torch dtype {dtype_name}")


def load_causal_model(model_name_or_path: str, args: argparse.Namespace):
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": parse_torch_dtype(args.torch_dtype),
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    except TypeError as exc:
        if "attn_implementation" not in str(exc):
            raise
        model_kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    except ValueError as exc:
        message = str(exc)
        if "AutoModelForCausalLM" not in message and "Unrecognized configuration" not in message:
            raise
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            raise
        return AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)


def _is_text_projection_name(name: str) -> bool:
    if not name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")):
        return False
    if ".self_attn." not in name:
        return False

    lower = name.lower()
    blocked = ("vision", "visual", "image", "audio", "encoder.layers")
    return not any(token in lower for token in blocked)


def resolve_lora_targets(model, *, allow_unscoped_fallback: bool) -> Union[str, List[str]]:
    target_names = sorted(name for name, _ in model.named_modules() if _is_text_projection_name(name))
    if target_names:
        escaped = [re.escape(name) for name in target_names]
        return r"^(?:" + "|".join(escaped) + r")$"

    if allow_unscoped_fallback:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    raise RuntimeError(
        "Could not resolve exact text-attention LoRA targets. Refusing to use unscoped suffix targets because "
        "Gemma 4 multimodal checkpoints may also contain q_proj/k_proj/v_proj/o_proj in vision/audio towers. "
        "Inspect model.named_modules() or rerun with --allow_unscoped_lora_fallback if this is a text-only class."
    )


def attach_lora_and_sba(student, args: argparse.Namespace):
    target_modules = resolve_lora_targets(student, allow_unscoped_fallback=args.allow_unscoped_lora_fallback)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    student = get_peft_model(student, lora_config)

    attention_softcap = None if args.attention_softcap <= 0 else args.attention_softcap
    patched_count = patch_gemma4_sba(
        student,
        initial_bias=4.0,
        initial_tau=args.initial_tau,
        min_tau=args.min_tau,
        max_tau=args.max_tau,
        attention_softcap=attention_softcap,
        row_l1_target=args.row_l1_target,
        collect_structure_loss=True,
    )

    freeze_non_lora_non_sba_parameters(student)
    return student, patched_count, target_modules


def split_trainable_parameters(model) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    bridge_param_ids = {id(param) for _, param in named_sba_bridge_parameters(model)}
    lora_params: List[torch.nn.Parameter] = []
    bridge_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if id(param) in bridge_param_ids:
            bridge_params.append(param)
        elif param.requires_grad:
            if "lora_" not in name:
                raise RuntimeError(f"unexpected trainable non-LoRA/non-SBA parameter: {name}")
            lora_params.append(param)

    if not lora_params:
        raise RuntimeError("no LoRA parameters are trainable")
    if not bridge_params:
        raise RuntimeError("no SBA bridge parameters are trainable")

    return lora_params, bridge_params


class CausalLMCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        return batch


def load_text_dataset(args: argparse.Namespace):
    if args.train_file is not None:
        data_path = Path(args.train_file)
        suffix = data_path.suffix.lower()

        if suffix == ".txt":
            extension = "text"
        elif suffix in {".json", ".jsonl"}:
            extension = "json"
        elif suffix == ".csv":
            extension = "csv"
        else:
            raise ValueError(f"unsupported --train_file extension: {suffix}")

        return load_dataset(extension, data_files={"train": str(data_path)})["train"]

    return load_dataset(args.dataset_name, args.dataset_config, split=args.train_split)


def tokenize_dataset(raw_dataset, tokenizer, args: argparse.Namespace):
    column_names = list(raw_dataset.column_names)
    if args.text_column not in column_names:
        if len(column_names) == 1:
            text_column = column_names[0]
        else:
            raise ValueError(f"text column {args.text_column!r} not found; available columns: {column_names}")
    else:
        text_column = args.text_column

    eos = tokenizer.eos_token or ""

    def tokenize_batch(examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        texts = []
        for value in examples[text_column]:
            if value is None:
                text = ""
            elif isinstance(value, str):
                text = value
            else:
                text = str(value)

            if args.append_eos and eos and not text.endswith(eos):
                text = text + eos

            texts.append(text)

        return tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            return_attention_mask=True,
        )

    tokenized_dataset = raw_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=column_names,
        num_proc=args.preprocessing_num_workers,
        desc="Tokenizing",
    )

    def has_two_tokens(example: Dict[str, Any]) -> bool:
        return len(example["input_ids"]) >= 2

    tokenized_dataset = tokenized_dataset.filter(
        has_two_tokens,
        num_proc=args.preprocessing_num_workers,
        desc="Filtering short examples",
    )

    return tokenized_dataset


@dataclass(frozen=True)
class StageState:
    name: str
    local_step: int
    progress: float
    bias: float
    kl_weight: float
    ce_weight: float
    rbos_weight: float
    row_l1_weight: float


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def lerp(start: float, end: float, t: float) -> float:
    return float(start) + (float(end) - float(start)) * float(t)


def progress_for(local_step: int, stage_steps: int) -> float:
    if stage_steps <= 1:
        return 1.0
    return min(1.0, max(0.0, local_step / float(stage_steps - 1)))


def stage_state(global_step: int, args: argparse.Namespace) -> StageState:
    s0 = args.stage0_steps
    s1 = args.stage1_steps
    s2 = args.stage2_steps

    if global_step < s0:
        p = progress_for(global_step, s0)
        return StageState(
            name="stage0_preservation",
            local_step=global_step,
            progress=p,
            bias=4.0,
            kl_weight=1.0,
            ce_weight=0.0,
            rbos_weight=0.0,
            row_l1_weight=0.0,
        )

    if global_step < s0 + s1:
        local = global_step - s0
        p = progress_for(local, s1)
        ramp = smoothstep(p)
        return StageState(
            name="stage1_transition",
            local_step=local,
            progress=p,
            bias=lerp(4.0, 1.5, p),
            kl_weight=1.0,
            ce_weight=0.0,
            rbos_weight=args.rbos_loss_weight * ramp,
            row_l1_weight=args.row_l1_loss_weight * ramp,
        )

    local = global_step - s0 - s1
    p = progress_for(local, s2)
    return StageState(
        name="stage2_recovery",
        local_step=local,
        progress=p,
        bias=lerp(1.5, 1.0, p),
        kl_weight=lerp(1.0, args.stage2_final_kl_weight, p),
        ce_weight=args.ce_weight,
        rbos_weight=args.rbos_loss_weight,
        row_l1_weight=args.row_l1_loss_weight,
    )


def valid_shift_mask(labels: torch.Tensor) -> torch.Tensor:
    return labels[:, 1:].ne(-100)


def chunked_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    student = student_logits[:, :-1, :]
    teacher = teacher_logits[:, :-1, :]
    valid = valid_shift_mask(labels)
    vocab = student.shape[-1]

    flat_valid = valid.reshape(-1)
    valid_indices = flat_valid.nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return student_logits.float().sum() * 0.0

    student_flat = student.reshape(-1, vocab)
    teacher_flat = teacher.reshape(-1, vocab)
    total = student_logits.float().sum() * 0.0
    denom = 0
    temp = float(temperature)

    for idx in valid_indices.split(chunk_size):
        s = student_flat.index_select(0, idx).float() / temp
        with torch.no_grad():
            t = teacher_flat.index_select(0, idx).float() / temp
            teacher_log_probs = F.log_softmax(t, dim=-1)
            teacher_probs = teacher_log_probs.exp()

        student_log_probs = F.log_softmax(s, dim=-1)
        kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        total = total + kl.sum() * (temp * temp)
        denom += int(idx.numel())

    return total / max(denom, 1)


def chunked_cross_entropy(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    shift_logits = student_logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    vocab = shift_logits.shape[-1]

    flat_valid = valid.reshape(-1)
    valid_indices = flat_valid.nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return student_logits.float().sum() * 0.0

    logits_flat = shift_logits.reshape(-1, vocab)
    labels_flat = shift_labels.reshape(-1)
    total = student_logits.float().sum() * 0.0
    denom = 0

    for idx in valid_indices.split(chunk_size):
        logits_chunk = logits_flat.index_select(0, idx).float()
        labels_chunk = labels_flat.index_select(0, idx)
        total = total + F.cross_entropy(logits_chunk, labels_chunk, reduction="sum")
        denom += int(idx.numel())

    return total / max(denom, 1)


def extract_logits(model_output) -> torch.Tensor:
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, dict) and "logits" in model_output:
        return model_output["logits"]
    raise TypeError("model output does not contain logits")


def save_conversion_checkpoint(
    accelerator: Accelerator,
    student,
    tokenizer,
    args: argparse.Namespace,
    *,
    step: int,
    final: bool,
) -> None:
    accelerator.wait_for_everyone()
    output_dir = args.output_dir if final else os.path.join(args.output_dir, f"checkpoint-{step}")

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

        unwrapped = accelerator.unwrap_model(student)
        unwrapped.save_pretrained(output_dir, save_function=accelerator.save)
        save_sba_bridge_state(unwrapped, output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "conversion_state.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": step,
                    "final": final,
                    "stage": stage_state(max(step - 1, 0), args).__dict__ if step > 0 else None,
                    "model_name_or_path": args.model_name_or_path,
                    "teacher_name_or_path": args.teacher_name_or_path or args.model_name_or_path,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "attention_softcap": args.attention_softcap,
                    "row_l1_target": args.row_l1_target,
                },
                f,
                indent=2,
                sort_keys=True,
            )

        if final and args.save_final_merged_lora:
            merged_dir = os.path.join(output_dir, "merged_lora_weights")
            os.makedirs(merged_dir, exist_ok=True)

            merged = unwrapped.merge_and_unload()
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)

    accelerator.wait_for_everyone()


def count_trainable_parameters(model) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def finite_float(tensor: torch.Tensor) -> float:
    value = tensor.detach().float()
    if value.ndim:
        value = value.mean()
    return float(value.cpu().item())


def main() -> None:
    args = parse_args()

    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )

    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "conversion_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=bool(args.trust_remote_code),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    tokenizer.model_max_length = min(int(getattr(tokenizer, "model_max_length", args.max_seq_length)), args.max_seq_length)

    raw_dataset = load_text_dataset(args)
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer, args)

    data_collator = CausalLMCollator(tokenizer, pad_to_multiple_of=args.pad_to_multiple_of)
    train_dataloader = DataLoader(
        tokenized_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if len(train_dataloader) == 0:
        raise RuntimeError("the tokenized training dataloader is empty")

    # Load only a single model instance to save memory. We toggle the SBA patch
    # and LoRA adapter layer to perform teacher and student passes on the same weights.
    student = load_causal_model(args.model_name_or_path, args)
    student, patched_count, lora_targets = attach_lora_and_sba(student, args)

    if args.gradient_checkpointing:
        if hasattr(student, "gradient_checkpointing_enable"):
            try:
                student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                student.gradient_checkpointing_enable()

        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()

        for cfg in (getattr(student, "config", None), getattr(getattr(student, "config", None), "text_config", None)):
            if cfg is not None and hasattr(cfg, "use_cache"):
                cfg.use_cache = False

    lora_params, bridge_params = split_trainable_parameters(student)
    optimizer = AdamW(
        [
            {"params": lora_params, "lr": args.learning_rate, "weight_decay": args.weight_decay, "name": "lora"},
            {"params": bridge_params, "lr": args.bridge_learning_rate, "weight_decay": 0.0, "name": "sba_bridge"},
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    total_stage_steps = args.stage0_steps + args.stage1_steps + args.stage2_steps
    total_train_steps = args.max_train_steps or total_stage_steps

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_train_steps,
    )

    student, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        student,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    bridge_param_ids = {id(param) for _, param in named_sba_bridge_parameters(student)}
    trainable_params_for_clip = [p for p in student.parameters() if p.requires_grad]

    if accelerator.is_main_process:
        trainable, total = count_trainable_parameters(student)
        target_count = len(lora_targets) if isinstance(lora_targets, list) else len(re.findall(r"\|", lora_targets)) + 1
        print(
            json.dumps(
                {
                    "patched_sba_attention_modules": patched_count,
                    "lora_target_count": target_count,
                    "trainable_parameters": trainable,
                    "total_parameters_visible_to_process": total,
                    "trainable_fraction": trainable / max(total, 1),
                    "total_train_steps": total_train_steps,
                },
                indent=2,
            )
        )

    completed_steps = 0
    progress = tqdm(
        total=total_train_steps,
        initial=completed_steps,
        disable=not accelerator.is_main_process,
        desc="SBA bridge conversion",
    )

    dataloader_iter = iter(train_dataloader)
    rolling: Dict[str, float] = {}
    rolling_count = 0

    while completed_steps < total_train_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_dataloader)
            batch = next(dataloader_iter)

        stage = stage_state(completed_steps, args)

        structural_enabled = (stage.rbos_weight > 0.0) or (stage.row_l1_weight > 0.0)
        unwrapped_student = accelerator.unwrap_model(student)

        with accelerator.accumulate(student):
            with accelerator.autocast():
                labels = batch["labels"]
                model_inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "use_cache": False,
                    "return_dict": True,
                }

                # 1. Teacher Pass: Disable LoRA adapters and bypass SBA attention
                set_sba_patch_bypass(unwrapped_student, True)
                with unwrapped_student.disable_adapter():
                    with torch.no_grad():
                        teacher_logits = extract_logits(student(**model_inputs))

                # 2. Student Pass: Enable LoRA adapters and reactivate SBA attention
                set_sba_patch_bypass(unwrapped_student, False)

                set_sba_bridge_bias(unwrapped_student, stage.bias, zero_delta=stage.name == "stage0_preservation")
                if stage.name == "stage0_preservation":
                    reset_sba_bridge_deltas(unwrapped_student, reset_bias_delta=True, reset_tau=args.initial_tau)

                set_sba_structure_loss_enabled(unwrapped_student, structural_enabled)
                clear_sba_attention_stats(unwrapped_student)
                set_sba_padding_mask(unwrapped_student, batch["attention_mask"])
                student_logits = extract_logits(student(**model_inputs))
                clear_sba_padding_mask(unwrapped_student)

                kl_loss = chunked_kl_divergence(
                    student_logits,
                    teacher_logits,
                    labels,
                    temperature=args.kl_temperature,
                    chunk_size=args.loss_chunk_size,
                )

                if stage.ce_weight > 0.0:
                    ce_loss = chunked_cross_entropy(student_logits, labels, chunk_size=args.loss_chunk_size)
                else:
                    ce_loss = student_logits.float().sum() * 0.0

                if structural_enabled:
                    struct = sba_structure_loss(student)
                    rbos_loss = struct["rbos_loss"]
                    row_l1_loss = struct["row_l1_loss"]
                    row_l1_mean = struct["row_l1_mean"]
                    rbos_mean = struct["rbos_mean"]
                else:
                    zero = student_logits.float().sum() * 0.0
                    rbos_loss = zero
                    row_l1_loss = zero
                    row_l1_mean = zero.detach()
                    rbos_mean = zero.detach()

                if stage.name == "stage0_preservation":
                    loss = kl_loss
                else:
                    loss = (
                        stage.kl_weight * kl_loss
                        + stage.ce_weight * ce_loss
                        + stage.rbos_weight * rbos_loss
                        + stage.row_l1_weight * row_l1_loss
                    )

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                if stage.name == "stage0_preservation":
                    for param in student.parameters():
                        if id(param) in bridge_param_ids:
                            param.grad = None

                if args.max_grad_norm and args.max_grad_norm > 0.0:
                    accelerator.clip_grad_norm_(trainable_params_for_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            metrics = {
                "loss": finite_float(loss),
                "kl": finite_float(kl_loss),
                "ce": finite_float(ce_loss),
                "rbos_loss": finite_float(rbos_loss),
                "row_l1_loss": finite_float(row_l1_loss),
                "rbos_mean": finite_float(rbos_mean),
                "row_l1_mean": finite_float(row_l1_mean),
                "bias": float(stage.bias),
                "kl_w": float(stage.kl_weight),
                "ce_w": float(stage.ce_weight),
                "rbos_w": float(stage.rbos_weight),
                "row_l1_w": float(stage.row_l1_weight),
            }

            for key, value in metrics.items():
                rolling[key] = rolling.get(key, 0.0) + value
            rolling_count += 1

            completed_steps += 1
            progress.update(1)

            if completed_steps % args.logging_steps == 0 or completed_steps == 1:
                averaged = {key: value / max(rolling_count, 1) for key, value in rolling.items()}
                averaged["stage"] = stage.name
                averaged["step"] = completed_steps

                if accelerator.is_main_process:
                    progress.set_postfix({k: v for k, v in averaged.items() if isinstance(v, float)})
                    print(json.dumps(averaged, sort_keys=True))

                rolling.clear()
                rolling_count = 0

            if args.save_steps and completed_steps % args.save_steps == 0:
                save_conversion_checkpoint(accelerator, student, tokenizer, args, step=completed_steps, final=False)

    progress.close()
    save_conversion_checkpoint(accelerator, student, tokenizer, args, step=completed_steps, final=True)
    accelerator.end_training()


if __name__ == "__main__":

    main()