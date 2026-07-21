from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_scratch_compare import EOS_TOKEN, ScratchConfig, ScratchLM


DEFAULT_PROMPTS = [
    "Question:\nSolve for x: 2x + 5 = 17.\n\nAnswer:\n",
    "Question:\nWrite a Python function that returns the factorial of n.\n\nAnswer:\n",
    "Question:\nIf a list is sorted, how can binary search find a target value?\n\nAnswer:\n",
]


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    prompt_path = Path(path)
    if prompt_path.suffix.lower() == ".jsonl":
        prompts: list[str] = []
        with prompt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompts.append(str(item.get("prompt", "")))
        return prompts
    return [chunk.strip() for chunk in prompt_path.read_text(encoding="utf-8").split("\n---\n") if chunk.strip()]


def encode(text: str, device: torch.device) -> torch.Tensor:
    return torch.tensor([list(text.encode("utf-8", errors="ignore"))], dtype=torch.long, device=device)


def decode(tokens: list[int]) -> str:
    raw = bytes([token for token in tokens if 0 <= token < 256])
    return raw.decode("utf-8", errors="replace")


@torch.no_grad()
def sample_one(
    model: ScratchLM,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    model.eval()
    ids = encode(prompt, device)
    generated: list[int] = []

    # 1. Prefill step
    window = ids[:, -model.config.block_size :]
    logits, _, past_key_values = model(window, use_cache=True)

    next_logits = logits[:, -1, :].float()
    next_logits[:, EOS_TOKEN] = -float("inf")
    if temperature <= 0:
        next_id = int(torch.argmax(next_logits, dim=-1).item())
    else:
        next_logits = next_logits / temperature
        if top_k > 0:
            values, indices = torch.topk(next_logits, k=min(top_k, next_logits.shape[-1]), dim=-1)
            probs = F.softmax(values, dim=-1)
            chosen = torch.multinomial(probs, num_samples=1)
            next_id = int(indices.gather(-1, chosen).item())
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

    generated.append(next_id)
    ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

    current_pos = ids.shape[1] - 1

    # 2. Incremental generation steps
    for _ in range(max_new_tokens - 1):
        cache_len = past_key_values[0][0].shape[-2]

        if cache_len >= model.config.block_size - 1:
            new_past_key_values = []
            for k, v in past_key_values:
                # FIX SINK DELETION: Keep index 0 (BOS sink), drop index 1 (the oldest non-BOS token)
                k_kept = torch.cat([k[:, :, :1, :], k[:, :, 2:, :]], dim=-2)
                v_kept = torch.cat([v[:, :, :1, :], v[:, :, 2:, :]], dim=-2)
                new_past_key_values.append((k_kept, v_kept))
            past_key_values = new_past_key_values

        current_input = ids[:, -1:]
        pos_id = torch.tensor([[current_pos]], dtype=torch.long, device=device)
        logits, _, past_key_values = model(
            current_input, past_key_values=past_key_values, use_cache=True, position_ids=pos_id
        )

        next_logits = logits[:, -1, :].float()
        next_logits[:, EOS_TOKEN] = -float("inf")
        if temperature <= 0:
            next_id = int(torch.argmax(next_logits, dim=-1).item())
        else:
            next_logits = next_logits / temperature
            if top_k > 0:
                values, indices = torch.topk(next_logits, k=min(top_k, next_logits.shape[-1]), dim=-1)
                probs = F.softmax(values, dim=-1)
                chosen = torch.multinomial(probs, num_samples=1)
                next_id = int(indices.gather(-1, chosen).item())
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        current_pos += 1

    return decode(generated)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ScratchLM, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") in {"sba-bridge-v3", "sba-standalone-scratch-lm-v3-tokenized"}:
        from run_softmax_bridge_conversion import load_centered_sba_model
        model, _ = load_centered_sba_model(checkpoint_path, device)
        return model, checkpoint
    config = ScratchConfig(**checkpoint["scratch_config"])
    sba_impl = checkpoint.get("sba_impl") or checkpoint.get("args", {}).get("sba_impl") or "eager"
    model = ScratchLM(config, checkpoint["attention"], sba_impl=sba_impl)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text from a standalone SBA/softmax scratch checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(Path(args.checkpoint), device)
    prompts = load_prompts(args.prompts)
    outputs = []
    for prompt in prompts:
        outputs.append(
            {
                "prompt": prompt,
                "completion": sample_one(
                    model=model,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                ),
            }
        )
    payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint.get("step"),
        "attention": checkpoint.get("attention"),
        "sba_impl": checkpoint.get("sba_impl"),
        "scratch_config": checkpoint.get("scratch_config"),
        "samples": outputs,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
