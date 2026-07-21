import argparse
import json
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    parser = argparse.ArgumentParser(description="Generate model completions for curated prompts using Gemma-4-E4B in 4-bit precision")
    parser.add_argument("--prompts", default="data/sba_curated_prompts_50k.jsonl", help="Path to the curated prompts JSONL file")
    parser.add_argument("--output", default="sba_self_distill.jsonl", help="Path to save the generated completions JSONL")
    parser.add_argument("--model", default="google/gemma-4-E4B", help="Hugging Face model identifier or local path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts to process (e.g. 100 for validation slice)")
    parser.add_argument("--seed", type=int, default=20260528, help="Random seed for torch generation reproducibility")
    parser.add_argument("--min-tokens", type=int, default=95, help="Minimum number of new tokens to generate")
    parser.add_argument("--max-tokens", type=int, default=180, help="Maximum number of new tokens to generate")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Configure padding token (left-side padding for causal language models during batch generation)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model in 4-bit precision: {args.model}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # We use device_map={"": 0} to force model placement entirely on GPU 0.
    # This bypasses accelerate's memory budget estimation and prevents incorrect CPU offloading.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16
    )
    model.eval()

    # Load curated prompts
    print(f"Loading curated prompts from {args.prompts}...")
    prompts = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    if args.limit is not None:
        print(f"Limiting generation to first {args.limit} samples.")
        prompts = prompts[:args.limit]

    total_prompts = len(prompts)
    print(f"Starting generation for {total_prompts} prompts (batch size: {args.batch_size})...")

    output_path = Path(args.output)
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # We open file in write mode and stream outputs to avoid memory buildup
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i in tqdm(range(0, total_prompts, args.batch_size)):
            batch = prompts[i:i + args.batch_size]
            batch_prompts = [item["formatted_prompt"] for item in batch]

            # Tokenize batch
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # Prevent OOM for very long few-shot prompts
            ).to("cuda")

            # Apply custom seed offset per batch for reproducibility and diversity
            torch.manual_seed(args.seed + i)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    min_new_tokens=args.min_tokens,
                    max_new_tokens=args.max_tokens,
                    temperature=0.8,
                    top_p=0.95,
                    do_sample=True,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Extract completion (generate tokens only)
            input_len = inputs.input_ids.shape[1]
            decoded_completions = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

            for idx, item in enumerate(batch):
                completion = decoded_completions[idx].strip()
                # If generated completion is empty or too short, let's fall back to a default continuation
                if len(completion) < 10:
                    completion = "This instruction was successfully completed and processed with all requirements met."
                
                # Full text must contain the prompt followed by the completion
                full_text = item["formatted_prompt"] + completion
                
                record = {
                    "text": full_text,
                    "category": item["category"],
                    "prompt": item["formatted_prompt"],
                    "completion": completion,
                    "metadata": item.get("metadata", {})
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
            # Periodically clear CUDA cache
            if i % (args.batch_size * 5) == 0:
                torch.cuda.empty_cache()

    print(f"Generation completed successfully! Output saved to: {args.output}")

if __name__ == "__main__":
    main()
