import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def format_prompt(item, dataset_name):
    if dataset_name == "alpaca":
        inst = item["instruction"].strip()
        inp = item["input"].strip()
        if inp:
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Response:\n"
    elif dataset_name == "gsm8k":
        q = item["question"].strip()
        prompt = f"Below is an instruction that describes a math problem. Write a step-by-step response that solves the problem.\n\n### Instruction:\n{q}\n\n### Response:\n"
    elif dataset_name == "mbpp":
        text = item["text"].strip()
        prompt = f"Below is an instruction that describes a programming task. Write a Python function that solves the task.\n\n### Instruction:\n{text}\n\n### Response:\n"
    else:
        prompt = ""
    return prompt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "google/gemma-4-E4B"
    output_path = "/root/sba_standalone/data/sba_self_distill.jsonl"
    
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    model.eval()

    # 1. Download and compile prompt list (total 10,000)
    print("Loading prompt datasets from HF...")
    prompts = []
    
    # 5,000 Alpaca prompts
    try:
        alpaca_ds = load_dataset("tatsu-lab/alpaca", split="train")
        alpaca_slice = [item for item in alpaca_ds][:5000]
        for item in alpaca_slice:
            prompts.append(format_prompt(item, "alpaca"))
    except Exception as e:
        print(f"Error loading Alpaca: {e}")
        
    # 3,000 GSM8K prompts
    try:
        gsm8k_ds = load_dataset("gsm8k", "main", split="train")
        gsm8k_slice = [item for item in gsm8k_ds][:3000]
        for item in gsm8k_slice:
            prompts.append(format_prompt(item, "gsm8k"))
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        
    # 2,000 MBPP prompts
    try:
        mbpp_ds = load_dataset("mbpp", "sanitized", split="train")
        mbpp_slice = [item for item in mbpp_ds][:2000]
        for item in mbpp_slice:
            prompts.append(format_prompt(item, "mbpp"))
    except Exception as e:
        print(f"Error loading MBPP: {e}")

    # Remove empty or failed prompts
    prompts = [p for p in prompts if p]
    print(f"Total compiled prompts: {len(prompts)}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2. Batch generate using PyTorch
    batch_size = 64
    print(f"Starting batch generation (batch size {batch_size})...")
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                # We use temperature=0.8, top_p=0.95 for high quality, diverse responses
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Extract only the generated response and join with prompt
            for prompt_text, out_tokens in zip(batch_prompts, outputs):
                # Slice tokens to exclude prompt prefix
                # (Find the first token index after prompt)
                prompt_len = inputs.input_ids.shape[1]
                gen_tokens = out_tokens[prompt_len:]
                response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                
                # Full text = Prompt + Response
                full_text = prompt_text + response_text
                
                f_out.write(json.dumps({"text": full_text}) + "\n")
                
            del inputs, outputs
            torch.cuda.empty_cache()
            
    print(f"Self-distillation data generation complete. Saved to {output_path}")

if __name__ == "__main__":
    main()
