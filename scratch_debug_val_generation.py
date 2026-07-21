import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
from run_scratch_tokenized_compare import ScratchConfig, ScratchLM
from evaluate_scratch_tokenized_checkpoint import generate_completion

def test_validation_prompts(checkpoint_path_str: str):
    print(f"\n=== Testing Checkpoint on Validation Prompts: {checkpoint_path_str} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path_str)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ScratchConfig(**checkpoint["scratch_config"])
    sba_impl = checkpoint.get("sba_impl") or checkpoint.get("args", {}).get("sba_impl") or "sdpa"
    model = ScratchLM(config, checkpoint["attention"], sba_impl=sba_impl)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load first 3 validation examples from GSM8k (after the training limit of 120)
    val_examples = []
    with Path("data/task0_gsm8k_cot.jsonl").open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= 120 and len(val_examples) < 3:
                val_examples.append(json.loads(line))
                
    for i, ex in enumerate(val_examples):
        prompt = f"Question:\n{ex['prompt'].strip()}\n\nAnswer:\n"
        print(f"\n--- Validation Prompt {i+1} ---")
        print(prompt)
        print(f"Expected Response:\n{ex['response']}")
        
        completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=256,
            temperature=0.0,
            top_k=0
        )
        print(f"Generated Completion:\n{completion}")

if __name__ == "__main__":
    test_validation_prompts("sba_standalone/results/tokenized_30m_seed42/softmax/checkpoints/softmax_step001000_6l_6h_384d.pt")
    test_validation_prompts("sba_standalone/results/tokenized_30m_seed42/sba/checkpoints/sba_step001000_6l_6h_384d.pt")
