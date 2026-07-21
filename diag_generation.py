import torch
from pathlib import Path
from transformers import AutoTokenizer
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_scratch_tokenized_checkpoint import load_model, generate_completion, DEFAULT_TASKS

def generate_no_cache(model, tokenizer, prompt, device, max_new_tokens):
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    generated = []
    for _ in range(max_new_tokens):
        window = ids_tensor[:, -model.config.block_size:]
        with torch.no_grad():
            logits, _ = model(window, use_cache=False)
        next_logits = logits[:, -1, :].float()
        next_id = int(torch.argmax(next_logits, dim=-1).item())
        if next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)
        ids_tensor = torch.cat([ids_tensor, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        
    return tokenizer.decode(generated, skip_special_tokens=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path("sba_standalone/results/homotopy_conversion_v5/homotopy_step004000_12l_12h_768d.pt")
    
    print("Loading model...")
    model, checkpoint = load_model(checkpoint_path, device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    for task in DEFAULT_TASKS:
        print(f"\n==================================================")
        print(f"Task ID: {task.task_id}")
        print(f"Prompt: {repr(task.prompt)}")
        print(f"==================================================")
        
        comp_cache = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=task.prompt,
            device=device,
            max_new_tokens=150,
            temperature=0.0,
            top_k=0
        )
        print("\n--- WITH KV Cache ---")
        print(repr(comp_cache))
        
        comp_no_cache = generate_no_cache(
            model=model,
            tokenizer=tokenizer,
            prompt=task.prompt,
            device=device,
            max_new_tokens=150
        )
        print("\n--- WITHOUT KV Cache ---")
        print(repr(comp_no_cache))
        
        if comp_cache == comp_no_cache:
            print("\n>>> MATCH: YES")
        else:
            print("\n>>> MATCH: NO")

if __name__ == "__main__":
    main()
