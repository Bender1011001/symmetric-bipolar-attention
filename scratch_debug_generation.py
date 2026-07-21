import torch
from pathlib import Path
from transformers import AutoTokenizer
from run_scratch_tokenized_compare import ScratchConfig, ScratchLM

def debug_checkpoint(checkpoint_path_str: str):
    print(f"=== Debugging Checkpoint: {checkpoint_path_str} ===")
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
    
    prompt = "Question:\nSolve for x: 2x + 5 = 17. Give only the value of x.\n\nAnswer:\n"
    print(f"Prompt: {repr(prompt)}")
    
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(f"Prompt tokens: {input_ids}")
    print(f"Decoded prompt tokens: {[tokenizer.decode([t]) for t in input_ids]}")
    
    ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for step in range(5):
            window = ids_tensor[:, -config.block_size:]
            logits, _ = model(window)
            next_logits = logits[0, -1, :].float()
            
            # Print top 5 predicted tokens
            probs = torch.softmax(next_logits, dim=-1)
            top_vals, top_inds = torch.topk(probs, k=5)
            
            print(f"\nStep {step + 1}:")
            for prob, ind in zip(top_vals.tolist(), top_inds.tolist()):
                print(f"  Token: {ind} ({repr(tokenizer.decode([ind]))}) -> Prob: {prob:.6f}")
                
            next_id = int(top_inds[0].item())
            ids_tensor = torch.cat([ids_tensor, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

if __name__ == "__main__":
    debug_checkpoint("sba_standalone/results/tokenized_30m_seed42/softmax/checkpoints/softmax_step001000_6l_6h_384d.pt")
    debug_checkpoint("sba_standalone/results/tokenized_30m_seed42/sba/checkpoints/sba_step001000_6l_6h_384d.pt")
