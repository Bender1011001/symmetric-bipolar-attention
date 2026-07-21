import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from run_scratch_compare import ScratchConfig, ScratchLM, CausalSelfAttention

@torch.no_grad()
def test_kv_cache_equivalence():
    print("Testing KV-cache equivalence...")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Small model configuration
    config = ScratchConfig(
        vocab_size=257,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0
    )

    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    for attention in ["softmax", "sba"]:
        for sba_impl in ["sdpa", "eager"] if attention == "sba" else [None]:
            impl_str = f"_{sba_impl}" if sba_impl else ""
            name = f"{attention}{impl_str}"
            print(f"\n--- Checking: {name} ---")

            # Create model
            model = ScratchLM(config, attention=attention, sba_impl=sba_impl or "sdpa")
            model.to(device)
            model.eval()

            # 1. Full sequence forward pass
            logits_full, _ = model(input_ids)

            # 2. Incremental forward pass using integrated cache
            past_key_values = None
            logits_incremental = []

            for t in range(seq_len):
                current_input = input_ids[:, t:t+1] # shape (batch, 1)
                logits_step, _, past_key_values = model(current_input, past_key_values=past_key_values, use_cache=True)
                logits_incremental.append(logits_step)
                
            logits_incremental = torch.cat(logits_incremental, dim=1)
            
            # Compare logits
            diff = (logits_full - logits_incremental).abs().max().item()
            print(f"Max absolute difference in logits: {diff}")
            assert diff < 1e-4, f"Cache equivalence failed for {name}"
            print("Passed!")

if __name__ == "__main__":
    test_kv_cache_equivalence()
