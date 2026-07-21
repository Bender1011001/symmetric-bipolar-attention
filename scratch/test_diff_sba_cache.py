import sys
from pathlib import Path
sys.path.insert(0, "E:/code.projects/CASCADES--continual-PEFT-for-Local-LLMs")

import torch
from run_scratch_tokenized_compare import ScratchConfig, ScratchLM

@torch.no_grad()
def main():
    config = ScratchConfig(
        vocab_size=257,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0
    )
    test_cases = [
        ("softmax", "sdpa"),
        ("sba", "sdpa"),
        ("sba", "eager"),
        ("diff_sba", "sdpa"),
    ]
    for attention, sba_impl in test_cases:
        model = ScratchLM(config, attention=attention, sba_impl=sba_impl)
        model.eval()
        x = torch.randint(0, 257, (2, 5))
        logits_full, _ = model(x)
        
        past = None
        logits_inc = []
        for i in range(5):
            logits_step, _, past = model(x[:, i:i+1], past_key_values=past, use_cache=True)
            logits_inc.append(logits_step)
            
        logits_inc = torch.cat(logits_inc, dim=1)
        diff = (logits_full - logits_inc).abs().max().item()
        print(f"[{attention}/{sba_impl}] Max difference: {diff}")
        assert diff < 1e-4, f"Mismatch found in [{attention}/{sba_impl}]: {diff}"
    print("ALL CACHE TESTS PASSED!")

if __name__ == "__main__":
    main()
