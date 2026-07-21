# verify_baseline_reproduces.py
"""
Concern 1 verification: Does the original step-15000 softmax checkpoint
reproduce the 1.0005 val_loss on the exact same 4k validation slice
used for the LoRA experiment?

This script runs the evaluation independently from scratch to confirm
the baseline number is not fabricated or computed differently.
"""
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_scratch_tokenized_compare import ScratchConfig, ScratchLM, evaluate
from data import build_dataloaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use same path resolution as LoRA script
    project_root = Path(__file__).resolve().parent.parent
    softmax_path = project_root / "sba_standalone/results/blackwell_125m_overnight/softmax/checkpoints/softmax_step015000_12l_12h_768d.pt"
    if not softmax_path.exists():
        softmax_path = Path("/root/sba_standalone/results/blackwell_125m_overnight/softmax/checkpoints/softmax_step015000_12l_12h_768d.pt")
    
    if not softmax_path.exists():
        print(f"ERROR: Softmax checkpoint not found at: {softmax_path}")
        sys.exit(1)

    print(f"Loading checkpoint from: {softmax_path}")
    checkpoint = torch.load(softmax_path, map_location="cpu")
    config = ScratchConfig(**checkpoint["scratch_config"])
    print(f"Config: n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}, block_size={config.block_size}")

    # Build dataloaders with EXACT same parameters as LoRA script
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = project_root / "data/open_reason_code_mix_25k.jsonl"
    if not dataset_path.exists():
        dataset_path = Path("/root/data/open_reason_code_mix_25k.jsonl")

    print(f"Loading dataset from: {dataset_path}")
    train_loader, val_loader, manifest = build_dataloaders(
        tokenizer=tokenizer,
        data_files=[str(dataset_path)],
        max_length=config.block_size,
        train_limit_per_file=21000,
        eval_limit_per_file=4000,
        batch_size=16,
        seed=42
    )
    print(f"Train examples: {manifest['train_examples']}, Eval examples: {manifest['eval_examples']}")

    # Load model as standard softmax (not SBA)
    print("Loading model as standard Softmax...")
    model = ScratchLM(config, "softmax")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Evaluate with bf16 (same as LoRA script's evaluate calls)
    print("Evaluating on validation set with bf16 autocast...")
    val_result = evaluate(model, val_loader, device, torch.bfloat16)
    print(f"\n{'='*60}")
    print(f"VERIFICATION RESULT")
    print(f"{'='*60}")
    print(f"Softmax val_loss on 4k eval slice: {val_result['loss']:.6f}")
    print(f"Expected (from LoRA experiment):    1.000500")
    print(f"Delta:                              {val_result['loss'] - 1.0005:.6f}")
    
    if abs(val_result['loss'] - 1.0005) < 0.01:
        print(f"\n✓ BASELINE REPRODUCES within 0.01 nats")
    else:
        print(f"\n✗ BASELINE DOES NOT REPRODUCE — discrepancy of {abs(val_result['loss'] - 1.0005):.4f} nats")

if __name__ == "__main__":
    main()
