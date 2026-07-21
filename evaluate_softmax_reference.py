# evaluate_softmax_reference.py
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

    project_root = Path(__file__).resolve().parent.parent
    softmax_path = project_root / "sba_standalone/results/blackwell_125m_overnight/softmax/checkpoints/softmax_step015000_12l_12h_768d.pt"

    if not softmax_path.exists():
        print(f"Pre-trained Softmax checkpoint not found at: {softmax_path}")
        sys.exit(1)

    print("Loading pre-trained Softmax checkpoint...")
    checkpoint = torch.load(softmax_path, map_location="cpu")
    config = ScratchConfig(**checkpoint["scratch_config"])

    print("Building dataloaders for open_reason_code_mix_25k.jsonl...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = project_root / "data/open_reason_code_mix_25k.jsonl"
    _, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        data_files=[str(dataset_path)],
        max_length=config.block_size,
        train_limit_per_file=21000,
        eval_limit_per_file=4000,
        batch_size=16,
        seed=42
    )

    print("Initializing original Softmax model...")
    model = ScratchLM(config, "softmax")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    print("Evaluating original Softmax model on validation set...")
    val = evaluate(model, val_loader, device, torch.bfloat16)
    print(f"\n=== Reference Results ===")
    print(f"Original Softmax Val Loss: {val['loss']:.4f}")

if __name__ == "__main__":
    main()
