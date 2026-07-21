import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ensure we can load local patch.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patch import patch_gemma4_sba, load_sba_bridge_state

def main():
    model_name = "google/gemma-4-E4B"
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "gemma4_e4b_sba_distill_bridge_b200",
        "checkpoint-5000"
    )

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading base model in 4-bit precision...")
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map={"": "cuda:0"},
        trust_remote_code=True
    )

    print(f"Base model loaded. VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    print("Monkey-patching model with SBA attention...")
    patched_count = patch_gemma4_sba(model)
    print(f"Patched {patched_count} attention modules.")

    print(f"Loading PEFT adapter weights from {checkpoint_dir}...")
    # Wrap PEFT model
    model = PeftModel.from_pretrained(model, checkpoint_dir, adapter_name="eval_adapter")

    print(f"Loading SBA bridge parameters from sba_bridge_v3.pt...")
    bridge_state_path = os.path.join(checkpoint_dir, "sba_bridge_v3.pt")
    load_sba_bridge_state(model, bridge_state_path)

    print("Running diagnostic forward pass...")
    prompt = "Question:\nSolve for x: 2x + 5 = 17.\n\nAnswer:\n"
    encoded = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    model.eval()
    with torch.no_grad():
        outputs = model(encoded.input_ids, use_cache=False)
        logits = outputs.logits
        
    print("Diagnostics complete!")
    print(f"Logits shape: {logits.shape}")
    print(f"Are logits finite? {torch.isfinite(logits).all().item()}")
    print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()
