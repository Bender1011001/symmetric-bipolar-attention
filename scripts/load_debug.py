import sys
import torch
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    model_id = "google/gemma-4-E4B"
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Tokenizer loaded.", flush=True)

    print("Configuring bitsandbytes...", flush=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    try:
        print("Starting AutoModelForCausalLM.from_pretrained...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully!", flush=True)
        print(model, flush=True)
    except Exception as e:
        print("EXCEPTION ENCOUNTERED:", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
