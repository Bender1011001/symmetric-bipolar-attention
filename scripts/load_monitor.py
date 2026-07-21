import time
import threading
import torch
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def get_vram():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True
        )
        return int(res.stdout.strip())
    except:
        return -1

def load_model():
    model_id = "google/gemma-4-E4B"
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Loading model...", flush=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map={"": 0},
        low_cpu_mem_usage=True
    )
    print("Loaded!", flush=True)

def main():
    t = threading.Thread(target=load_model)
    t.start()
    
    while t.is_alive():
        vram = get_vram()
        print(f"[VRAM Monitor] Used: {vram} MB", flush=True)
        time.sleep(1.0)
    
    t.join()

if __name__ == "__main__":
    main()
