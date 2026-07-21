import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = r"D:\hf_cache\hub\models--google--gemma-4-E4B\snapshots\7aa32e6889efd6300124851b164f8b364314c3d8"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model in 4-bit...")
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
    low_cpu_mem_usage=True,
    dtype=torch.bfloat16,
    disable_mmap=True
)
print("Model loaded successfully!")
print(model)
