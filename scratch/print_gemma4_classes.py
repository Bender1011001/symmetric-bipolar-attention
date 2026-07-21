import torch
from transformers import AutoModelForCausalLM

print("Loading google/gemma-4-E4B-it...")
try:
    # Let's import the model class and print its structure
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E4B-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("SUCCESS: Model loaded!")
    attention_classes = set()
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if "Attention" in class_name:
            attention_classes.add(class_name)
            print(f"Module: {name} | Class: {class_name}")
    print("Unique attention classes found:", list(attention_classes))
except Exception as e:
    print("FAILED loading model:", e)
