import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from patch import apply_sba_patch, collect_patch_report

print("Loading model and tokenizer...")
model_id = "google/gemma-4-E4B-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model in bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Applying SBA patch...")
try:
    report = apply_sba_patch(model)
    print("SBA Patch applied successfully! Report:")
    print(report)
except Exception as e:
    print("Failed to apply SBA patch:", e)
    import sys
    sys.exit(1)

# Run a test forward pass
print("Running dummy forward pass...")
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

try:
    with torch.no_grad():
        outputs = model(**inputs)
    print("Forward pass successful!")
    print("Loss:", outputs.loss)
    print("Logits shape:", outputs.logits.shape)
    print("Are logits finite?", torch.isfinite(outputs.logits).all().item())
except Exception as e:
    print("Forward pass FAILED:", e)
    import sys
    sys.exit(1)

print("All tests passed successfully!")
