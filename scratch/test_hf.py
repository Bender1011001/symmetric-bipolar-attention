from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained("google/gemma-4-E4B-it")
    print("SUCCESS Gemma 4 E4B config:")
    print(cfg)
except Exception as e:
    print("FAILED Gemma 4 E4B config:", e)

try:
    cfg = AutoConfig.from_pretrained("google/gemma-2-2b")
    print("SUCCESS Gemma 2 2B config:")
    print(cfg)
except Exception as e:
    print("FAILED Gemma 2 2B config:", e)
