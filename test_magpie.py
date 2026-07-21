from datasets import load_dataset
try:
    ds = load_dataset('Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B', split='train')
    print("Successfully loaded. Columns:", ds.column_names)
    print("Sample keys:", ds[0].keys())
    print("Sample prompt:", ds[0].get("instruction") or ds[0].get("prompt"))
except Exception as e:
    print("Failed to load Magpie:", e)
