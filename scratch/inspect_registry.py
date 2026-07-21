from transformers.models.gemma4.modeling_gemma4 import ALL_ATTENTION_FUNCTIONS
print("=== ALL_ATTENTION_FUNCTIONS ===")
print(type(ALL_ATTENTION_FUNCTIONS))
print(dir(ALL_ATTENTION_FUNCTIONS))
try:
    print("Keys/Registry contents:", list(ALL_ATTENTION_FUNCTIONS.keys()))
except Exception:
    try:
        print("Registry contents:", ALL_ATTENTION_FUNCTIONS.__dict__)
    except Exception as e:
        print("Error inspecting registry:", e)
