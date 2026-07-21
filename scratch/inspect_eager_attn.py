import inspect
from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward

print("=== eager_attention_forward Source ===")
try:
    print(inspect.getsource(eager_attention_forward))
except Exception as e:
    print("Error getting source:", e)
