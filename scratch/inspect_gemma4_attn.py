import inspect
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

print("=== Gemma4TextAttention Source ===")
try:
    print(inspect.getsource(Gemma4TextAttention))
except Exception as e:
    print("Error getting source:", e)
