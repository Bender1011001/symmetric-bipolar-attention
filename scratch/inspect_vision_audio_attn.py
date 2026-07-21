import inspect
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionAttention, Gemma4AudioAttention

print("=== Gemma4VisionAttention ===")
try:
    print(inspect.getsource(Gemma4VisionAttention))
except Exception as e:
    print("Error:", e)

print("=== Gemma4AudioAttention ===")
try:
    print(inspect.getsource(Gemma4AudioAttention))
except Exception as e:
    print("Error:", e)
