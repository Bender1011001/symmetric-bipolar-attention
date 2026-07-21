import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

print("Loading model config/weights...")
model_id = "google/gemma-4-E4B-it"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Testing LoraConfig with regex target_modules...")
try:
    # Use regex to only target the language model's standard linear layers
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        # This matches q_proj, k_proj, v_proj, o_proj inside language_model only!
        target_modules=".*\\.language_model\\..*\\.(q_proj|k_proj|v_proj|o_proj)"
    )
    peft_model = get_peft_model(model, config)
    print("SUCCESS! PEFT model wrapped successfully with regex.")
    # Print trainable parameters to verify
    peft_model.print_trainable_parameters()
except Exception as e:
    print("FAILED with regex:", e)

print("Testing LoraConfig by registering Gemma4ClippableLinear...")
try:
    # Another approach: register Gemma4ClippableLinear class in peft's custom list!
    from peft.tuners.tuners_utils import BaseTuner
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    
    # We can add Gemma4ClippableLinear to the set of supported modules!
    # PEFT checks isinstance or class name. Let's see how PEFT checks it:
    # In PEFT, we can modify LoraModel._subclasses_to_register or similar
    # or just add to the allowed modules list.
    pass
except Exception as e:
    print("Error during class registration test:", e)
