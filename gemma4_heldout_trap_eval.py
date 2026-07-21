# gemma4_heldout_trap_eval.py
import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ensure we can load local patch.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch
from patch import patch_gemma4_sba, load_sba_bridge_state

# Define held-out traps and matching controls (flat conversational format, no XML tags)
TEST_CASES = [
    {
        "name": "bat_ball_trap",
        "prompt": "Question: A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. What is the price of the ball?\nAnswer:",
        "is_trap": True,
        "trap_concept": "bat_and_ball",
        "target_keywords": ["5 cents", "0.05", "$0.05", "five cents"]
    },
    {
        "name": "bat_ball_control",
        "prompt": "Question: A bat and a ball together cost $1.10. The bat costs $0.70. What is the price of the ball?\nAnswer:",
        "is_trap": False,
        "trap_concept": "bat_and_ball",
        "target_keywords": ["40 cents", "0.40", "$0.40", "forty cents"]
    },
    {
        "name": "widgets_trap",
        "prompt": "Question: If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?\nAnswer:",
        "is_trap": True,
        "trap_concept": "widgets",
        "target_keywords": ["5 minutes", "five minutes", "5 min"]
    },
    {
        "name": "widgets_control",
        "prompt": "Question: If it takes 5 machines 5 minutes to make 5 widgets, how many widgets can 5 machines make in 10 minutes?\nAnswer:",
        "is_trap": False,
        "trap_concept": "widgets",
        "target_keywords": ["10 widgets", "ten widgets", "10"]
    },
    {
        "name": "lilypad_trap",
        "prompt": "Question: In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?\nAnswer:",
        "is_trap": True,
        "trap_concept": "lilypads",
        "target_keywords": ["47 days", "forty-seven days", "47"]
    },
    {
        "name": "lilypad_control",
        "prompt": "Question: A patch of lily pads covers a lake in 48 days. It grows at a constant rate of 2 square meters per day. On which day does it cover half of the lake if the lake is 96 square meters?\nAnswer:",
        "is_trap": False,
        "trap_concept": "lilypads",
        "target_keywords": ["24 days", "twenty-four days", "24"]
    },
    {
        "name": "surgeon_trap",
        "prompt": "Question: A father and son are in a car crash. The father is killed instantly. The boy is rushed to the hospital. The surgeon looks at him and says 'I cannot operate on this boy, he is my son.' Who was the surgeon?\nAnswer:",
        "is_trap": True,
        "trap_concept": "surgeon",
        "target_keywords": ["mother", "mom", "female surgeon", "parent"]
    },
    {
        "name": "surgeon_control",
        "prompt": "Question: A doctor is the brother of a lawyer. The lawyer says they have no brother. Who is the doctor?\nAnswer:",
        "is_trap": False,
        "trap_concept": "surgeon",
        "target_keywords": ["sister", "she", "woman", "female"]
    },
    {
        "name": "rule_exception_trap",
        "prompt": "Question: All items in the clearance bin are non-returnable. You bought a widget from the clearance bin. Can you return it?\nAnswer:",
        "is_trap": True,
        "trap_concept": "clearance",
        "target_keywords": ["no", "cannot", "not returnable", "impossible"]
    },
    {
        "name": "rule_exception_control",
        "prompt": "Question: All items in the store are returnable except those bought on clearance. You bought a widget from the regular shelf. Can you return it?\nAnswer:",
        "is_trap": False,
        "trap_concept": "clearance",
        "target_keywords": ["yes", "can return", "refundable", "possible"]
    },
    {
        "name": "shipping_priority_trap",
        "prompt": "Question: Normally, shipping takes 3-5 days. However, priority shipping overrides normal shipping to take 1 day. A package is shipped via priority shipping. How long does it take?\nAnswer:",
        "is_trap": True,
        "trap_concept": "priority",
        "target_keywords": ["1 day", "one day"]
    },
    {
        "name": "shipping_priority_control",
        "prompt": "Question: Shipping takes 3-5 days. Priority shipping takes 1 day. A package is shipped via regular shipping. How long does it take?\nAnswer:",
        "is_trap": False,
        "trap_concept": "priority",
        "target_keywords": ["3-5 days", "3 to 5 days", "three to five days"]
    }
]

captured_layer_metrics = []
original_stable_sinh_over_sum_cosh = patch._stable_sinh_over_sum_cosh

def hooked_stable_sinh_over_sum_cosh(x, valid):
    weights = original_stable_sinh_over_sum_cosh(x, valid)
    with torch.no_grad():
        neg_mask = (weights < 0) & valid
        valid_edges = valid.sum().clamp_min(1)
        neg_fraction = neg_mask.sum().item() / valid_edges.item()
        neg_sum_abs = weights[neg_mask].abs().sum().item() / valid_edges.item()
        captured_layer_metrics.append({
            "neg_fraction": neg_fraction,
            "neg_sum_abs": neg_sum_abs
        })
    return weights

patch._stable_sinh_over_sum_cosh = hooked_stable_sinh_over_sum_cosh

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate held-out traps and controls on Gemma 4 SBA checkpoint")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-4-E4B")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="PEFT SBA adapter checkpoint directory")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )

    print("Patching model with SBA...")
    patch_gemma4_sba(model)

    print(f"Loading PEFT adapter weights from {args.checkpoint_dir}...")
    model = PeftModel.from_pretrained(model, args.checkpoint_dir, adapter_name="eval_adapter")

    bridge_state_path = os.path.join(args.checkpoint_dir, "sba_bridge_v3.pt")
    if os.path.exists(bridge_state_path):
        print(f"Loading SBA bridge parameters from {bridge_state_path}...")
        load_sba_bridge_state(model, bridge_state_path)
    else:
        raise FileNotFoundError(f"SBA bridge state not found at {bridge_state_path}")

    # Set to eval mode
    model.eval()

    # Disable KV cache because we need to hook into the full attention matrix calculation
    # during generate forward passes
    for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False

    print("\nRunning held-out trap evaluation...")
    results = []

    for case in TEST_CASES:
        name = case["name"]
        prompt = case["prompt"]
        is_trap = case["is_trap"]
        keywords = case["target_keywords"]

        captured_layer_metrics.clear()
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        # Generate response (do_sample=False to make it deterministic, use_cache=False to compute full matrices)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=48,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_ids = outputs[0][prompt_len:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Check accuracy
        response_lower = response_text.lower()
        correct = any(kw.lower() in response_lower for kw in keywords)

        # Process captured metrics
        if len(captured_layer_metrics) > 0:
            avg_neg_fraction = sum(m["neg_fraction"] for m in captured_layer_metrics) / len(captured_layer_metrics)
            avg_neg_sum_abs = sum(m["neg_sum_abs"] for m in captured_layer_metrics) / len(captured_layer_metrics)
            max_neg_fraction = max(m["neg_fraction"] for m in captured_layer_metrics)
        else:
            avg_neg_fraction, avg_neg_sum_abs, max_neg_fraction = 0.0, 0.0, 0.0

        results.append({
            "name": name,
            "prompt": prompt,
            "is_trap": is_trap,
            "concept": case["trap_concept"],
            "response": response_text,
            "correct": correct,
            "avg_neg_fraction": avg_neg_fraction,
            "avg_neg_sum_abs": avg_neg_sum_abs,
            "max_neg_fraction": max_neg_fraction
        })

        label = "TRAP" if is_trap else "CONTROL"
        status = "PASSED" if correct else "FAILED"
        print(f"[{label}] {name:<22} | {status} | Neg Frac: {avg_neg_fraction*100:5.2f}% | Response: {response_text[:60]}...")

    # Calculate summaries
    trap_results = [r for r in results if r["is_trap"]]
    control_results = [r for r in results if not r["is_trap"]]

    avg_trap_neg_frac = sum(r["avg_neg_fraction"] for r in trap_results) / len(trap_results)
    avg_control_neg_frac = sum(r["avg_neg_fraction"] for r in control_results) / len(control_results)
    
    avg_trap_neg_sum = sum(r["avg_neg_sum_abs"] for r in trap_results) / len(trap_results)
    avg_control_neg_sum = sum(r["avg_neg_sum_abs"] for r in control_results) / len(control_results)

    trap_acc = sum(1 for r in trap_results if r["correct"]) / len(trap_results)
    control_acc = sum(1 for r in control_results if r["correct"]) / len(control_results)

    # R_inhibit ratio
    r_inhibit_frac = avg_trap_neg_frac / max(avg_control_neg_frac, 1e-9)
    r_inhibit_sum = avg_trap_neg_sum / max(avg_control_neg_sum, 1e-9)

    print("\n=========================================")
    print("HELDOUT TRAP EVALUATION SUMMARY:")
    print(f"  Trap Accuracy:    {trap_acc*100:.1f}%")
    print(f"  Control Accuracy: {control_acc*100:.1f}%")
    print(f"  Average Neg Fraction (Traps):    {avg_trap_neg_frac*100:.3f}%")
    print(f"  Average Neg Fraction (Controls): {avg_control_neg_frac*100:.3f}%")
    print(f"  R_inhibit (Fraction Ratio):      {r_inhibit_frac:.4f}")
    print(f"  R_inhibit (Abs Sum Ratio):       {r_inhibit_sum:.4f}")
    print("=========================================")
    if r_inhibit_frac >= 1.5:
        print("SUCCESS: Clear evidence of generalized cognitive inhibition on trap prompts!")
    else:
        print("WARNING: Flat or low inhibition ratio. Attention weights may be template-locked.")
    print("=========================================")

    output_data = {
        "checkpoint_dir": args.checkpoint_dir,
        "trap_accuracy": trap_acc,
        "control_accuracy": control_acc,
        "avg_trap_neg_fraction": avg_trap_neg_frac,
        "avg_control_neg_fraction": avg_control_neg_frac,
        "avg_trap_neg_sum_abs": avg_trap_neg_sum,
        "avg_control_neg_sum_abs": avg_control_neg_sum,
        "r_inhibit_fraction_ratio": r_inhibit_frac,
        "r_inhibit_sum_ratio": r_inhibit_sum,
        "results": results
    }

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
