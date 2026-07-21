# evaluate_mmlu.py
import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# Ensure we can load local patch.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch import patch_gemma4_sba, load_sba_bridge_state

MMLU_SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics',
    'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
    'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning',
    'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes',
    'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
    'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations',
    'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]

def format_subject(subject_name):
    return subject_name.replace("_", " ")

def format_example(question, choices, answer_idx, include_answer=True):
    prompt = f"Question: {question}\n"
    for label, choice in zip(["A", "B", "C", "D"], choices):
        prompt += f"{label}. {choice}\n"
    if include_answer:
        prompt += f"Answer: {['A', 'B', 'C', 'D'][answer_idx]}\n\n"
    else:
        prompt += "Answer:"
    return prompt

def gen_prompt(dev_dataset, subject_name, test_example, k=5):
    prompt = f"The following are multiple-choice questions (with answers) about {format_subject(subject_name)}.\n\n"
    for i in range(min(k, len(dev_dataset))):
        prompt += format_example(
            dev_dataset[i]["question"],
            dev_dataset[i]["choices"],
            dev_dataset[i]["answer"],
            include_answer=True
        )
    prompt += format_example(
        test_example["question"],
        test_example["choices"],
        test_example["answer"],
        include_answer=False
    )
    return prompt

def get_choice_token_ids(tokenizer):
    a_ids = tokenizer.encode("A", add_special_tokens=False)
    b_ids = tokenizer.encode("B", add_special_tokens=False)
    c_ids = tokenizer.encode("C", add_special_tokens=False)
    d_ids = tokenizer.encode("D", add_special_tokens=False)
    
    space_a_ids = tokenizer.encode(" A", add_special_tokens=False)
    space_b_ids = tokenizer.encode(" B", add_special_tokens=False)
    space_c_ids = tokenizer.encode(" C", add_special_tokens=False)
    space_d_ids = tokenizer.encode(" D", add_special_tokens=False)
    
    choices_map = {}
    for idx, (raw, space) in enumerate([
        (a_ids, space_a_ids),
        (b_ids, space_b_ids),
        (c_ids, space_c_ids),
        (d_ids, space_d_ids)
    ]):
        toks = []
        if len(raw) > 0:
            toks.append(raw[0])
        if len(space) > 0:
            toks.append(space[0])
        choices_map[idx] = list(set(toks))
    return choices_map

def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemma 4 Softmax or SBA models on MMLU")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-4-E4B", help="Base model path or name")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="PEFT SBA adapter checkpoint directory (optional)")
    parser.add_argument("--batch_size", type=str, default="16", help="Batch size for evaluation")
    parser.add_argument("--num_shots", type=int, default=5, help="Number of shots for MMLU few-shot evaluation")
    parser.add_argument("--subjects", type=str, default="all", help="Comma-separated list of subjects or 'all'")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--limit_questions", type=int, default=None, help="Limit number of questions per subject for quick testing")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load the base model in 4-bit precision for low VRAM environments")
    args = parser.parse_args()

    batch_size = int(args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve subjects
    if args.subjects.lower() == "all":
        subjects_to_eval = MMLU_SUBJECTS
    else:
        subjects_to_eval = [s.strip() for s in args.subjects.split(",") if s.strip()]

    print(f"Loading tokenizer and base model for: {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    # Load model
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        print("Loading base model in 4-bit precision...")
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": "cuda:0"}
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs
    )

    is_sba = False
    if args.checkpoint_dir:
        print(f"Patching model with SBA attention...")
        # First monkey-patch base model text attention layers
        patch_gemma4_sba(model)
        
        print(f"Loading PEFT adapter weights from {args.checkpoint_dir}...")
        if hasattr(model, "_no_split_modules") and isinstance(model._no_split_modules, set):
            model._no_split_modules = list(model._no_split_modules)
        model = PeftModel.from_pretrained(model, args.checkpoint_dir, adapter_name="eval_adapter")
        
        bridge_state_path = os.path.join(args.checkpoint_dir, "sba_bridge_v3.pt")
        if os.path.exists(bridge_state_path):
            print(f"Loading SBA bridge parameters from {bridge_state_path}...")
            load_sba_bridge_state(model, bridge_state_path)
        else:
            print(f"WARNING: SBA bridge state file not found at {bridge_state_path}")
        
        is_sba = True
    else:
        print("Evaluating original Softmax base model.")

    if not args.load_in_4bit:
        model = model.to(device)
    model.eval()

    # Get token IDs for choices A, B, C, D
    choices_map = get_choice_token_ids(tokenizer)
    print("Mapped choice tokens:")
    for choice_idx, label in enumerate(["A", "B", "C", "D"]):
        print(f"  {label} -> token IDs: {choices_map[choice_idx]}")

    results = {}
    total_correct = 0
    total_questions = 0

    for subject in tqdm(subjects_to_eval, desc="Subjects"):
        try:
            # Try CAIS MMLU dataset
            dev_ds = load_dataset("cais/mmlu", subject, split="dev")
            test_ds = load_dataset("cais/mmlu", subject, split="test")
        except Exception as e:
            print(f"\nFailed to load cais/mmlu for {subject}. Error: {e}. Trying fallback dataset...")
            try:
                # Try fallback: Rukaemon/MMLU or similar if available, or print and skip
                dev_ds = load_dataset("lukaemon/mmlu", subject, split="validation")
                test_ds = load_dataset("lukaemon/mmlu", subject, split="test")
            except Exception as e2:
                print(f"Failed to load fallback for {subject}. Error: {e2}. Skipping subject.")
                continue

        subject_correct = 0
        subject_questions = len(test_ds)
        if args.limit_questions is not None:
            subject_questions = min(subject_questions, args.limit_questions)

        prompts = []
        labels = []
        for idx in range(subject_questions):
            test_example = test_ds[idx]
            prompt = gen_prompt(dev_ds, subject, test_example, k=args.num_shots)
            prompts.append(prompt)
            labels.append(test_example["answer"])

        predictions = []
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)
                logits = outputs.logits[:, -1, :] # last token logits
                probs = F.softmax(logits, dim=-1)
                
                batch_probs = []
                for choice_idx in range(4):
                    token_ids = choices_map[choice_idx]
                    # Sum probabilities of space-prefixed and non-space-prefixed tokens
                    prob_c = probs[:, token_ids].sum(dim=-1)
                    batch_probs.append(prob_c)
                
                # Stack to shape [batch_size, 4]
                batch_probs = torch.stack(batch_probs, dim=-1)
                batch_preds = batch_probs.argmax(dim=-1).cpu().tolist()
                predictions.extend(batch_preds)
                
            # Free VRAM
            del inputs, outputs, logits, probs, batch_probs
            
        torch.cuda.empty_cache()

        for pred, label in zip(predictions, labels):
            if pred == label:
                subject_correct += 1

        accuracy = subject_correct / subject_questions if subject_questions > 0 else 0.0
        results[subject] = {
            "accuracy": accuracy,
            "correct": subject_correct,
            "total": subject_questions
        }
        total_correct += subject_correct
        total_questions += subject_questions

    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
    print(f"\n========================================")
    print(f"MMLU Evaluation Results:")
    print(f"  Model: {args.model_name_or_path}")
    if args.checkpoint_dir:
        print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Overall Accuracy: {overall_accuracy * 100:.2f}% ({total_correct}/{total_questions})")
    print(f"========================================")

    output_data = {
        "model_name_or_path": args.model_name_or_path,
        "checkpoint_dir": args.checkpoint_dir,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "subject_results": results
    }

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
