import argparse
import json
import os
import random
import sys
from pathlib import Path
from datasets import load_dataset

# Layout functions for few-shot multiple choice to prevent template degradation
def format_qa_layout_a(question: str, choices: list[str], answer: str, include_answer: bool = True) -> str:
    prompt = f"Question: {question.strip()}\n"
    labels = ["A", "B", "C", "D"]
    for i, choice in enumerate(choices[:4]):
        prompt += f"{labels[i]}. {choice.strip()}\n"
    if include_answer:
        prompt += f"Answer: {answer.strip()}\n\n"
    else:
        prompt += "Answer:"
    return prompt

def format_qa_layout_b(question: str, choices: list[str], answer: str, include_answer: bool = True) -> str:
    prompt = f"Q: {question.strip()}\n"
    labels = ["A", "B", "C", "D"]
    choices_str = " ".join(f"({labels[i]}) {choice.strip()}" for i, choice in enumerate(choices[:4]))
    prompt += f"Choices: {choices_str}\n"
    if include_answer:
        prompt += f"Answer: {answer.strip()}\n\n"
    else:
        prompt += "Answer:"
    return prompt

def format_qa_layout_c(question: str, choices: list[str], answer: str, include_answer: bool = True) -> str:
    prompt = f"Question: {question.strip()}\n"
    labels = ["1", "2", "3", "4"]
    for i, choice in enumerate(choices[:4]):
        prompt += f"  {labels[i]}) {choice.strip()}\n"
    ans_map = {"A": "1", "B": "2", "C": "3", "D": "4"}
    mapped_answer = ans_map.get(answer.strip(), answer.strip())
    if include_answer:
        prompt += f"Correct Option: {mapped_answer}\n\n"
    else:
        prompt += "Correct Option:"
    return prompt

def format_qa_layout_d(question: str, choices: list[str], answer: str, include_answer: bool = True) -> str:
    prompt = f"Select the best option for: {question.strip()}\n"
    labels = ["A", "B", "C", "D"]
    for i, choice in enumerate(choices[:4]):
        prompt += f"- {labels[i]}: {choice.strip()}\n"
    if include_answer:
        prompt += f"Correct Answer: {answer.strip()}\n\n"
    else:
        prompt += "Correct Answer:"
    return prompt

# Layout variations for other categories to prevent template degradation
def format_prose_prompt(prompt_text: str, rng: random.Random) -> str:
    templates = [
        f"User: {prompt_text.strip()}\nAssistant:",
        f"Task: {prompt_text.strip()}\nResponse:",
        f"Provide a detailed response for: {prompt_text.strip()}\n\nAnswer:",
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt_text.strip()}\n\n### Response:"
    ]
    return rng.choice(templates)

def format_math_prompt(prompt_text: str, rng: random.Random) -> str:
    templates = [
        f"Solve the following math problem step-by-step:\n{prompt_text.strip()}\n\nAnswer:",
        f"Problem: {prompt_text.strip()}\nReason step-by-step and provide the final solution.",
        f"Below is a math query. Write a step-by-step response that solves the problem.\n\n### Instruction:\n{prompt_text.strip()}\n\n### Response:",
        f"Provide a step-by-step mathematical reasoning to solve this problem: {prompt_text.strip()}\nSolution:"
    ]
    return rng.choice(templates)

def format_code_prompt(prompt_text: str, rng: random.Random) -> str:
    templates = [
        f"User: {prompt_text.strip()}\nAssistant:",
        f"Write a programming solution for: {prompt_text.strip()}\nCode:",
        f"Below is a programming task. Write a solution that solves the task.\n\n### Instruction:\n{prompt_text.strip()}\n\n### Response:",
        f"Task: {prompt_text.strip()}\n\nProvide the implementation:"
    ]
    return rng.choice(templates)

def make_few_shot_qa_prompt(
    subject: str,
    exemplars: list[dict],
    target: dict,
    num_shots: int,
    rng: random.Random
) -> str:
    instructions = [
        f"The following are multiple-choice questions (with answers) about {subject.replace('_', ' ')}.",
        f"Answer these multiple choice questions regarding {subject.replace('_', ' ')}.",
        f"Review the questions below about {subject.replace('_', ' ')} and provide the correct option.",
        f"Test your knowledge on {subject.replace('_', ' ')}. Select the best choice for each query.",
        ""
    ]
    selected_instruction = rng.choice(instructions)
    layout_funcs = [
        format_qa_layout_a,
        format_qa_layout_b,
        format_qa_layout_c,
        format_qa_layout_d
    ]
    selected_layout = rng.choice(layout_funcs)
    
    prompt = ""
    if selected_instruction:
        prompt += selected_instruction + "\n\n"
        
    sampled_exemplars = rng.sample(exemplars, min(num_shots, len(exemplars))) if exemplars else []
    for ex in sampled_exemplars:
        prompt += selected_layout(
            question=ex["question"],
            choices=ex["choices"],
            answer=ex["answer"],
            include_answer=True
        )
    prompt += selected_layout(
        question=target["question"],
        choices=target["choices"],
        answer=target["answer"],
        include_answer=False
    )
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Curate 50,000 prompts across four categories for self-distillation")
    parser.add_argument("--output", default="data/sba_curated_prompts_50k.jsonl", help="Output JSONL file path")
    parser.add_argument("--seed", type=int, default=20260528, help="Random seed for replication")
    parser.add_argument("--prose-target", type=int, default=7500, help="Target count for plain prose prompts")
    parser.add_argument("--qa-target", type=int, default=7500, help="Target count for few-shot QA prompts")
    parser.add_argument("--math-target", type=int, default=15000, help="Target count for math reasoning prompts")
    parser.add_argument("--code-target", type=int, default=20000, help="Target count for general instruction/code prompts")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    compiled_items = []
    
    # -------------------------------------------------------------------------
    # 1. Plain Prose Curation (Target: >=15%, e.g., 7500)
    # -------------------------------------------------------------------------
    print(f"Curating prose prompts (target: {args.prose_target})...")
    prose_count = 0
    
    # Sourced from databricks/databricks-dolly-15k (creative_writing, summarization, open_qa)
    try:
        dolly_ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        filtered_dolly = [
            row for row in dolly_ds 
            if row.get("category") in {"creative_writing", "summarization", "open_qa", "brainstorming"}
        ]
        rng.shuffle(filtered_dolly)
        for row in filtered_dolly:
            if prose_count >= args.prose_target:
                break
            raw_prompt = row.get("instruction", "").strip()
            if row.get("context", "").strip():
                raw_prompt += f"\nContext:\n{row.get('context').strip()}"
            if len(raw_prompt) < 30:
                continue
            formatted = format_prose_prompt(raw_prompt, rng)
            compiled_items.append({
                "category": "plain_prose",
                "source": "databricks/databricks-dolly-15k",
                "original_prompt": raw_prompt,
                "formatted_prompt": formatted,
                "metadata": {"dolly_category": row.get("category")}
            })
            prose_count += 1
    except Exception as e:
        print(f"Warning: Failed to load Databricks Dolly: {e}")

    # Fallback/supplement with tatsu-lab/alpaca (prose instruction filter)
    if prose_count < args.prose_target:
        try:
            alpaca_ds = load_dataset("tatsu-lab/alpaca", split="train")
            for row in alpaca_ds:
                if prose_count >= args.prose_target:
                    break
                inst = row.get("instruction", "").strip()
                inp = row.get("input", "").strip()
                # Exclude mathematical or coding instructions
                lowered = (inst + " " + inp).lower()
                is_math_or_code = any(word in lowered for word in ["python", "code", "equation", "solve", "math", "integer", "matrix", "function"])
                if is_math_or_code:
                    continue
                raw_prompt = inst
                if inp:
                    raw_prompt += f"\nInput:\n{inp}"
                if len(raw_prompt) < 30:
                    continue
                formatted = format_prose_prompt(raw_prompt, rng)
                compiled_items.append({
                    "category": "plain_prose",
                    "source": "tatsu-lab/alpaca",
                    "original_prompt": raw_prompt,
                    "formatted_prompt": formatted,
                    "metadata": {"alpaca_source": "alpaca_train"}
                })
                prose_count += 1
        except Exception as e:
            print(f"Warning: Failed to load Alpaca for prose: {e}")

    print(f"Prose curated: {prose_count}/{args.prose_target}")

    # -------------------------------------------------------------------------
    # 2. Few-shot Multiple Choice QA Curation (Target: >=15%, e.g., 7500)
    # -------------------------------------------------------------------------
    print(f"Curating few-shot QA prompts (target: {args.qa_target})...")
    qa_count = 0
    
    # Sourced from cais/mmlu (all 57 subjects)
    mmlu_subjects = [
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
    
    # Collect subjects and their exemplars and test questions
    mmlu_pools = []
    for subject in mmlu_subjects:
        try:
            dev_ds = load_dataset("cais/mmlu", subject, split="dev")
            test_ds = load_dataset("cais/mmlu", subject, split="test")
            
            # Exemplars pool
            exemplars = []
            for row in dev_ds:
                ans_letter = ["A", "B", "C", "D"][int(row.get("answer"))]
                exemplars.append({
                    "question": row.get("question"),
                    "choices": row.get("choices"),
                    "answer": ans_letter
                })
            
            # Convert test_ds to a list of dicts/rows
            test_rows = list(test_ds)
            mmlu_pools.append({
                "subject": subject,
                "exemplars": exemplars,
                "rows": test_rows,
                "index": 0
            })
        except Exception as e:
            print(f"Warning: Failed to load MMLU subject {subject}: {e}")
            
    # Draw from pools round-robin until qa_count reaches args.qa_target
    if mmlu_pools:
        pool_idx = 0
        while qa_count < args.qa_target:
            # check if all pools are exhausted
            all_exhausted = True
            for p in mmlu_pools:
                if p["index"] < len(p["rows"]):
                    all_exhausted = False
                    break
            if all_exhausted:
                print("Warning: All MMLU questions exhausted!")
                break
                
            pool = mmlu_pools[pool_idx]
            if pool["index"] < len(pool["rows"]):
                row = pool["rows"][pool["index"]]
                pool["index"] += 1
                
                ans_letter = ["A", "B", "C", "D"][int(row.get("answer"))]
                target_q = {
                    "question": row.get("question"),
                    "choices": row.get("choices"),
                    "answer": ans_letter
                }
                
                num_shots = rng.randint(3, 5)
                formatted = make_few_shot_qa_prompt(
                    subject=pool["subject"],
                    exemplars=pool["exemplars"],
                    target=target_q,
                    num_shots=num_shots,
                    rng=rng
                )
                
                compiled_items.append({
                    "category": "few_shot_qa",
                    "source": f"cais/mmlu/{pool['subject']}",
                    "original_prompt": target_q["question"],
                    "formatted_prompt": formatted,
                    "metadata": {
                        "subject": pool["subject"],
                        "num_shots": num_shots,
                        "correct_answer": ans_letter
                    }
                })
                qa_count += 1
                
            pool_idx = (pool_idx + 1) % len(mmlu_pools)
            
    print(f"Few-shot QA curated: {qa_count}/{args.qa_target}")

    # -------------------------------------------------------------------------
    # 3. Mathematical Reasoning Curation (Target: >=30%, e.g., 15000)
    # -------------------------------------------------------------------------
    print(f"Curating math reasoning prompts (target: {args.math_target})...")
    math_count = 0
    
    # Sourced from AI-MO/NuminaMath-CoT (streaming to avoid loading 860k dataset)
    try:
        numina_ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
        for row in numina_ds:
            if math_count >= args.math_target:
                break
            problem = row.get("problem", "").strip()
            if len(problem) < 20:
                continue
            formatted = format_math_prompt(problem, rng)
            compiled_items.append({
                "category": "mathematical_reasoning",
                "source": "AI-MO/NuminaMath-CoT",
                "original_prompt": problem,
                "formatted_prompt": formatted,
                "metadata": {"numina_source": row.get("source", "unknown")}
            })
            math_count += 1
    except Exception as e:
        print(f"Warning: Failed to load NuminaMath: {e}")
        
    # Sourced from gsm8k
    if math_count < args.math_target:
        try:
            gsm_ds = load_dataset("gsm8k", "main", split="train")
            for row in gsm_ds:
                if math_count >= args.math_target:
                    break
                q = row.get("question", "").strip()
                if len(q) < 20:
                    continue
                formatted = format_math_prompt(q, rng)
                compiled_items.append({
                    "category": "mathematical_reasoning",
                    "source": "gsm8k",
                    "original_prompt": q,
                    "formatted_prompt": formatted,
                    "metadata": {"gsm_split": "train"}
                })
                math_count += 1
        except Exception as e:
            print(f"Warning: Failed to load GSM8K: {e}")

    print(f"Math reasoning curated: {math_count}/{args.math_target}")

    # -------------------------------------------------------------------------
    # 4. General Instruction/Code Curation (Target: >=40%, e.g., 20000)
    # -------------------------------------------------------------------------
    print(f"Curating code prompts (target: {args.code_target})...")
    code_count = 0
    
    # Sourced from nvidia/OpenCodeReasoning
    try:
        nvidia_ds = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True)
        for row in nvidia_ds:
            if code_count >= args.code_target:
                break
            inp = row.get("input", "").strip()
            if len(inp) < 30:
                continue
            formatted = format_code_prompt(inp, rng)
            compiled_items.append({
                "category": "general_instruction_code",
                "source": "nvidia/OpenCodeReasoning",
                "original_prompt": inp,
                "formatted_prompt": formatted,
                "metadata": {
                    "difficulty": row.get("difficulty", "unknown"),
                    "row_source": row.get("source", "unknown")
                }
            })
            code_count += 1
    except Exception as e:
        print(f"Warning: Failed to load OpenCodeReasoning: {e}")

    # Sourced from ise-uiuc/Magicoder-OSS-Instruct-75K
    if code_count < args.code_target:
        try:
            magicoder_ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True)
            for row in magicoder_ds:
                if code_count >= args.code_target:
                    break
                prob = row.get("problem", "").strip()
                if len(prob) < 30:
                    continue
                formatted = format_code_prompt(prob, rng)
                compiled_items.append({
                    "category": "general_instruction_code",
                    "source": "ise-uiuc/Magicoder-OSS-Instruct-75K",
                    "original_prompt": prob,
                    "formatted_prompt": formatted,
                    "metadata": {"magicoder_type": "oss_instruct"}
                })
                code_count += 1
        except Exception as e:
            print(f"Warning: Failed to load Magicoder: {e}")

    # Sourced from mbpp
    if code_count < args.code_target:
        try:
            mbpp_ds = load_dataset("mbpp", "sanitized", split="train")
            for row in mbpp_ds:
                if code_count >= args.code_target:
                    break
                text = row.get("text", "").strip()
                if len(text) < 15:
                    continue
                formatted = format_code_prompt(text, rng)
                compiled_items.append({
                    "category": "general_instruction_code",
                    "source": "mbpp",
                    "original_prompt": text,
                    "formatted_prompt": formatted,
                    "metadata": {"mbpp_id": row.get("task_id")}
                })
                code_count += 1
        except Exception as e:
            print(f"Warning: Failed to load MBPP: {e}")

    print(f"Code curated: {code_count}/{args.code_target}")

    # -------------------------------------------------------------------------
    # 5. Saving and Validation
    # -------------------------------------------------------------------------
    total_curated = len(compiled_items)
    print(f"\nTotal curated examples: {total_curated}")
    
    # Calculate category percentages
    category_counts = {}
    for item in compiled_items:
        cat = item["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
    print("Category Breakdown:")
    for cat, count in category_counts.items():
        percentage = (count / total_curated) * 100
        print(f"  {cat}: {count} ({percentage:.2f}%)")
        
    # Write to JSONL
    print(f"Saving compiled prompts to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in compiled_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Compilation complete and verified!")

if __name__ == "__main__":
    main()
