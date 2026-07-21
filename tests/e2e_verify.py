#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SBA Self-Distill Compiler - E2E Verification Suite
This script programmatically verifies that the compiled self-distillation dataset
and the LLM judge audit report meet all target quality, quantity, and formatting criteria.
"""

import os
import sys
import json
import math
import argparse
import warnings
# Suppress all tokenizer-related FutureWarning and HF Hub unauthenticated warnings
warnings.simplefilter("ignore")
from collections import Counter
from transformers import AutoTokenizer

def extract_completion(record):
    """
    Extracts the completion string from a dataset record.
    Supports multiple formats:
    1. Direct 'completion' key
    2. 'prompt' and 'text' keys (completion = text[len(prompt):])
    3. 'text' only, split by standard delimiters (e.g. ### Response:)
    4. Fallback to entire text if no split can be found.
    """
    if "completion" in record:
        return record["completion"]
    
    if "prompt" in record and "text" in record:
        prompt = record["prompt"]
        text = record["text"]
        if text.startswith(prompt):
            return text[len(prompt):]
            
    if "text" in record:
        text = record["text"]
        # Try common separators in order of specificity
        for sep in ["### Response:\n", "\n### Response:\n", "Response:", "Answer:", "\nAnswer:"]:
            if sep in text:
                parts = text.split(sep, 1)
                return parts[1]
        
        # If it's plain prose, the entire text may represent the completion or continuation
        if record.get("category") == "plain_prose":
            return text
            
        return text
        
    return ""

def calculate_entropy(token_ids_list):
    """
    Calculates the vocabulary entropy (base 2) of a tokenized corpus.
    """
    # Flatten the token list of lists for fast counting
    flat_ids = [tid for ids in token_ids_list for tid in ids]
    counter = Counter(flat_ids)
        
    total_tokens = sum(counter.values())
    if total_tokens == 0:
        return 0.0
        
    entropy = 0.0
    for count in counter.values():
        p = count / total_tokens
        entropy -= p * math.log2(p)
    return entropy

def analyze_layout(text, category):
    """
    Analyzes the layout characteristics of a text sequence.
    Returns a dict of detected features (e.g. formatting style, shot count).
    """
    features = {
        "is_bulleted": False,
        "is_numbered": False,
        "shot_count": 0
    }
    
    if not text:
        return features
        
    # Check for bullets
    if "\n- " in text or "\n* " in text or "\n• " in text:
        features["is_bulleted"] = True
        
    # Check for numbering
    if "\n1. " in text or "\n2. " in text or "\n1) " in text:
        features["is_numbered"] = True
        
    # Estimate shot count for few-shot QA
    if category == "few_shot_qa":
        # Look for typical QA separators to count shots
        q_indicators = ["Question:", "Q:", "Q :", "Question ", "Input:"]
        best_count = 0
        for indicator in q_indicators:
            count = text.count(indicator)
            if count > best_count:
                best_count = count
        # In a few-shot prompt, the final QA is part of the prompt,
        # so shots count is usually total QA blocks minus 1
        features["shot_count"] = max(0, best_count - 1)
        
    return features

def main():
    parser = argparse.ArgumentParser(description="E2E Verification for SBA Self-Distill Compiler")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=r"D:\teamwork_projects\sba_self_distill_compiler\sba_self_distill.jsonl",
        help="Path to the compiled sba_self_distill.jsonl file"
    )
    parser.add_argument(
        "--report", 
        type=str, 
        default=r"D:\teamwork_projects\sba_self_distill_compiler\dataset_audit_report.json",
        help="Path to the dataset_audit_report.json file"
    )
    parser.add_argument(
        "--tokenizer", 
        type=str, 
        default="google/gemma-4-E4B",
        help="HF Tokenizer name or local path"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SBA SELF-DISTILL COMPILER - RUNNING E2E VERIFICATION")
    print("=" * 70)
    
    # Check 1: Verify file existence
    print("Check 1: Verifying file existence...")
    files_exist = True
    if not os.path.exists(args.dataset):
        print(f"[-] ERROR: Dataset file not found at: {args.dataset}")
        files_exist = False
    else:
        print(f"[+] Dataset file exists at: {args.dataset}")
        
    if not os.path.exists(args.report):
        print(f"[-] ERROR: Audit report file not found at: {args.report}")
        files_exist = False
    else:
        print(f"[+] Audit report file exists at: {args.report}")
        
    if not files_exist:
        print("[-] E2E Verification failed: Required files are missing.")
        sys.exit(1)
        
    # Check 2: Load Tokenizer
    print("\nCheck 2: Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"[+] Tokenizer '{args.tokenizer}' loaded successfully (vocab size: {len(tokenizer)}).")
    except Exception as e:
        print(f"[-] ERROR: Failed to load tokenizer '{args.tokenizer}': {e}")
        sys.exit(2)
        
    # Check 3: Parse Dataset and verify constraints
    print("\nCheck 3: Parsing dataset records...")
    records = []
    unique_texts = set()
    duplicate_count = 0
    
    category_counts = Counter()
    few_shot_qa_shots = []
    layout_types = Counter()
    
    try:
        with open(args.dataset, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as jde:
                    print(f"[-] ERROR: Invalid JSON on line {idx}: {jde}")
                    sys.exit(3)
                
                # Check for 'text' field
                if "text" not in record:
                    print(f"[-] ERROR: Record on line {idx} is missing the required 'text' field.")
                    sys.exit(3)
                    
                text_content = record["text"]
                if text_content in unique_texts:
                    duplicate_count += 1
                unique_texts.add(text_content)
                
                # Retrieve category
                category = record.get("category", "unknown")
                category_counts[category] += 1
                
                # Layout analysis
                layout_info = analyze_layout(text_content, category)
                if category == "few_shot_qa":
                    few_shot_qa_shots.append(layout_info["shot_count"])
                    
                if layout_info["is_bulleted"]:
                    layout_types["bulleted"] += 1
                if layout_info["is_numbered"]:
                    layout_types["numbered"] += 1
                if not layout_info["is_bulleted"] and not layout_info["is_numbered"]:
                    layout_types["standard_or_inline"] += 1
                    
                records.append(record)
    except Exception as e:
        print(f"[-] ERROR: Reading dataset file failed: {e}")
        sys.exit(4)
        
    total_records = len(records)
    print(f"[+] Loaded {total_records} records from dataset.")
    print(f"[+] Duplicate text count: {duplicate_count}")
    
    # Batch tokenization for performance
    print("Tokenizing completions and text in batches...")
    total_completion_tokens = 0
    all_token_ids = []
    
    batch_size = 5000
    for i in range(0, total_records, batch_size):
        batch_records = records[i:i+batch_size]
        batch_completions = [extract_completion(r) for r in batch_records]
        batch_texts = [r["text"] for r in batch_records]
        
        # Tokenize completions
        comp_encodings = tokenizer(batch_completions, add_special_tokens=False, padding=False, truncation=False)
        total_completion_tokens += sum(len(ids) for ids in comp_encodings["input_ids"])
        
        # Tokenize full texts
        text_encodings = tokenizer(batch_texts, add_special_tokens=False, padding=False, truncation=False)
        all_token_ids.extend(text_encodings["input_ids"])
    
    # Calculate proportions
    proportions = {}
    expected_proportions = {
        "plain_prose": 0.15,
        "few_shot_qa": 0.15,
        "mathematical_reasoning": 0.30,
        "general_instruction_code": 0.40
    }
    
    print("\nCategory Distribution:")
    for cat, exp in expected_proportions.items():
        count = category_counts[cat]
        prop = count / total_records if total_records > 0 else 0.0
        proportions[cat] = prop
        print(f"  - {cat}: {count} ({prop:.2%}) [Expected >= {exp:.0%}]")
        
    # Calculate average completion length
    avg_completion_length = total_completion_tokens / total_records if total_records > 0 else 0.0
    print(f"[+] Average completion length: {avg_completion_length:.2f} tokens (Expected >= 100)")
    
    # Calculate vocabulary entropy
    print("Calculating vocabulary entropy (this might take a few moments)...")
    vocab_entropy = calculate_entropy(all_token_ids)
    print(f"[+] Vocabulary entropy: {vocab_entropy:.4f} (Expected >= 6.5)")
    
    # Check 4: Parse Audit Report
    print("\nCheck 4: Parsing audit report...")
    try:
        with open(args.report, "r", encoding="utf-8") as r_file:
            audit_data = json.load(r_file)
    except Exception as e:
        print(f"[-] ERROR: Failed to parse audit report JSON: {e}")
        sys.exit(5)
        
    if not isinstance(audit_data, dict):
        print("[-] ERROR: Audit report root is not a JSON object (dictionary).")
        sys.exit(5)
        
    # Resolve LLM judge statistics
    # Look for llm_judge section or check top-level
    llm_judge_data = None
    for key in ["llm_judge", "llm_judge_evaluation", "llm_judge_results", "judge", "llm_judge_stats"]:
        if key in audit_data:
            llm_judge_data = audit_data[key]
            break
    if llm_judge_data is None:
        llm_judge_data = audit_data
        
    if not isinstance(llm_judge_data, dict):
        print("[-] ERROR: LLM judge data section is not a JSON object (dictionary).")
        sys.exit(6)
        
    total_evaluated = None
    for key in ["evaluated_samples", "total_evaluated", "samples_evaluated", "total_samples", "evaluated", "total"]:
        if key in llm_judge_data:
            total_evaluated = llm_judge_data[key]
            break
            
    passed_samples = None
    for key in ["passed_samples", "passed", "pass_count", "passed_count", "successes", "success_count"]:
        if key in llm_judge_data:
            passed_samples = llm_judge_data[key]
            break
            
    if total_evaluated is None or passed_samples is None:
        print("[-] ERROR: Could not find LLM judge evaluation details in the audit report.")
        print(f"Audit report content: {json.dumps(audit_data, indent=2)[:500]}...")
        sys.exit(6)
        
    try:
        total_evaluated = int(total_evaluated)
    except (ValueError, TypeError):
        print(f"[-] ERROR: total_evaluated could not be parsed as an integer (Actual: {total_evaluated})")
        sys.exit(6)

    try:
        passed_samples = int(passed_samples)
    except (ValueError, TypeError):
        print(f"[-] ERROR: passed_samples could not be parsed as an integer (Actual: {passed_samples})")
        sys.exit(6)
        
    print(f"[+] LLM Judge evaluation: {passed_samples}/{total_evaluated} passed (Expected: evaluated = 100, passed >= 95)")
    
    # Check 5: Evaluate all assertions
    print("\nCheck 5: Evaluating all acceptance criteria...")
    
    checks = {
        "dataset_exists": (True, "Dataset file exists"),
        "report_exists": (True, "Audit report file exists"),
        "min_records": (total_records >= 50000, f"Dataset contains >= 50,000 records (Actual: {total_records})"),
        "no_duplicates": (duplicate_count == 0, f"Duplicate text count is 0 (Actual: {duplicate_count})"),
        "plain_prose_proportion": (proportions.get("plain_prose", 0.0) >= 0.15, f"plain_prose proportion >= 15% (Actual: {proportions.get('plain_prose', 0.0):.2%})"),
        "few_shot_qa_proportion": (proportions.get("few_shot_qa", 0.0) >= 0.15, f"few_shot_qa proportion >= 15% (Actual: {proportions.get('few_shot_qa', 0.0):.2%})"),
        "mathematical_reasoning_proportion": (proportions.get("mathematical_reasoning", 0.0) >= 0.30, f"mathematical_reasoning proportion >= 30% (Actual: {proportions.get('mathematical_reasoning', 0.0):.2%})"),
        "general_instruction_code_proportion": (proportions.get("general_instruction_code", 0.0) >= 0.40, f"general_instruction_code proportion >= 40% (Actual: {proportions.get('general_instruction_code', 0.0):.2%})"),
        "min_avg_completion_len": (avg_completion_length >= 100.0, f"Average completion length >= 100 tokens (Actual: {avg_completion_length:.2f})"),
        "min_vocab_entropy": (vocab_entropy >= 6.5, f"Vocabulary entropy >= 6.5 (Actual: {vocab_entropy:.4f})"),
        "llm_judge_evaluated": (total_evaluated == 100, f"LLM judge evaluated 100 samples (Actual: {total_evaluated})"),
        "llm_judge_passed": (passed_samples >= 95, f"LLM judge passed >= 95 samples (Actual: {passed_samples})")
    }
    
    # Extra validation: Layout diversity stats (warning-only if not hard-constrained by prompt but required by TEST_INFRA)
    print("\nLayout Diversity Statistics:")
    print(f"  - Bulleted layout count: {layout_types['bulleted']}")
    print(f"  - Numbered layout count: {layout_types['numbered']}")
    print(f"  - Standard/inline layout count: {layout_types['standard_or_inline']}")
    if few_shot_qa_shots:
        min_shots = min(few_shot_qa_shots)
        max_shots = max(few_shot_qa_shots)
        avg_shots = sum(few_shot_qa_shots) / len(few_shot_qa_shots)
        print(f"  - Few-shot QA shots: min={min_shots}, max={max_shots}, avg={avg_shots:.2f}")
        # Programmatic check for shots variation between 3 and 5
        # If the dataset has QA, let's verify if shots are indeed in that range
        checks["few_shot_qa_shots_range"] = (
            min_shots >= 3 and max_shots <= 5,
            f"Few-shot QA shots vary between 3 and 5 (Actual range: [{min_shots}, {max_shots}])"
        )
    else:
        print("  - No few-shot QA examples parsed, layout stats skipped.")
        checks["few_shot_qa_shots_range"] = (False, "No few_shot_qa records found in dataset to verify shots range")
        
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for check_id, (passed, desc) in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status:<8} {desc}")
        if not passed:
            all_passed = False
            
    print("=" * 70)
    if all_passed:
        print("[+] SUCCESS: All programmatic verification checks passed.")
        sys.exit(0)
    else:
        print("[-] FAILURE: One or more programmatic verification checks failed.")
        sys.exit(7)

if __name__ == "__main__":
    main()
