#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validation script for tests/e2e_verify.py.
This script:
1. Generates a temporary valid dataset of 50,000 records with appropriate categories, lengths, and vocabulary diversity.
2. Generates a temporary valid audit report.
3. Runs the e2e_verify.py script pointing to these temporary files.
4. Confirms that the verification script exits with 0 (success).
5. Cleans up the temporary files.
"""

import sys
import json
import random
import string
import subprocess
import uuid
from pathlib import Path

def generate_test_data(dataset_path, report_path):
    print("Generating temporary test dataset (50,000 records) with high vocabulary entropy...")
    
    random.seed(42)
    
    # 50k records split:
    # plain_prose: 15% = 7500
    # few_shot_qa: 15% = 7500
    # mathematical_reasoning: 30% = 15000
    # general_instruction_code: 40% = 20000
    
    categories = []
    categories.extend(["plain_prose"] * 7500)
    categories.extend(["few_shot_qa"] * 7500)
    categories.extend(["mathematical_reasoning"] * 15000)
    categories.extend(["general_instruction_code"] * 20000)
    
    with open(dataset_path, "w", encoding="utf-8") as f:
        for idx, cat in enumerate(categories):
            # Generate random unique completion of 120 words using random character strings
            # to guarantee extremely high vocabulary entropy
            completion_words = []
            for _ in range(120):
                word_len = random.randint(4, 9)
                word = "".join(random.choices(string.ascii_lowercase, k=word_len))
                completion_words.append(word)
            
            completion_words.append(f"unique_id_{idx}")
            completion_text = " ".join(completion_words)
            
            if cat == "few_shot_qa":
                # Create a few-shot QA prompt with 3 to 5 shots
                shots = random.choice([3, 4, 5])
                prompt_parts = []
                for s in range(shots):
                    q_word = "".join(random.choices(string.ascii_lowercase, k=6))
                    a_word = "".join(random.choices(string.ascii_lowercase, k=6))
                    prompt_parts.append(f"Question: What is {q_word}?\nAnswer: {a_word}")
                prompt_parts.append("Question: Final question?\n### Response:\n")
                prompt_text = "\n".join(prompt_parts)
            elif cat == "plain_prose":
                p_word = "".join(random.choices(string.ascii_lowercase, k=6))
                prompt_text = f"Continuous prose about {p_word}: "
            elif cat == "mathematical_reasoning":
                p_word = "".join(random.choices(string.ascii_lowercase, k=6))
                prompt_text = f"Solve the mathematical equation of {p_word}: ### Response:\n"
            else:  # general_instruction_code
                p_word = "".join(random.choices(string.ascii_lowercase, k=6))
                prompt_text = f"Write an instruction to solve {p_word}: ### Response:\n"
                
            full_text = prompt_text + completion_text
            
            record = {
                "text": full_text,
                "category": cat,
                "prompt": prompt_text,
                "completion": completion_text
            }
            f.write(json.dumps(record) + "\n")
            
    print(f"Dataset generated with {len(categories)} records at: {dataset_path}")
    
    # Generate temporary audit report
    print("Generating temporary audit report...")
    report_data = {
        "dataset_statistics": {
            "total_records": len(categories),
            "duplicates": 0
        },
        "llm_judge": {
            "evaluated_samples": 100,
            "passed_samples": 98
        }
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Audit report generated at: {report_path}")

def main():
    project_root = Path(__file__).resolve().parent.parent
    unique_id = uuid.uuid4().hex
    temp_dataset = project_root / "tests" / f"temp_sba_self_distill_{unique_id}.jsonl"
    temp_report = project_root / "tests" / f"temp_dataset_audit_report_{unique_id}.json"
    verify_script = project_root / "tests" / "e2e_verify.py"
    
    try:
        generate_test_data(temp_dataset, temp_report)
        
        print("\nRunning verification script on temporary files...")
        cmd = [
            sys.executable,
            str(verify_script),
            "--dataset", str(temp_dataset),
            "--report", str(temp_report)
        ]
        
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("\n=== Verification Script Output ===")
        print(result.stdout)
        if result.stderr:
            print("=== Verification Script Error Output ===")
            print(result.stderr)
            
        print(f"Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("\n[+] SUCCESS: e2e_verify.py passed validation!")
        else:
            print("\n[-] FAILURE: e2e_verify.py failed validation!")
            sys.exit(1)
            
    finally:
        # Cleanup
        print("\nCleaning up temporary files...")
        if temp_dataset.exists():
            temp_dataset.unlink()
            print("Removed temporary dataset.")
        if temp_report.exists():
            temp_report.unlink()
            print("Removed temporary audit report.")
            
if __name__ == "__main__":
    main()
