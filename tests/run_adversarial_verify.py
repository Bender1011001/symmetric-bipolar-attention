#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adversarial test runner for tests/e2e_verify.py.
This script tests how e2e_verify.py handles various malformed inputs and boundary conditions.
"""

import os
import sys
import json
import subprocess
import tempfile
import traceback
from pathlib import Path

# Color helpers for terminal output
def print_success(msg):
    print(f"\033[92m[PASS] {msg}\033[0m")

def print_failure(msg):
    print(f"\033[91m[FAIL] {msg}\033[0m")

def print_info(msg):
    print(f"[INFO] {msg}")

def run_verify(verify_script, dataset_path, report_path, tokenizer_name="google/gemma-4-E4B"):
    cmd = [
        sys.executable,
        str(verify_script),
        "--dataset", str(dataset_path),
        "--report", str(report_path),
        "--tokenizer", tokenizer_name
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr

def create_valid_dataset_line(idx, cat="general_instruction_code"):
    # Generate minimal valid line to pass basic checks
    # The tokenizer for gemma-4-E4B can tokenize standard text
    text = f"Some prompt for record {idx}: ### Response:\nSome completion of sufficient length that is unique {idx} " + "word " * 100
    record = {
        "text": text,
        "category": cat,
        "prompt": f"Some prompt for record {idx}: ### Response:\n",
        "completion": f"Some completion of sufficient length that is unique {idx} " + "word " * 100
    }
    return json.dumps(record)

def main():
    project_root = Path(__file__).resolve().parent.parent
    verify_script = project_root / "tests" / "e2e_verify.py"
    
    if not verify_script.exists():
        print_failure(f"Verification script not found at {verify_script}")
        sys.exit(1)
        
    print("=" * 80)
    print("RUNNING ADVERSARIAL ROBUSTNESS CHECKS ON e2e_verify.py")
    print("=" * 80)
    
    test_cases = []
    
    # ----------------------------------------------------
    # Test Case 1: Valid Baseline (To ensure testing setup works)
    # ----------------------------------------------------
    test_cases.append({
        "name": "TC1: Valid baseline (small dataset, should fail on checks but exit cleanly without traceback)",
        "setup": lambda d_path, r_path: (
            open(d_path, "w", encoding="utf-8").write(
                create_valid_dataset_line(1) + "\n"
            ),
            json.dump({"llm_judge": {"evaluated_samples": 100, "passed_samples": 98}}, open(r_path, "w", encoding="utf-8"))
        ),
        "expected_exit_code": 7, # Failing acceptance criteria (not enough records)
        "must_not_contain_traceback": True
    })

    # ----------------------------------------------------
    # Test Case 2: JSONL with invalid JSON lines
    # ----------------------------------------------------
    test_cases.append({
        "name": "TC2: JSONL dataset contains invalid JSON line",
        "setup": lambda d_path, r_path: (
            open(d_path, "w", encoding="utf-8").write(
                create_valid_dataset_line(1) + "\n" +
                "{invalid_json_here}\n" +
                create_valid_dataset_line(2) + "\n"
            ),
            json.dump({"llm_judge": {"evaluated_samples": 100, "passed_samples": 98}}, open(r_path, "w", encoding="utf-8"))
        ),
        "expected_exit_code": 3, # Invalid JSON error code
        "must_not_contain_traceback": True
    })

    # ----------------------------------------------------
    # Test Case 3: JSONL with empty lines
    # ----------------------------------------------------
    test_cases.append({
        "name": "TC3: JSONL dataset contains empty lines and whitespace lines",
        "setup": lambda d_path, r_path: (
            open(d_path, "w", encoding="utf-8").write(
                "\n" +
                create_valid_dataset_line(1) + "\n" +
                "   \n" +
                create_valid_dataset_line(2) + "\n" +
                "\n"
            ),
            json.dump({"llm_judge": {"evaluated_samples": 100, "passed_samples": 98}}, open(r_path, "w", encoding="utf-8"))
        ),
        "expected_exit_code": 7, # Should skip empty lines, parse records, and fail on min_records check (code 7)
        "must_not_contain_traceback": True
    })

    # ----------------------------------------------------
    # Test Case 4: Audit report is completely empty (0 bytes)
    # ----------------------------------------------------
    test_cases.append({
        "name": "TC4: Audit report is completely empty (0 bytes)",
        "setup": lambda d_path, r_path: (
            open(d_path, "w", encoding="utf-8").write(create_valid_dataset_line(1) + "\n"),
            open(r_path, "w", encoding="utf-8").close() # Create 0-byte file
        ),
        "expected_exit_code": 5, # Failed to parse audit report JSON
        "must_not_contain_traceback": True
    })

    # ----------------------------------------------------
    # Test Case 5: Dataset is completely empty (0 bytes)
    # ----------------------------------------------------
    test_cases.append({
        "name": "TC5: Dataset is completely empty (0 bytes)",
        "setup": lambda d_path, r_path: (
            open(d_path, "w", encoding="utf-8").close(), # Create 0-byte file
            json.dump({"llm_judge": {"evaluated_samples": 100, "passed_samples": 98}}, open(r_path, "w", encoding="utf-8"))
        ),
        "expected_exit_code": 7, # Should parse 0 records and fail on check constraints (not crash with division by zero)
        "must_not_contain_traceback": True
    })

    # ----------------------------------------------------
    # Test Case 6: Audit report has missing keys
    # ----------------------------------------------------
    test_cases.append({
        "name": "TC6: Audit report has missing keys (empty JSON object)",
        "setup": lambda d_path, r_path: (
            open(d_path, "w", encoding="utf-8").write(create_valid_dataset_line(1) + "\n"),
            json.dump({}, open(r_path, "w", encoding="utf-8"))
        ),
        "expected_exit_code": 6, # Could not find LLM judge evaluation details
        "must_not_contain_traceback": True
    })

    # ----------------------------------------------------
    # Test Case 7: Audit report is not a dictionary (e.g. integer)
    # ----------------------------------------------------
    test_cases.append({
        "name": "TC7: Audit report is not a JSON object (integer)",
        "setup": lambda d_path, r_path: (
            open(d_path, "w", encoding="utf-8").write(create_valid_dataset_line(1) + "\n"),
            open(r_path, "w", encoding="utf-8").write("123")
        ),
        "expected_exit_code": None, # We want to observe if it raises traceback (if it crashes, exit code is likely 1 or TypeError)
        "must_not_contain_traceback": True
    })

    # ----------------------------------------------------
    # Test Case 8: Audit report contains wrong data type for passed_samples
    # ----------------------------------------------------
    test_cases.append({
        "name": "TC8: Audit report contains string for passed_samples",
        "setup": lambda d_path, r_path: (
            open(d_path, "w", encoding="utf-8").write(create_valid_dataset_line(1) + "\n"),
            json.dump({"llm_judge": {"evaluated_samples": 100, "passed_samples": "ninety-eight"}}, open(r_path, "w", encoding="utf-8"))
        ),
        "expected_exit_code": None,
        "must_not_contain_traceback": True
    })

    # ----------------------------------------------------
    # Test Case 9: Dataset record is not a dictionary (e.g. integer)
    # ----------------------------------------------------
    test_cases.append({
        "name": "TC9: Dataset record is a JSON integer instead of an object",
        "setup": lambda d_path, r_path: (
            open(d_path, "w", encoding="utf-8").write("123\n"),
            json.dump({"llm_judge": {"evaluated_samples": 100, "passed_samples": 98}}, open(r_path, "w", encoding="utf-8"))
        ),
        "expected_exit_code": 4, # Reading dataset file failed
        "must_not_contain_traceback": True
    })

    # ----------------------------------------------------
    # Test Case 10: Dataset record has a list for 'text' field instead of string
    # ----------------------------------------------------
    test_cases.append({
        "name": "TC10: Dataset record has a list for 'text' field instead of string",
        "setup": lambda d_path, r_path: (
            open(d_path, "w", encoding="utf-8").write(
                json.dumps({"text": [], "category": "plain_prose"}) + "\n"
            ),
            json.dump({"llm_judge": {"evaluated_samples": 100, "passed_samples": 98}}, open(r_path, "w", encoding="utf-8"))
        ),
        "expected_exit_code": None,
        "must_not_contain_traceback": True
    })

    # Run tests
    failures = 0
    for tc in test_cases:
        print_info(f"Running {tc['name']}")
        
        # Create temp files
        fd_d, temp_d_path = tempfile.mkstemp(suffix=".jsonl", prefix="adv_d_")
        fd_r, temp_r_path = tempfile.mkstemp(suffix=".json", prefix="adv_r_")
        os.close(fd_d)
        os.close(fd_r)
        
        try:
            # Set up files
            tc["setup"](temp_d_path, temp_r_path)
            
            # Run verify script
            exit_code, stdout, stderr = run_verify(verify_script, temp_d_path, temp_r_path)
            
            # Check for tracebacks
            has_traceback = "Traceback (most recent call last):" in stderr or "Traceback (most recent call last):" in stdout
            
            print(f"  Exit code: {exit_code}")
            if has_traceback:
                print_failure("  CRASH DETECTED! Traceback found in output:")
                print("-" * 50)
                print(stderr if "Traceback" in stderr else stdout)
                print("-" * 50)
                failures += 1
            else:
                # If expected exit code is specified, verify it
                if tc["expected_exit_code"] is not None:
                    if exit_code == tc["expected_exit_code"]:
                        print_success(f"  Exited with expected code {exit_code}")
                    else:
                        print_failure(f"  Exited with code {exit_code}, but expected {tc['expected_exit_code']}")
                        failures += 1
                else:
                    print_success(f"  Exited with code {exit_code} (no traceback)")
                    
        except Exception as e:
            print_failure(f"  Test harness exception: {e}")
            traceback.print_exc()
            failures += 1
        finally:
            # Cleanup temp files safely
            for path in [temp_d_path, temp_r_path]:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception as cleanup_err:
                    print_info(f"  Could not delete temp file {path}: {cleanup_err}")
                    
        print("-" * 80)
        
    if failures == 0:
        print_success("All adversarial test scenarios handled gracefully (no tracebacks, expected exit codes).")
    else:
        print_failure(f"{failures} adversarial test failure(s) / crash(es) detected.")
        sys.exit(1)

if __name__ == "__main__":
    main()
