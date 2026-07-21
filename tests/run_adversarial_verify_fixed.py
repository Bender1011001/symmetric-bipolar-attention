#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adversarial test runner for tests/e2e_verify.py (with diagnostic prints).
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
    print(f"DEBUG run_verify cmd: {cmd}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode, res.stdout, res.stderr
    except Exception as e:
        print(f"DEBUG run_verify failed: {e}")
        traceback.print_exc()
        raise e

def create_valid_dataset_line(idx, cat="general_instruction_code"):
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
    print("RUNNING DIAGNOSTIC ADVERSARIAL ROBUSTNESS CHECKS ON e2e_verify.py")
    print("=" * 80)
    
    test_cases = []
    
    # TC1
    test_cases.append({
        "name": "TC1: Valid baseline (small dataset, should fail on checks but exit cleanly without traceback)",
        "setup": lambda d_path, r_path: (
            open(d_path, "w", encoding="utf-8").write(
                create_valid_dataset_line(1) + "\n"
            ),
            json.dump({"llm_judge": {"evaluated_samples": 100, "passed_samples": 98}}, open(r_path, "w", encoding="utf-8"))
        ),
        "expected_exit_code": 7,
        "must_not_contain_traceback": True
    })

    # TC2
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
        "expected_exit_code": 3,
        "must_not_contain_traceback": True
    })

    # Run tests
    failures = 0
    for tc in test_cases:
        print_info(f"Running {tc['name']}")
        
        fd_d, temp_d_path = tempfile.mkstemp(suffix=".jsonl", prefix="adv_d_")
        fd_r, temp_r_path = tempfile.mkstemp(suffix=".json", prefix="adv_r_")
        os.close(fd_d)
        os.close(fd_r)
        
        try:
            tc["setup"](temp_d_path, temp_r_path)
            exit_code, stdout, stderr = run_verify(verify_script, temp_d_path, temp_r_path)
            
            has_traceback = "Traceback (most recent call last):" in stderr or "Traceback (most recent call last):" in stdout
            
            print(f"  Exit code: {exit_code}")
            if has_traceback:
                print_failure("  CRASH DETECTED! Traceback found in output:")
                print("-" * 50)
                print(stderr if "Traceback" in stderr else stdout)
                print("-" * 50)
                failures += 1
            else:
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
            failures += 1
        finally:
            for path in [temp_d_path, temp_r_path]:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception as cleanup_err:
                    print_info(f"  Could not delete temp file {path}: {cleanup_err}")
                    
        print("-" * 80)
        
    if failures == 0:
        print_success("All adversarial test scenarios handled gracefully.")
    else:
        print_failure(f"{failures} failure(s) detected.")

if __name__ == "__main__":
    main()
