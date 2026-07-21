#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Concurrency test for tests/run_e2e_test_validation.py.
This script executes two instances of the validation script in parallel
to verify that there are no filename collisions or race conditions.
"""

import subprocess
import concurrent.futures
import sys
from pathlib import Path

def run_validation():
    project_root = Path(__file__).resolve().parent.parent
    verify_script = project_root / "tests" / "run_e2e_test_validation.py"
    cmd = [sys.executable, str(verify_script)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr

def main():
    print("Starting 2 concurrent instances of run_e2e_test_validation.py...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_validation) for _ in range(2)]
        results = [f.result() for f in futures]
    
    all_passed = True
    for idx, (code, stdout, stderr) in enumerate(results):
        print(f"\n--- Instance {idx+1} Output ---")
        print(f"Exit code: {code}")
        lines = stdout.splitlines()
        # Print the last 15 lines of output to verify status and cleanup
        print("\n".join(lines[-15:]))
        if code != 0:
            all_passed = False
            print(f"Error output:\n{stderr}")
            
    if all_passed:
        print("\n[PASS] Concurrency test passed: Both instances finished successfully without collisions!")
        sys.exit(0)
    else:
        print("\n[FAIL] Concurrency test failed: One or both instances failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
