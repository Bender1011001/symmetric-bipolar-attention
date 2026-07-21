#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Concurrency stress test for tests/run_e2e_test_validation.py.
This script executes 5 instances of the validation script in parallel
to verify complete absence of file locking or collisions.
"""

import subprocess
import concurrent.futures
import sys
from pathlib import Path

def run_validation(instance_id):
    print(f"[INFO] Launching concurrency instance {instance_id}...")
    project_root = Path(__file__).resolve().parent.parent
    verify_script = project_root / "tests" / "run_e2e_test_validation.py"
    cmd = [sys.executable, str(verify_script)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return instance_id, res.returncode, res.stdout, res.stderr

def main():
    num_instances = 5
    print(f"Starting {num_instances} concurrent instances of run_e2e_test_validation.py...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_instances) as executor:
        # Submit all tasks
        futures = {executor.submit(run_validation, i): i for i in range(1, num_instances + 1)}
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            
    # Sort results by instance ID for cleaner output
    results.sort(key=lambda x: x[0])
    
    all_passed = True
    for instance_id, code, stdout, stderr in results:
        print(f"\n" + "=" * 50)
        print(f"INSTANCE {instance_id} RESULTS (Exit Code: {code})")
        print("=" * 50)
        
        # Output summary lines
        lines = stdout.splitlines()
        summary_lines = [l for l in lines if "[+]" in l or "[-]" in l or "SUCCESS" in l or "FAIL" in l]
        if summary_lines:
            print("Summary findings:")
            for sl in summary_lines:
                print(f"  {sl}")
        else:
            # Fallback to last few lines
            print("\n".join(lines[-10:]))
            
        if code != 0:
            all_passed = False
            print(f"\nError output for Instance {instance_id}:\n{stderr}")
            
    print("\n" + "=" * 50)
    if all_passed:
        print(f"[PASS] Concurrency stress test passed: All {num_instances} instances finished successfully with exit code 0!")
        sys.exit(0)
    else:
        print(f"[FAIL] Concurrency stress test failed: One or more instances exited with non-zero code.")
        sys.exit(1)

if __name__ == "__main__":
    main()
