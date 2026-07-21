import sys
import subprocess
import tempfile
import json
import os
from pathlib import Path

def create_valid_dataset_line(idx):
    text = f"Some prompt for record {idx}: ### Response:\nSome completion of sufficient length that is unique {idx} " + "word " * 100
    record = {
        "text": text,
        "category": "general_instruction_code",
        "prompt": f"Some prompt for record {idx}: ### Response:\n",
        "completion": f"Some completion of sufficient length that is unique {idx} " + "word " * 100
    }
    return json.dumps(record)

project_root = Path(__file__).resolve().parent.parent
verify_script = project_root / "tests" / "e2e_verify.py"

fd_d, temp_d_path = tempfile.mkstemp(suffix=".jsonl", prefix="adv_d_")
fd_r, temp_r_path = tempfile.mkstemp(suffix=".json", prefix="adv_r_")
os.close(fd_d)
os.close(fd_r)

try:
    with open(temp_d_path, "w", encoding="utf-8") as f:
        f.write(create_valid_dataset_line(1) + "\n")
    with open(temp_r_path, "w", encoding="utf-8") as f:
        json.dump({"llm_judge": {"evaluated_samples": 100, "passed_samples": 98}}, f)
        
    cmd = [
        sys.executable,
        str(verify_script),
        "--dataset", str(temp_d_path),
        "--report", str(temp_r_path),
        "--tokenizer", "google/gemma-4-E4B"
    ]
    print("Command:", cmd)
    print("Executable exists:", Path(sys.executable).exists())
    print("Verify script exists:", verify_script.exists())
    print("Dataset exists:", Path(temp_d_path).exists())
    print("Report exists:", Path(temp_r_path).exists())
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Exit code:", res.returncode)
    print("Stdout:", res.stdout)
    print("Stderr:", res.stderr)
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    for p in [temp_d_path, temp_r_path]:
        if os.path.exists(p):
            os.unlink(p)
