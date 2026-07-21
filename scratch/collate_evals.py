import json
import re
from pathlib import Path

def collate_evals(results_dir: Path):
    folders = [
        "softmax", "softmax_seed43", "softmax_seed44",
        "sba", "sba_seed43", "sba_seed44",
        "softmax_iso"
    ]
    
    print("# Downstream Task Evaluations: CASCADES Continual PEFT (125M Blackwell Sweep)")
    print("\nThis summary shows the correctness of each downstream task at different training steps.\n")

    for folder in folders:
        folder_dir = results_dir / folder
        if not folder_dir.exists():
            continue
            
        eval_files = sorted(list(folder_dir.glob("task_eval_step*.json")))
        if not eval_files:
            continue
            
        print(f"## Run: **{folder}**\n")
        
        # We want to dynamically extract the list of tasks from the first file
        try:
            with eval_files[0].open("r", encoding="utf-8") as f:
                data = json.load(f)
                tasks = [t["task_id"] for t in data.get("results", [])]
        except Exception as e:
            print(f"Error loading tasks from {eval_files[0]}: {e}\n")
            continue
            
        # Headers: Step | Task 1 | Task 2 | ... | Accuracy
        headers = ["Step"] + tasks + ["Overall Acc"]
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for file in eval_files:
            # Extract step number from filename
            match = re.search(r"step(\d+)", file.name)
            step_str = f"@{int(match.group(1))}" if match else file.name
            
            try:
                with file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                results_dict = {r["task_id"]: r["score"].get("correct", False) for r in data.get("results", [])}
                summary = data.get("summary", {})
                acc_pct = summary.get("accuracy", 0.0) * 100.0
                
                row = [step_str]
                for task in tasks:
                    correct = results_dict.get(task, False)
                    row.append("✅ PASS" if correct else "❌ FAIL")
                row.append(f"{acc_pct:.1f}%")
                
                print("| " + " | ".join(row) + " |")
            except Exception as e:
                print(f"| {step_str} | *Error reading file* |")
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    results_path = Path("sba_standalone/results/blackwell_125m_overnight")
    collate_evals(results_path)
