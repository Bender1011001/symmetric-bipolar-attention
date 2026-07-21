import os
import json
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("Loading ServiceNow-AI/R1-Distill-SFT dataset from Hugging Face...")
    try:
        ds = load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1", split="train")
    except Exception as e:
        print(f"Error loading dataset from HF: {e}")
        return

    print(f"Dataset loaded. Total samples: {len(ds)}")
    
    max_samples = 15000
    output_file = "/root/sba_standalone/results/reasoning_sft_data.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Formatting first {max_samples} samples using Gemma chat template with <think> tags...")
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for idx in tqdm(range(min(max_samples, len(ds)))):
            row = ds[idx]
            
            prompt = row.get("question") or row.get("prompt") or row.get("instruction")
            thought = row.get("thought") or row.get("reasoning")
            response = row.get("output") or row.get("response") or row.get("answer")
            
            if not prompt or not response:
                continue
                
            if not thought:
                thought = ""
            
            formatted_text = (
                f"<start_of_turn>user\n{prompt.strip()}<end_of_turn>\n"
                f"<start_of_turn>model\n<think>\n{thought.strip()}\n</think>\n{response.strip()}<end_of_turn>"
            )
            
            item = {"text": formatted_text}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            
    print(f"Successfully created reasoning dataset at: {output_file}")
    print(f"Total formatted examples written: {count}")

if __name__ == "__main__":
    main()
