import json
from pathlib import Path
import numpy as np

def collate_metrics(results_dir: Path):
    # Map from arm folder name to label/seed metadata
    runs = {
        "softmax": {"name": "Softmax", "seed": 42, "is_iso": False},
        "softmax_seed43": {"name": "Softmax", "seed": 43, "is_iso": False},
        "softmax_seed44": {"name": "Softmax", "seed": 44, "is_iso": False},
        "sba": {"name": "SBA", "seed": 42, "is_iso": False},
        "sba_seed43": {"name": "SBA", "seed": 43, "is_iso": False},
        "sba_seed44": {"name": "SBA", "seed": 44, "is_iso": False},
        "softmax_iso": {"name": "Softmax (Iso-Clock)", "seed": 42, "is_iso": True},
    }
    
    summary = {}

    for folder, meta in runs.items():
        metrics_file = results_dir / folder / "metrics.jsonl"
        if not metrics_file.exists():
            continue
        
        steps = []
        train_losses = []
        val_losses = []
        peak_vram = []
        elapsed_times = []
        
        with metrics_file.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    steps.append(data["step"])
                    train_losses.append(data.get("train_loss"))
                    val_losses.append(data.get("val_loss"))
                    peak_vram.append(data.get("peak_vram_mb"))
                    elapsed_times.append(data.get("elapsed_seconds"))
                except Exception as e:
                    pass
                    
        if steps:
            final_step = steps[-1]
            valid_val_losses = [v for v in val_losses if v is not None]
            if not valid_val_losses:
                continue
            best_val_idx = val_losses.index(min(valid_val_losses))
            best_val_loss = val_losses[best_val_idx]
            best_step = steps[best_val_idx]
            max_vram = max(v for v in peak_vram if v is not None)
            total_time = elapsed_times[-1]
            
            # Compute average step time after warmup/stabilization (step 1000 onwards)
            stable_times = []
            stable_steps = []
            for s, t in zip(steps, elapsed_times):
                if s >= 1000:
                    stable_steps.append(s)
                    stable_times.append(t)
            
            if len(stable_times) > 1:
                avg_step_sec = (stable_times[-1] - stable_times[0]) / (stable_steps[-1] - stable_steps[0])
            else:
                avg_step_sec = total_time / final_step if final_step > 0 else 0
                
            summary[folder] = {
                "name": meta["name"],
                "seed": meta["seed"],
                "is_iso": meta["is_iso"],
                "final_val_loss": val_losses[-1],
                "best_val_loss": best_val_loss,
                "best_step": best_step,
                "peak_vram_gb": max_vram / 1024.0,
                "total_time_hours": total_time / 3600.0,
                "avg_step_ms": avg_step_sec * 1000.0,
                "steps_per_sec": 1.0 / avg_step_sec if avg_step_sec > 0 else 0,
                "steps": final_step
            }

    print("# CASCADES Continual PEFT: 125M Blackwell Rigorous Evaluation Sweep\n")
    print("## 1. Individual Run Details\n")
    print("| Folder | Architecture | Seed | Steps | Final Val Loss | Best Val Loss (Step) | Peak VRAM | Avg Step Time | Total Time |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for folder, meta in runs.items():
        if folder in summary:
            s = summary[folder]
            print(f"| `{folder}` | {s['name']} | {s['seed']} | {s['steps']} | {s['final_val_loss']:.5f} | {s['best_val_loss']:.5f} (@{s['best_step']}) | {s['peak_vram_gb']:.2f} GB | {s['avg_step_ms']:.1f} ms | {s['total_time_hours']:.2f} hrs |")
        else:
            print(f"| `{folder}` | {meta['name']} | {meta['seed']} | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |")

    print("\n## 2. Statistical Aggregates (n=3 Seed Sweep for 15k steps)\n")
    print("| Architecture | Final Val Loss (Mean ± Std) | Best Val Loss (Mean ± Std) | Peak VRAM (Avg) | Speed (Avg) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    # Calculate aggregates for Softmax (seeds 42, 43, 44)
    softmax_runs = [summary[k] for k in ["softmax", "softmax_seed43", "softmax_seed44"] if k in summary]
    sba_runs = [summary[k] for k in ["sba", "sba_seed43", "sba_seed44"] if k in summary]
    
    for label, r_list in [("Softmax Baseline", softmax_runs), ("SBA Treatment", sba_runs)]:
        if len(r_list) > 0:
            final_vals = [r["final_val_loss"] for r in r_list]
            best_vals = [r["best_val_loss"] for r in r_list]
            vrams = [r["peak_vram_gb"] for r in r_list]
            speeds = [r["avg_step_ms"] for r in r_list]
            
            mean_final, std_final = np.mean(final_vals), np.std(final_vals)
            mean_best, std_best = np.mean(best_vals), np.std(best_vals)
            avg_vram = np.mean(vrams)
            avg_speed = np.mean(speeds)
            
            print(f"| **{label}** | {mean_final:.5f} ± {std_final:.5f} | {mean_best:.5f} ± {std_best:.5f} | {avg_vram:.2f} GB | {avg_speed:.1f} ms |")
        else:
            print(f"| **{label}** | *Insufficient data* | *Insufficient data* | *N/A* | *N/A* |")

    print("\n## 3. Compute-Parity & Latency Trade-Off\n")
    if "sba" in summary and "softmax_iso" in summary:
        sba_val = summary["sba"]["final_val_loss"]
        iso_val = summary["softmax_iso"]["final_val_loss"]
        diff = iso_val - sba_val
        print(f"* **SBA (15,000 steps)**: Final Val Loss = **{sba_val:.5f}** (Wall-clock: {summary['sba']['total_time_hours']:.2f} hrs)")
        print(f"* **Softmax (Iso-Clock: 22,760 steps)**: Final Val Loss = **{iso_val:.5f}** (Wall-clock: {summary['softmax_iso']['total_time_hours']:.2f} hrs)")
        print(f"* **Margin (Iso-Clock Softmax - SBA)**: **{diff:.5f}** nats")
        if diff > 0:
            print(f"  * 👉 **SBA outperforms Softmax** even when Softmax is given 1.52x more training steps to match SBA's compute budget.")
        else:
            print(f"  * 👉 **Softmax outperforms SBA** when matching wall-clock compute, indicating SBA's edge is illusory under iso-compute budget.")
    else:
        print("* *Waiting for completed runs to resolve compute-parity comparison.*")

if __name__ == "__main__":
    results_path = Path("sba_standalone/results/blackwell_125m_overnight")
    collate_metrics(results_path)
