#!/usr/bin/env python3
"""Extract all sweep results for iso-clock analysis."""
import json
import os
import glob

base = "/root/sba_standalone/results/blackwell_125m_overnight"

results_files = glob.glob(os.path.join(base, "*/result.json"))
results_files.sort()

print("=" * 80)
print("ALL SWEEP RESULTS — blackwell_125m_overnight")
print("=" * 80)

for f in results_files:
    name = os.path.basename(os.path.dirname(f))
    with open(f) as fh:
        d = json.load(fh)
    attn = d.get("attention", "?")
    steps = d.get("steps", "?")
    val = d.get("final_val_loss", float("nan"))
    best = d.get("best_val_loss_seen", float("nan"))
    elapsed = d.get("elapsed_seconds", 0)
    ms_per_step = (elapsed / steps * 1000) if steps and steps > 0 else 0
    print(f"  {name:20s} | attn={attn:10s} | steps={steps:6d} | val_loss={val:.6f} | best={best:.6f} | {elapsed:.0f}s | {ms_per_step:.1f}ms/step")

# Now extract the softmax val_loss curve for interpolation
print("\n" + "=" * 80)
print("SOFTMAX VAL_LOSS CURVE (seed42, for interpolation)")
print("=" * 80)

metrics_file = os.path.join(base, "softmax/metrics.jsonl")
if os.path.exists(metrics_file):
    with open(metrics_file) as fh:
        lines = fh.readlines()
    # Extract val_loss at key milestones
    for line in lines:
        d = json.loads(line)
        step = d.get("step", 0)
        if step % 500 == 0 or step >= 14000:
            if step % 250 == 0:
                print(f"  step={step:6d} | val_loss={d['val_loss']:.6f}")

# Also check multi-seed SBA/softmax for average
print("\n" + "=" * 80)
print("MULTI-SEED SUMMARY")
print("=" * 80)

sba_vals = []
sm_vals = []
for f in results_files:
    name = os.path.basename(os.path.dirname(f))
    with open(f) as fh:
        d = json.load(fh)
    attn = d.get("attention", "?")
    val = d.get("final_val_loss", None)
    steps = d.get("steps", 0)
    if val and steps == 15000:
        if attn == "sba":
            sba_vals.append((name, val))
        elif attn == "softmax" and "iso" not in name:
            sm_vals.append((name, val))

print("SBA at 15k steps:")
for name, val in sba_vals:
    print(f"  {name}: {val:.6f}")
if sba_vals:
    avg = sum(v for _, v in sba_vals) / len(sba_vals)
    print(f"  Average: {avg:.6f}")

print("Softmax at 15k steps:")
for name, val in sm_vals:
    print(f"  {name}: {val:.6f}")
if sm_vals:
    avg = sum(v for _, v in sm_vals) / len(sm_vals)
    print(f"  Average: {avg:.6f}")

# Iso-clock result
print("\nSoftmax iso-clock:")
iso_file = os.path.join(base, "softmax_iso/result.json")
if os.path.exists(iso_file):
    with open(iso_file) as fh:
        d = json.load(fh)
    print(f"  steps={d.get('steps')}, val_loss={d.get('final_val_loss'):.6f}, elapsed={d.get('elapsed_seconds'):.0f}s")
    print(f"  ms/step = {d.get('elapsed_seconds') / d.get('steps') * 1000:.1f}")
