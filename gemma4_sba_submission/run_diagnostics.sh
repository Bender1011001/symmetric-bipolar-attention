#!/bin/bash
# run_diagnostics.sh
# Runs the precise, GQA-safe attention metrics evaluation script.
# Executes on CPU to avoid VRAM OOM conflicts with active training runs.

echo "Running Gemma 4 Symmetric Bipolar Attention (SBA) diagnostics..."

python gemma4_diagnostics_v3.py

echo "Diagnostics completed! Results saved to results/gemma4_e4b_sba_bridge/gemma4_diagnostics_report.json"
