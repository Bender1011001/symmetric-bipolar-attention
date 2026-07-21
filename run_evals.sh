#!/bin/bash
set -e

echo "Starting evaluation suite for step 15000..."

cd /root/sba_standalone

# 1. Run Diagnostics
echo "Running Diagnostics..."
/venv/main/bin/python /root/sba_standalone/gemma4_diagnostics_v4.py

# 2. Run MMLU Quick
echo "Running MMLU Quick Evaluation (10 subjects, 30 questions)..."
/venv/main/bin/python /root/sba_standalone/evaluate_mmlu.py \
    --checkpoint_dir /root/sba_standalone/results/gemma4_e4b_sba_ecot_b200/checkpoint-15000 \
    --subjects "high_school_psychology,high_school_computer_science,clinical_knowledge,conceptual_physics,professional_accounting,college_computer_science,elementary_mathematics,college_mathematics,machine_learning,abstract_algebra" \
    --limit_questions 30 \
    --batch_size 4 \
    --output_file /root/sba_standalone/results/mmlu_final_quick.json

# 3. Run Gradient Audit
echo "Running Section 5 Gradient Audit..."
/venv/main/bin/python /root/sba_standalone/gemma4_gradient_audit.py \
    --checkpoint_dir /root/sba_standalone/results/gemma4_e4b_sba_ecot_b200/checkpoint-15000 \
    --validation_file /root/sba_standalone/data/ecot_validation.jsonl \
    --output_file /root/sba_standalone/results/gradient_audit_report.json

# 4. Run Held-out Trap Evaluation
echo "Running Held-out Trap Evaluation..."
/venv/main/bin/python /root/sba_standalone/gemma4_heldout_trap_eval.py \
    --checkpoint_dir /root/sba_standalone/results/gemma4_e4b_sba_ecot_b200/checkpoint-15000 \
    --output_file /root/sba_standalone/results/heldout_trap_report.json

echo "Evaluation suite finished successfully."
