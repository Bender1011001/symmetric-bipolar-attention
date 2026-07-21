#!/bin/bash
set -e

echo "Starting V5 Hard Homotopy training..."
/venv/main/bin/python sba_standalone/run_homotopy_conversion.py \
    --output-dir sba_standalone/results/homotopy_conversion_v5 \
    --steps 4000 \
    --warmup-steps 200 \
    --batch-size 8 \
    --grad-accum 4 \
    --checkpoint-every 500 \
    --max-length 1024 \
    --data-files data/open_reason_code_mix.jsonl

echo "Training complete! Evaluating all checkpoints..."
for step in 500 1000 1500 2000 2500 3000 3500 4000; do
    step_padded=$(printf "%06d" $step)
    ckpt="sba_standalone/results/homotopy_conversion_v5/homotopy_step${step_padded}_12l_12h_768d.pt"
    if [ -f "$ckpt" ]; then
        echo "Evaluating $ckpt..."
        /venv/main/bin/python sba_standalone/evaluate_scratch_tokenized_checkpoint.py \
            --checkpoint "$ckpt" \
            --max-new-tokens 50 \
            --output "sba_standalone/results/homotopy_conversion_v5/eval_step${step_padded}.json"
    else
        echo "Checkpoint $ckpt not found!"
    fi
done

echo "All evaluations done!"
