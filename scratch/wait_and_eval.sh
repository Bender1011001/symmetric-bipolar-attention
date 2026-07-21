#!/bin/bash
echo "Waiting for training to complete..."
while pgrep -f run_homotopy_conversion.py > /dev/null; do
    sleep 30
done
echo "Training completed! Running evaluation..."
/venv/main/bin/python sba_standalone/evaluate_scratch_tokenized_checkpoint.py \
    --checkpoint sba_standalone/results/homotopy_conversion_corrected/homotopy_step003000_12l_12h_768d.pt \
    --max-new-tokens 50 \
    --output sba_standalone/results/homotopy_conversion_corrected/eval_final.json
echo "All done!"
