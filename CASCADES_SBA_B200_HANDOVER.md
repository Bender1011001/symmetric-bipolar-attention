# CASCADES SBA B200 Handover Document

## 🌐 Remote B200 Host Connection
* **Host Address**: `44.227.210.72`
* **SSH Port**: `41569`
* **User**: `root`
* **SSH Connection Command**:
  ```bash
  ssh -p 41569 root@44.227.210.72 -L 8080:localhost:8080
  ```
  *(Note: A shortcut `Host b200` has been configured in the local `C:\Users\admin\.ssh\config` file using the `dualsystem_h100_profile_20260424` key, so you can also connect via `ssh b200`)*.

---

## 📈 Current Training Status (As of 7:25 PM Local Time)
The run is active on the B200 GPU and has crossed the halfway mark:
* **Current Step**: `2583 / 5000` (**52% completed**)
* **Current Stage**: `stage2_recovery` (recovering language modelling capacity under full SBA Patched Attention)
* **Step Latency**: **~1.77 seconds per step**
* **Expected Completion Time (ETA)**: **~8:37 PM local time** (~1 hour 11 minutes remaining)
* **Metrics (Step 2580)**:
  * **KL Divergence Loss**: `0.429`
  * **Cross-Entropy Loss (LM Coherence)**: `0.490` (dropping steadily)
  * **SBA Row $L_1$ Loss**: `0.0518` (Row $L_1$ mean `0.876` — successfully breaking softmax constraints)
  * **$r_{BOS}$ Loss**: `6.96e-6` (BOS attention sink density `0.0207` — indicates attention sink on BOS is nearly completely suppressed)
  * **Total Loss**: `0.8523`

---

## 📂 Directories & Log Files on B200
* **Working Directory**: `/root/sba_standalone`
* **Active Python Env**: `/venv/main/bin/python`
* **Training Output Folder**: `/root/sba_standalone/results/gemma4_e4b_sba_ecot_b200`
* **Training Log File**:
  ```bash
  tail -f /root/sba_standalone/results/gemma4_e4b_sba_ecot_b200/gemma4_ecot_b200_training.log
  ```
* **Background Process Logs**:
  ```bash
  cat /root/sba_standalone/b200_training_init.log
  ```

---

## 🛠️ Work Done This Session
1. **Blackwell 10k Run Audit**: Confirmed the 10,000-step Softmax-to-SBA Bridge training run for `gemma-4-E4B` completed successfully.
2. **GPU-Accelerated Diagnostics**: Upgraded the diagnostics script [gemma4_diagnostics_v3.py](file:///e:/code.projects/CASCADES--continual-PEFT-for-Local-LLMs/sba_standalone/gemma4_diagnostics_v3.py) to run on GPU, generated an up-to-date trajectory report showing successful breakout of softmax row $L_1$ sum constraint to `0.7992` and decoupling of BOS attention sinks down to `9.86%`.
3. **Dataset Packaging**: Copied the 55k record ECoT V2 dataset files (including `ecot_train.jsonl` with Confident Proposals, Inhibit, and Recalculate phases) into the local `sba_standalone/data` folder.
4. **B200 Script Configuration**: Created [run_b200_ecot_training.sh](file:///e:/code.projects/CASCADES--continual-PEFT-for-Local-LLMs/sba_standalone/run_b200_ecot_training.sh) with optimized parameters (sequence length `1024`, batch size `16`, gradient accumulation `2`).
5. **Instance Bootstrap**: Connected to the fresh B200 GPU instance, transferred the compressed workspace tarball, installed all package dependencies (`transformers`, `datasets`, `peft`, `accelerate`) under `/venv/main/bin/pip`, and successfully kicked off the training run.

---

## 📋 Next Steps for Incoming Agent
Once training reaches step 5000 and successfully finishes:
1. **Run Diagnostics**:
   Run the GPU-accelerated diagnostics script on B200 to generate the final trajectory metrics:
   ```bash
   /venv/main/bin/python /root/sba_standalone/gemma4_diagnostics_v3.py
   ```
2. **Run MMLU Evaluation**:
   Evaluate MMLU performance of the final SBA checkpoint using a smaller batch size to avoid VRAM overhead:
   ```bash
   /venv/main/bin/python /root/sba_standalone/evaluate_mmlu.py \
       --checkpoint_dir /root/sba_standalone/results/gemma4_e4b_sba_ecot_b200 \
       --batch_size 4 \
       --output_file /root/sba_standalone/results/mmlu_final_quick.json
   ```
3. **Download Results**:
   Sync the final weights, diagnostics JSON, and MMLU evaluation reports back to the local workspace folder.
