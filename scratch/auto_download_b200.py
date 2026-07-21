import os
import time
import subprocess
import datetime

REMOTE_HOST = "root@44.227.210.72"
PORT = "41569"
KEY_PATH = "C:/Users/admin/.ssh/dualsystem_h100_profile_20260424"
REMOTE_LOG = "/root/sba_standalone/nohup.out"
REMOTE_OUT_DIR = "/root/sba_standalone/results/gemma4_e4b_sba_distill_bridge_b200"
LOCAL_OUT_DIR = "e:/code.projects/CASCADES--continual-PEFT-for-Local-LLMs/sba_standalone/results/gemma4_e4b_sba_distill_bridge_b200"

def check_log():
    cmd = f'ssh -p {PORT} -i {KEY_PATH} -o StrictHostKeyChecking=no {REMOTE_HOST} "tail -n 10 {REMOTE_LOG}"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
        if result.stdout is not None:
            return result.stdout.decode('utf-8', errors='replace')
        return ""
    except Exception as e:
        print(f"Error checking log: {e}")
        return ""

def download_data():
    print(f"[{datetime.datetime.now()}] Downloading data from B200...")
    os.makedirs(LOCAL_OUT_DIR, exist_ok=True)
    # Using scp to recursively download the checkpoints
    cmd = f'scp -r -P {PORT} -i {KEY_PATH} -o StrictHostKeyChecking=no {REMOTE_HOST}:{REMOTE_OUT_DIR}/* {LOCAL_OUT_DIR}/'
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)
    print(f"[{datetime.datetime.now()}] Download complete!")

def main():
    print(f"[{datetime.datetime.now()}] Starting B200 training monitor...")
    while True:
        log_out = check_log()
        
        if "Training Completed Successfully" in log_out or "5000/5000" in log_out:
            print(f"[{datetime.datetime.now()}] Training finished detected. Waiting 30s to let file writes flush...")
            time.sleep(30)
            download_data()
            break
            
        # Check if the process died unexpectedly
        if "CUDA out of memory" in log_out or "Traceback" in log_out:
            print(f"[{datetime.datetime.now()}] ERROR DETECTED in training! Downloading whatever exists...")
            download_data()
            break
            
        # Sleep for 2 minutes before checking again
        time.sleep(120)

if __name__ == "__main__":
    main()
