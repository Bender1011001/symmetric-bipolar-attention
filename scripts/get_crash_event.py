import subprocess
import json

def main():
    # Use powershell to query event log, formatting as json to avoid parsing issues
    cmd = [
        "powershell",
        "-Command",
        "Get-EventLog -LogName Application -EntryType Error -Newest 50 | Where-Object { $_.Message -like '*python*' } | Select-Object TimeGenerated, Source, EventID, Message | ConvertTo-Json"
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:")
        print(res.stdout)
    except Exception as e:
        print("Error running command:", e)

if __name__ == "__main__":
    main()
