import os
import time
import requests
import shutil

API_URL = "http://127.0.0.1:8000/api/predict/single"
WATCH_DIR = "incoming_bank_scans"

def start_realtime_scanner():
    """Simulates a real-time banking pipeline (like an ATM scanner or bank teller scanner)."""
    if not os.path.exists(WATCH_DIR):
        os.makedirs(WATCH_DIR)

    processed_files = set()

    print("=====================================================")
    print(f"📡 REAL-TIME BANK SCANNER ACTIVE")
    print(f"📂 Monitoring folder: ./{WATCH_DIR}")
    print("👉 To test: Copy and paste any image into the 'incoming_bank_scans' folder!")
    print("=====================================================\n")

    while True:
        try:
            current_files = set(os.listdir(WATCH_DIR))
            new_files = current_files - processed_files
            
            for file in new_files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(WATCH_DIR, file)
                    print(f"🚨 [NEW SCAN DETECTED] Automatically processing '{file}'...")
                    
                    try:
                        with open(filepath, "rb") as f:
                            start_t = time.time()
                            response = requests.post(API_URL, files={"file": f})
                            latency = (time.time() - start_t) * 1000
                            
                        if response.status_code == 200:
                            data = response.json()
                            risk = data.get("risk_level", "UNKNOWN")
                            cls = data.get("predicted_class", "unknown").upper()
                            
                            if risk in ["CRITICAL", "HIGH"]:
                                print(f"   ❌ FRAUD ALERT DETECTED ({latency:.1f}ms)!")
                                print(f"      Type: {cls}")
                                print(f"      AI Logic: {data.get('explanation')}")
                                print("   -------------------------------------------------")
                            else:
                                print(f"   ✅ SECURE / GENUINE ({latency:.1f}ms). Transaction Approved.")
                                print("   -------------------------------------------------")
                        else:
                            print(f"   ⚠️ REJECTED BY AI FILTER (Blur/Glare/Error):")
                            print(f"      {response.json().get('detail')}")
                            print("   -------------------------------------------------")
                    except Exception as e:
                        print(f"   ⚠️ Connection Error: Is the API running? ({e})")
                        
                    processed_files.add(file)
            
            time.sleep(1)  # Real-time polling every 1 second
            
        except KeyboardInterrupt:
            print("\n🛑 Scanner stopped.")
            break

if __name__ == "__main__":
    start_realtime_scanner()
