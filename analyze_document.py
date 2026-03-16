import requests
import sys
import time
import os

API_URL = "http://127.0.0.1:8000/api/predict/single"

def analyze_real_document(image_path):
    """Sends a real document (cheque, bank statement, signature) to the live AI for analysis."""
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found at '{image_path}'")
        return

    print(f"🔍 Analyzing document: {image_path}")
    print("⏳ Sending to Bank ViT API...\n")

    try:
        with open(image_path, "rb") as f:
            # The API expects a file upload
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            
            start_time = time.time()
            response = requests.post(API_URL, files=files)
            latency = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Analysis Complete! (Speed: {latency*1000:.1f}ms) [ONNX Accelerated]")
                print("="*50)
                print(f"📄 Predicted Class: {data.get('predicted_class', 'Unknown').upper()}")
                print(f"⚠️ Risk Level:      {data.get('risk_level', 'Unknown')}")
                print(f"🎯 Confidence:      {data.get('confidence', 0)*100:.1f}%")
                
                print("\n📊 Probabilities:")
                for category, prob in data.get('probabilities', {}).items():
                    print(f"  - {category.capitalize().ljust(10)}: {prob*100:.1f}%")
                
                print("\n💡 AI Explanation:")
                print(f"  {data.get('explanation', 'No explanation provided.')}")
                print("="*50)
            else:
                print(f"⚠️ Quality Check Failed or API Error ({response.status_code}):")
                print(response.json())

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is FastAPI running on port 8000?")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_document.py <path_to_your_image>")
        print("Example: python analyze_document.py my_bank_statement.jpg")
    else:
        analyze_real_document(sys.argv[1])
