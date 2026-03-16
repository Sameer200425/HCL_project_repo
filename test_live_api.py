import requests
import cv2
import numpy as np
import io
import time

API_URL = "http://127.0.0.1:8000/api/predict/single"

def create_dummy_image(text, is_blurry=False, is_glare=False):
    """Creates a dummy image for testing."""
    # Create a light gray background image representing a document
    img = np.ones((400, 800, 3), dtype=np.uint8) * 180
    
    # Add some text
    cv2.putText(img, text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "Sign here: [X] __________________", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add dense mock text to simulate a real document and pass the blur filter (which needs high frequency details)
    for i in range(10, 400, 20):
        cv2.line(img, (20, i), (780, i), (100, 100, 100), 1)
    
    if is_blurry:
        # Apply intense blur
        img = cv2.GaussianBlur(img, (45, 45), 0)
        
    if is_glare:
        # Add a massive harsh white spot simulating a camera flash
        cv2.circle(img, (400, 200), 100, (255, 255, 255), -1)
        # Increase overall brightness
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=50)

    # Encode to PNG
    _, buffer = cv2.imencode('.png', img)
    return io.BytesIO(buffer.tobytes())

def test_api():
    print("🚀 Testing Bank ViT API...")
    
    # Test 1: Normal Image
    print("\n--- Test 1: Standard Image ---")
    img_bytes = create_dummy_image("Valid Cheque #12345")
    files = {"file": ("test_normal.png", img_bytes, "image/png")}
    
    start_time = time.time()
    response = requests.post(API_URL, files=files)
    latency = time.time() - start_time
    
    if response.status_code == 200:
        print(f"✅ Success! (Latency: {latency*1000:.1f}ms) [ONNX Optimized]")
        print("Result:", response.json())
    else:
        print(f"⚠️ API Error ({response.status_code}):", response.text)

    # Test 2: Blurry Image (Should trigger our new Image Quality check)
    print("\n--- Test 2: Blurry Image ---")
    img_bytes = create_dummy_image("Blurry Cheque", is_blurry=True)
    files = {"file": ("test_blur.png", img_bytes, "image/png")}
    
    response = requests.post(API_URL, files=files)
    if response.status_code != 200:
        print("✅ Expected Rejection Caught!")
        print("Result:", response.json())
    else:
        print("Result:", response.json())

if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is FastAPI running on port 8000?")
