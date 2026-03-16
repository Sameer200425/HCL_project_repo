import cv2
import numpy as np
import os
from pathlib import Path

def create_bank_cheque(filename, is_fraud=False, is_tampered=False):
    # Create base cheque background (light blue/gray tint)
    img = np.ones((400, 900, 3), dtype=np.uint8) * 240
    img[:, :, 0] = 250 # More blue
    
    # Add border
    cv2.rectangle(img, (10, 10), (890, 390), (100, 100, 100), 2)
    
    # Add Bank Name
    cv2.putText(img, "GLOBAL SECURE BANK", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (50, 50, 100), 2)
    cv2.putText(img, "Date: 2026-03-16", (650, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Payee line
    cv2.putText(img, "PAY TO THE", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "ORDER OF:", (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.line(img, (140, 175), (650, 175), (0, 0, 0), 1)
    
    payee_name = "John Doe"
    cv2.putText(img, payee_name, (160, 170), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.2, (20, 20, 20), 2)
    
    # Amount Box
    cv2.rectangle(img, (680, 140), (870, 185), (0, 0, 0), 1)
    amount_text = "$ 1,500.00"
    
    if is_tampered:
        # Simulate tampering - added a zero with a different thickness/alignment
        amount_text = "$ 15,000.00"
        cv2.putText(img, "$ 15,00", (690, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(img, "0.00", (810, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 3) # tampered part
        
        # Smudge mark over amount
        cv2.circle(img, (800, 160), 20, (200, 200, 200), -1)
    else:
        cv2.putText(img, amount_text, (690, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
    # Amount in words
    cv2.line(img, (30, 240), (870, 240), (0, 0, 0), 1)
    words = "Fifteen Thousand Dollars" if is_tampered else "One Thousand Five Hundred Dollars"
    cv2.putText(img, words, (50, 235), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.8, (0, 0, 0), 1)

    # Signature line
    cv2.line(img, (550, 340), (870, 340), (0, 0, 0), 1)
    cv2.putText(img, "AUTHORIZED SIGNATURE", (600, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Draw Signature
    if is_fraud:
        # Scribbly, jagged, erratic signature (fraud/forged)
        pts = np.array([[570, 320], [600, 290], [620, 330], [650, 280], [680, 350], [700, 310], [750, 330], [770, 300], [800, 320], [850, 290]], np.int32)
        cv2.polylines(img, [pts], False, (0, 0, 150), 3) # Drawn with red pen, jagged
    else:
        # Smooth, natural signature (genuine)
        pts = np.array([[560, 320], [580, 290], [590, 310], [630, 310], [650, 280], [680, 310], [720, 290], [760, 315], [820, 300]], np.int32)
        from scipy.interpolate import make_interp_spline
        x = pts[:,0]
        y = pts[:,1]
        x_new = np.linspace(x.min(), x.max(), 300)
        spl = make_interp_spline(x, y, k=3)
        y_new = spl(x_new)
        smooth_pts = np.column_stack((x_new, y_new)).astype(np.int32)
        cv2.polylines(img, [smooth_pts], False, (0, 50, 150), 2) # Smooth dark blue pen

    # Add MICR code at bottom
    micr = "A123456789A  123456789  0123"
    cv2.putText(img, micr, (100, 360), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

    os.makedirs("demo_cheques", exist_ok=True)
    filepath = os.path.join("demo_cheques", filename)
    cv2.imwrite(filepath, img)
    print(f"✅ Created: {filepath}")

if __name__ == "__main__":
    print("🎨 Generating High-Res Demo Cheques for Testing...")
    create_bank_cheque("01_genuine_cheque.jpg", is_fraud=False, is_tampered=False)
    create_bank_cheque("02_forged_signature_cheque.jpg", is_fraud=True, is_tampered=False)
    create_bank_cheque("03_tampered_amount_cheque.jpg", is_fraud=False, is_tampered=True)
    print("🎉 Done! Check the 'demo_cheques' folder.")
