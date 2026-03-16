import cv2
import numpy as np
import os

def create_statement(filename, tampered=False):
    # A standard 8.5 x 11 piece of paper proportions
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Header
    cv2.putText(img, "GLOBAL SECURE BANK - ACCOUNT STATEMENT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Account: 1234-5678-9012", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.line(img, (30, 110), (570, 110), (0, 0, 0), 2)
    
    cv2.putText(img, "Date       | Description                | Amount", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.line(img, (30, 150), (570, 150), (0, 0, 0), 1)
    
    # Normal Transactions
    transactions = [
        ("03/01/2026", "Grocery Store", "-$125.40", False),
        ("03/05/2026", "Salary Deposit", "+$3200.00", False),
        ("03/10/2026", "Rent Transfer", "-$1500.00", False),
    ]
    
    y = 180
    for date, desc, amt, _ in transactions:
        cv2.putText(img, f"{date} | {desc.ljust(25)} | {amt}", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        y += 40
        
    if tampered:
        # Simulate a fraudster changing a bank statement to hide stolen money
        # Look closely: The font is slightly different, it's misaligned, and darker ink
        cv2.putText(img, f"03/12/2026 | UNKNOWN OFFSHORE WIRE   | -$9500.00", (28, y+2), cv2.FONT_HERSHEY_COMPLEX, 0.55, (20,20,20), 2)
        # Add a digital smudge/blur indicating photoshop tapering
        cv2.circle(img, (480, y), 15, (200, 200, 200), -1)
    else:
        cv2.putText(img, f"03/12/2026 | Utility Bill               | -$85.00", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    os.makedirs("demo_statements", exist_ok=True)
    cv2.imwrite(f"demo_statements/{filename}", img)
    print(f"📄 Generated Statement: demo_statements/{filename}")

if __name__ == "__main__":
    print("🖨️  Generating fake banking statements...")
    create_statement("01_genuine_statement.jpg", tampered=False)
    create_statement("02_tampered_statement.jpg", tampered=True)
    print("✅ Done!")
