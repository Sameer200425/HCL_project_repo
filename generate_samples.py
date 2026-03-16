"""Generate synthetic sample images for testing K-fold and ensemble."""
from PIL import Image, ImageDraw
import random
import os

classes = ["genuine", "fraud", "tampered", "forged"]
colors = {
    "genuine": (240, 240, 230),
    "fraud": (255, 200, 200),
    "tampered": (200, 200, 255),
    "forged": (255, 255, 200),
}

for cls in classes:
    d = f"data/raw_images/{cls}"
    os.makedirs(d, exist_ok=True)
    for i in range(25):
        base = colors[cls]
        c = tuple(max(0, min(255, v + random.randint(-30, 30))) for v in base)
        img = Image.new("RGB", (224, 224), c)
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(3, 10)):
            x1, y1 = random.randint(0, 200), random.randint(0, 200)
            x2, y2 = x1 + random.randint(10, 50), y1 + random.randint(10, 50)
            rc = tuple(random.randint(0, 255) for _ in range(3))
            draw.rectangle([x1, y1, x2, y2], fill=rc)
        draw.text((10, 10), f"{cls}_{i}", fill=(0, 0, 0))
        img.save(f"{d}/{cls}_{i:03d}.png")
    print(f"  {cls}: 25 images created")

print("Done: 100 total synthetic images")
