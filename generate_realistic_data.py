"""
Realistic Bank Cheque Image Generator
======================================
Generates 600+ high-quality synthetic bank cheque images across 4 classes:
  - genuine  : Clean, properly formatted cheques
  - fraud    : Subtle colour shifts, wrong fonts, blurry details
  - tampered : Visible overwrites on amount/payee fields
  - forged   : Completely fabricated with inconsistencies

Each image is 224×224 RGB, suitable for ViT/CNN training.

Usage:
    python generate_realistic_data.py                  # 600 images (default)
    python generate_realistic_data.py --count 1000     # 1000 images
    python generate_realistic_data.py --count 200 --clean  # wipe old + 200 new
"""

import os
import sys
import math
import random
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# Compatibility for different Pillow versions
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS  # type: ignore
    RESAMPLE_BILINEAR = Image.BILINEAR  # type: ignore

# ============================================================
#  Constants
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "raw_images"
CLASSES = ["genuine", "fraud", "tampered", "forged"]
IMAGE_SIZE = (224, 224)

# Bank names pool
BANK_NAMES = [
    "Federal Reserve Bank", "Golden State Bank", "Metropolitan Trust",
    "Summit Banking", "Pacific National Bank", "Heritage Financial",
    "Continental Bank", "Prestige Trust", "Atlantic Savings Bank",
    "Pinnacle Bank Corp", "First Citizens Bank", "Northern Trust",
    "Unity Federal Bank", "Quantum Financial", "Diamond Credit Union",
    "Central Valley Bank", "Lakeside Trust", "Eagle National Bank",
    "Sunrise Savings", "Westfield Banking", "Oakwood Financial",
    "Cascade Bank Corp", "Horizon Trust", "Redwood Credit Union",
]

PAYEE_NAMES = [
    "Red Mountain Mining", "Silver Creek LLC", "Blue Ocean Ventures",
    "Green Valley Farms", "Golden Ridge Corp", "White Stone Holdings",
    "Black Pearl Trading", "Iron Bridge Co", "Crystal Lake Properties",
    "Amber Wave Industries", "Maple Leaf Logistics", "Pine Creek Services",
    "Sarah Mitchell", "James Roberts", "Emily Chen", "Michael Brown",
    "David Wilson", "Jennifer Adams", "Robert Taylor", "Lisa Martinez",
    "Jane Doe", "John Smith", "Alice Johnson", "Bob Williams",
]

MEMO_TEXTS = [
    "Payment", "Rent", "Invoice #183", "Services Rendered",
    "Consulting Fee", "Monthly Lease", "Equipment Purchase",
    "Office Supplies", "Deposit", "Repair Work", "Subscription",
    "Annual Dues", "Maintenance", "Salary Advance", "Project Fee",
]

AMOUNT_WORDS = {
    500: "Five Hundred", 1000: "One Thousand", 1250: "One Thousand Two Hundred Fifty",
    2500: "Two Thousand Five Hundred", 3000: "Three Thousand",
    5000: "Five Thousand", 7500: "Seven Thousand Five Hundred",
    10000: "Ten Thousand", 15000: "Fifteen Thousand", 15600: "Fifteen Thousand Six Hundred",
    20000: "Twenty Thousand", 25000: "Twenty Five Thousand",
    50000: "Fifty Thousand", 75000: "Seventy Five Thousand",
}


def get_font(size: int = 12, bold: bool = False):
    """Try to load a system font; fall back to default."""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def random_amount():
    """Generate a random dollar amount."""
    amounts = list(AMOUNT_WORDS.keys())
    return random.choice(amounts)


def random_date():
    """Generate a random date string."""
    base = datetime(2025, 1, 1)
    offset = random.randint(0, 365)
    d = base + timedelta(days=offset)
    return d.strftime("%m/%d/%Y")


def draw_signature(draw, x: int, y: int, style: int = 0):  # type: (ImageDraw.ImageDraw, int, int, int) -> None
    """Draw a random squiggly signature."""
    pen_color = (0, 0, random.randint(100, 180))  # dark blue ink
    width = 2
    
    # Generate random control points for signature curves
    points = [(x, y)]
    cx, cy = x, y
    num_strokes = random.randint(4, 8)
    for _ in range(num_strokes):
        cx += random.randint(5, 20)
        cy += random.randint(-15, 15)
        points.append((cx, cy))
    
    # Draw the connected lines
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=pen_color, width=width)
    
    # Add loops/curves for realism
    for _ in range(random.randint(1, 3)):
        lx = random.randint(x, x + 40)
        ly = random.randint(y - 10, y + 10)
        draw.arc([lx, ly, lx + 15, ly + 15], 0, 360, fill=pen_color, width=width)


def generate_base_cheque(bank_name: str, check_number: str, date_str: str,
                          payee: str, amount: int, memo: str,
                          bg_color: tuple = (245, 245, 235)):
    """Generate a clean base cheque image."""
    img = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Cheque border
    draw.rectangle([5, 5, 218, 218], outline=(180, 180, 200), width=2)
    
    # Bank name (top-left, italic-style)
    bank_font = get_font(14, bold=True)
    draw.text((12, 10), bank_name, fill=(0, 0, 140), font=bank_font)
    
    # Check number (top-right)
    small_font = get_font(9)
    draw.text((175, 10), f"#{check_number}", fill=(100, 100, 100), font=small_font)
    
    # Date
    date_font = get_font(10)
    draw.text((130, 30), f"Date: {date_str}", fill=(0, 0, 0), font=date_font)
    
    # "Pay to the order of" label
    label_font = get_font(8)
    draw.text((12, 48), "Pay to the order of:", fill=(100, 100, 100), font=label_font)
    
    # Payee name
    payee_font = get_font(11, bold=True)
    draw.text((12, 60), payee, fill=(0, 0, 0), font=payee_font)
    
    # Amount box (top-right area)
    draw.rectangle([155, 48, 212, 68], outline=(100, 100, 100), width=1)
    amount_str = f"${amount:,.2f}"
    amount_font = get_font(10, bold=True)
    draw.text((158, 52), amount_str, fill=(0, 0, 0), font=amount_font)
    
    # Written amount line
    amount_word = AMOUNT_WORDS.get(amount, "Amount")
    word_font = get_font(10, bold=True)
    draw.text((12, 80), amount_word, fill=(0, 0, 0), font=word_font)
    draw.text((12 + len(amount_word) * 6 + 10, 80), "DOLLARS", fill=(80, 80, 80), font=get_font(9))
    
    # Horizontal line under amount
    draw.line([(12, 92), (210, 92)], fill=(150, 150, 150), width=1)
    
    # Memo line
    memo_font = get_font(8)
    draw.text((12, 100), "Memo:", fill=(100, 100, 100), font=memo_font)
    draw.text((45, 100), memo, fill=(0, 0, 0), font=get_font(9))
    
    # Signature line
    draw.line([(100, 130), (210, 130)], fill=(150, 150, 150), width=1)
    
    # Draw signature
    draw_signature(draw, 120, 115, style=random.randint(0, 3))
    
    # MICR line (bottom)
    micr_font = get_font(8)
    routing = f":{random.randint(10000000, 99999999)}:"
    account = f"{random.randint(10000000, 99999999)}:"
    check_digits = f"{random.randint(1000, 9999)}"
    draw.text((12, 145), f"{routing} {account} {check_digits}", 
              fill=(50, 50, 50), font=micr_font)
    
    return img


# ============================================================
#  Class-Specific Generators
# ============================================================

def generate_genuine() -> Image.Image:
    """Generate a clean, authentic-looking bank cheque."""
    bg_colors = [
        (245, 245, 235), (235, 245, 250), (250, 248, 230),
        (240, 240, 245), (248, 245, 238), (238, 245, 240),
    ]
    img = generate_base_cheque(
        bank_name=random.choice(BANK_NAMES),
        check_number=str(random.randint(1000, 9999)),
        date_str=random_date(),
        payee=random.choice(PAYEE_NAMES),
        amount=random_amount(),
        memo=random.choice(MEMO_TEXTS),
        bg_color=random.choice(bg_colors),
    )
    # Slight natural variation
    if random.random() < 0.3:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))
    return img


def generate_fraud() -> Image.Image:
    """Generate a fraudulent cheque — subtle anomalies."""
    img = generate_base_cheque(
        bank_name=random.choice(BANK_NAMES),
        check_number=str(random.randint(1000, 9999)),
        date_str=random_date(),
        payee=random.choice(PAYEE_NAMES),
        amount=random_amount(),
        memo=random.choice(MEMO_TEXTS),
        bg_color=(random.randint(240, 255), random.randint(200, 220),
                  random.randint(200, 220)),  # slightly pinkish
    )
    draw = ImageDraw.Draw(img)
    
    # Fraud indicators: wrong font sizes, color mismatches, blurriness
    effects = random.sample(["blur", "color_shift", "noise", "ghost_text", "double_sig"], 
                           k=random.randint(1, 3))
    
    if "blur" in effects:
        # Selective blur on amount area
        region = img.crop((150, 45, 215, 70))
        region = region.filter(ImageFilter.GaussianBlur(radius=1.5))
        img.paste(region, (150, 45))
    
    if "color_shift" in effects:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.7, 0.85))
    
    if "noise" in effects:
        # Add speckle noise
        pixels = img.load()
        if pixels is not None:
            for _ in range(random.randint(50, 200)):
                nx, ny = random.randint(0, 223), random.randint(0, 223)
                pixels[nx, ny] = (
                    random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)
                )
    
    if "ghost_text" in effects:
        draw = ImageDraw.Draw(img)
        ghost_font = get_font(8)
        draw.text((random.randint(40, 100), random.randint(50, 80)),
                  random.choice(["VOID", "COPY", "SAMPLE"]),
                  fill=(200, 200, 200), font=ghost_font)
    
    if "double_sig" in effects:
        draw = ImageDraw.Draw(img)
        draw_signature(draw, 125, 118, style=1)  # second overlapping signature
    
    return img


def generate_tampered() -> Image.Image:
    """Generate a tampered cheque — visible overwrites on key fields."""
    # Start with a genuine base
    original_amount = random_amount()
    original_payee = random.choice(PAYEE_NAMES)
    
    img = generate_base_cheque(
        bank_name=random.choice(BANK_NAMES),
        check_number=str(random.randint(1000, 9999)),
        date_str=random_date(),
        payee=original_payee,
        amount=original_amount,
        memo=random.choice(MEMO_TEXTS),
        bg_color=(random.randint(230, 250), random.randint(230, 250),
                  random.randint(240, 255)),  # slightly bluish
    )
    draw = ImageDraw.Draw(img)
    
    # Tamper effects: white-out + rewrite on amount or payee
    tamper_targets = random.sample(["amount", "payee", "date"], 
                                    k=random.randint(1, 2))
    
    if "amount" in tamper_targets:
        # White-out the amount box and rewrite with different amount
        draw.rectangle([155, 48, 212, 68], fill=(255, 255, 250), outline=(100, 100, 100))
        new_amount = random_amount()
        while new_amount == original_amount:
            new_amount = random_amount()
        draw.text((158, 52), f"${new_amount:,.2f}", fill=(0, 0, 0), font=get_font(10, True))
    
    if "payee" in tamper_targets:
        # Semi-transparent overlay on payee name
        overlay_color = (random.randint(240, 255), random.randint(240, 255), random.randint(230, 250))
        draw.rectangle([10, 58, 150, 75], fill=overlay_color)
        new_payee = random.choice(PAYEE_NAMES)
        while new_payee == original_payee:
            new_payee = random.choice(PAYEE_NAMES)
        draw.text((12, 60), new_payee, fill=(0, 0, random.randint(0, 40)), font=get_font(11, True))
    
    if "date" in tamper_targets:
        draw.rectangle([128, 28, 210, 42], fill=(248, 248, 240))
        draw.text((130, 30), f"Date: {random_date()}", fill=(0, 0, 0), font=get_font(10))
    
    # Add tamper artifacts: slight smudging, color inconsistency
    if random.random() < 0.5:
        region = img.crop((10, 55, 160, 78))
        region = region.filter(ImageFilter.GaussianBlur(radius=0.8))
        img.paste(region, (10, 55))
    
    return img


def generate_forged() -> Image.Image:
    """Generate a completely forged cheque with inconsistencies."""
    # Wrong proportions, misaligned elements, made-up bank names
    fake_banks = [
        "Natioal Bank", "Fedral Reserve", "Goldon Trust",  # intentional misspellings
        "ABC BANK", "XXXX Financial", "Test Credit Union",
    ]
    
    bg_color = (random.randint(230, 255), random.randint(230, 255),
                random.randint(200, 230))  # yellowish
    
    img = Image.new("RGB", IMAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Irregular border
    draw.rectangle([3, 3, 220, 220], outline=(150, 150, 170), width=random.choice([1, 2, 3]))
    
    # Bank name (possibly misspelled or wrong style)
    bank = random.choice(fake_banks + BANK_NAMES[:5])
    bank_font = get_font(random.randint(11, 16), bold=True)
    draw.text((random.randint(8, 15), random.randint(8, 15)), bank, 
              fill=(0, 0, random.randint(100, 180)), font=bank_font)
    
    # Check number (wrong position sometimes)
    draw.text((random.randint(160, 200), random.randint(8, 20)),
              f"#{random.randint(100, 99999)}", fill=(100, 100, 100), font=get_font(9))
    
    # Date (inconsistent format)
    date_formats = [
        random_date(),
        f"{random.randint(1,12)}.{random.randint(1,28)}.{random.randint(2024,2026)}",
        f"{random.randint(2024,2026)}/{random.randint(1,12)}/{random.randint(1,28)}",
    ]
    draw.text((120, random.randint(28, 40)), f"Date: {random.choice(date_formats)}", 
              fill=(0, 0, 0), font=get_font(10))
    
    # Payee
    draw.text((12, 48), "Pay to the order of:", fill=(100, 100, 100), font=get_font(8))
    draw.text((12, 62), random.choice(PAYEE_NAMES), fill=(0, 0, 0), font=get_font(11, True))
    
    # Amount with misalignment
    amount = random_amount()
    draw.rectangle([random.randint(150, 160), 50, 215, 70], 
                   outline=(100, 100, 100), width=1)
    draw.text((random.randint(155, 165), 53), f"${amount:,.2f}", 
              fill=(0, 0, 0), font=get_font(10, True))
    
    # Written amount (might not match)
    wrong_amount = random_amount()
    word = AMOUNT_WORDS.get(wrong_amount, "Amount")
    draw.text((12, 82), word, fill=(0, 0, 0), font=get_font(10, True))
    draw.text((len(word) * 6 + 20, 82), "DOLLARS", fill=(80, 80, 80), font=get_font(9))
    draw.line([(12, 94), (210, 94)], fill=(150, 150, 150), width=1)
    
    # Memo
    draw.text((12, 102), "Memo:", fill=(100, 100, 100), font=get_font(8))
    draw.text((45, 102), random.choice(MEMO_TEXTS), fill=(0, 0, 0), font=get_font(9))
    
    # Bad signature (too clean or too messy)
    draw.line([(100, 132), (210, 132)], fill=(150, 150, 150), width=1)
    if random.random() < 0.5:
        # Overly simple signature
        draw.line([(120, 125), (180, 118)], fill=(0, 0, 140), width=2)
        draw.line([(180, 118), (195, 128)], fill=(0, 0, 140), width=2)
    else:
        draw_signature(draw, 120, 115)
        draw_signature(draw, 122, 117)  # doubled = forged attempt
    
    # MICR (might be missing or garbled)
    if random.random() < 0.7:
        draw.text((12, 147), 
                  f":{random.randint(10000, 99999)}:  {random.randint(10000, 99999)}: {random.randint(100, 999)}", 
                  fill=(50, 50, 50), font=get_font(8))
    
    # Forged artifacts: overall quality issues
    effects = random.sample(["jpeg_artifact", "low_res", "uneven_color"], 
                           k=random.randint(1, 2))
    
    if "jpeg_artifact" in effects:
        import io
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=random.randint(15, 40))
        buffer.seek(0)
        img = Image.open(buffer).convert("RGB")
        img = img.resize(IMAGE_SIZE, RESAMPLE_LANCZOS)
    
    if "low_res" in effects:
        small = img.resize((random.randint(80, 140), random.randint(80, 140)), RESAMPLE_BILINEAR)
        img = small.resize(IMAGE_SIZE, RESAMPLE_BILINEAR)
    
    if "uneven_color" in effects:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.6, 0.8))
    
    return img


# ============================================================
#  Main Generator
# ============================================================
GENERATORS = {
    "genuine": generate_genuine,
    "fraud": generate_fraud,
    "tampered": generate_tampered,
    "forged": generate_forged,
}


def generate_dataset(total_per_class: int = 150, clean: bool = False):
    """Generate the full dataset."""
    print("=" * 60)
    print("  Realistic Bank Cheque Dataset Generator")
    print("=" * 60)
    print(f"  Classes: {CLASSES}")
    print(f"  Images per class: {total_per_class}")
    print(f"  Total images: {total_per_class * len(CLASSES)}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Output: {DATA_DIR}")
    print("=" * 60)

    for cls in CLASSES:
        cls_dir = DATA_DIR / cls
        if clean and cls_dir.exists():
            import shutil
            shutil.rmtree(cls_dir)
            print(f"  Cleaned {cls}/")
        cls_dir.mkdir(parents=True, exist_ok=True)

        gen_func = GENERATORS[cls]
        count = 0
        for i in range(total_per_class):
            try:
                img = gen_func()
                img = img.resize(IMAGE_SIZE, RESAMPLE_LANCZOS)
                img.save(cls_dir / f"{cls}_{i:03d}.png")
                count += 1
            except Exception as e:
                print(f"  ⚠ {cls}_{i}: {e}")
        
        print(f"  ✅ {cls}: {count} images generated")

    total = total_per_class * len(CLASSES)
    print(f"\n{'=' * 60}")
    print(f"  DONE: {total} total images in {DATA_DIR}")
    print(f"{'=' * 60}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Generate realistic bank cheque dataset")
    parser.add_argument("--count", type=int, default=150,
                        help="Number of images per class (default: 150, total=600)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove existing images before generating")
    args = parser.parse_args()
    
    random.seed(42)  # Reproducible
    generate_dataset(total_per_class=args.count, clean=args.clean)


if __name__ == "__main__":
    main()
