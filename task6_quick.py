import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'raw_images'
RESULTS_DIR = BASE_DIR / 'results'

analysis = {'total_images': 0, 'class_distribution': {}}
for cls_dir in DATA_DIR.iterdir():
    if cls_dir.is_dir():
        count = len(list(cls_dir.glob('*')))
        analysis['class_distribution'][cls_dir.name] = count
        analysis['total_images'] += count

with open(RESULTS_DIR / 'data_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

guide = """# Real Data Integration Guide

## Directory Structure
Place images in:
- data/real_images/genuine/
- data/real_images/fraud/
- data/real_images/tampered/
- data/real_images/forged/

## Usage
Load model and run inference on real images.
"""

with open(BASE_DIR / 'REAL_DATA_GUIDE.md', 'w', encoding='utf-8') as f:
    f.write(guide)

print(f"Task 6 Complete: {analysis['total_images']} images")
print(f"Distribution: {analysis['class_distribution']}")
print("[TASK 6 COMPLETE]")
