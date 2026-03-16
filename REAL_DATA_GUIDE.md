# Real Data Integration Guide

This project uses **real data only** — no synthetic/demo images.  
Follow these steps to get everything working.

---

## 1. Directory Structure

```
data/
├── raw_images/            ← YOUR bank documents go here
│   ├── genuine/           ← Authentic cheques / statements
│   ├── fraud/             ← Known fraudulent documents
│   ├── tampered/          ← Documents with visible alterations
│   └── forged/            ← Completely fake documents
├── processed/             ← Auto-populated by setup_datasets.py --prepare
├── uploads/               ← Real-time uploads from API / frontend
├── cedar_signatures/      ← CEDAR signature dataset
│   ├── genuine/
│   └── forged/
├── rvl_cdip/              ← RVL-CDIP document images (optional)
└── transactions/
    └── creditcard.csv     ← Kaggle credit-card fraud CSV
```

---

## 2. Adding Your Own Bank Documents

Place scanned/photographed images into:
- `data/raw_images/genuine/`  — real, authentic documents
- `data/raw_images/fraud/`    — known fraudulent documents
- `data/raw_images/tampered/` — documents with alterations (white-out, edits)
- `data/raw_images/forged/`   — completely fake documents

Supported formats: PNG, JPG, JPEG, TIF, TIFF, BMP

**Minimum recommendation:** 50+ images per class (200+ total).  
For production accuracy: 500+ per class.

---

## 3. External Datasets

### CEDAR Signatures (recommended)
```bash
python setup_datasets.py --cedar
```
Follow the printed instructions to download from https://www.cedar.buffalo.edu/NIJ/data/

### Kaggle Credit Card Fraud
```bash
python setup_datasets.py --creditcard
```
Or manually download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### RVL-CDIP (optional, for ViT pretraining)
```bash
python setup_datasets.py --rvl-cdip
```
Downloads ~40GB from HuggingFace. Only needed for self-supervised pretraining.

---

## 4. Prepare & Train

```bash
# Check what data is available
python setup_datasets.py --check

# Copy raw_images → processed/ (validates images)
python setup_datasets.py --prepare

# Train the ViT fraud detector
python run_pipeline.py

# Train sub-models (signature + risk scorer)
python data_integration/unified_loader.py --train-all
```

---

## 5. Real-Time Upload & Detection

### Via API
```bash
# Start the backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Upload and detect (saves to data/uploads/)
curl -X POST http://localhost:8000/api/upload/detect \
  -F "file=@my_cheque.jpg" \
  -F "model=vit"

# Add a labelled image to the training set
curl -X POST "http://localhost:8000/api/upload/add-to-dataset?label=genuine" \
  -F "file=@my_cheque.jpg"

# List pending uploads
curl http://localhost:8000/api/upload/pending

# Label a pending upload (moves it to training set)
curl -X POST "http://localhost:8000/api/upload/label-pending?filename=FILE.jpg&label=fraud"
```

### Via Frontend
Upload documents through the Next.js UI at http://localhost:3000.  
The frontend calls the same API endpoints above.

---

## 6. Re-training After Adding Data

After adding new labelled images via the API or manually:

```bash
python setup_datasets.py --prepare   # re-validate & copy
python run_pipeline.py               # retrain model
```

The system works with **both** your existing training dataset **and** newly uploaded documents simultaneously.

---

## 7. Summary

| What | Where |
|------|-------|
| Your bank documents | `data/raw_images/{class}/` |
| Real-time uploads | `data/uploads/` → label → `data/raw_images/{class}/` |
| CEDAR signatures | `data/cedar_signatures/` |
| Credit card CSV | `data/transactions/creditcard.csv` |
| RVL-CDIP (optional) | `data/rvl_cdip/` |
| Prepared for training | `data/processed/` |
