# 📊 Recommended Datasets for Bank Fraud Detection with ViT

This guide provides curated dataset recommendations to enhance your Vision Transformer-based banking fraud detection system.

---

## 🏦 1️⃣ Financial Document Image Datasets

These are ideal for ViT-based document understanding and self-supervised pretraining.

### 📄 1. RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing)

| Property | Details |
|----------|---------|
| **Type** | Document Classification |
| **Size** | 400,000+ grayscale document images |
| **Classes** | 16 (Letters, Forms, Memos, Emails, Invoices, etc.) |
| **Download** | [HuggingFace](https://huggingface.co/datasets/rvl_cdip) |
| **Paper** | Harley et al., ICDAR 2015 |

✅ **Why useful for your project?**
- Perfect for **self-supervised pretraining** (MAE)
- Large dataset → Excellent for Vision Transformer training
- Document structure learning transfers to cheque analysis
- Can simulate financial document classification

**Integration path:**
```
data/
├── raw_images/
│   └── rvl_cdip/
│       ├── train/
│       ├── val/
│       └── test/
```

**Code to load:**
```python
from datasets import load_dataset
dataset = load_dataset("rvl_cdip")
```

---

### 📑 2. FUNSD (Form Understanding in Noisy Scanned Documents)

| Property | Details |
|----------|---------|
| **Type** | Scanned form dataset |
| **Size** | 199 annotated forms |
| **Use Case** | Document understanding & layout detection |
| **Download** | [GitHub](https://guillaumejaume.github.io/FUNSD/) |

✅ **Useful for:**
- Learning document **structure and layout**
- Fraud detection in **structured forms** (like cheques)
- **Attention visualization** on form fields
- Entity extraction from banking documents

---

### 🧾 3. DocVQA (Document Visual Question Answering)

| Property | Details |
|----------|---------|
| **Type** | Document images + layout understanding |
| **Size** | 50,000+ questions on 12,000+ documents |
| **Download** | [DocVQA](https://www.docvqa.org/) |

✅ **Useful for:**
- **Multi-modal extension** (image + text)
- Financial document reasoning
- Building intelligent document processing systems
- Understanding context in banking documents

---

## ✍️ 2️⃣ Signature Verification Datasets (Very Relevant for Banking)

### 🖊️ 4. CEDAR (Center of Excellence for Document Analysis and Recognition)

| Property | Details |
|----------|---------|
| **Type** | Genuine vs Forged Signatures |
| **Size** | 2,640 signatures (55 signers × 24 genuine + 24 forged each) |
| **Classes** | Genuine, Forged |
| **Download** | [CEDAR](https://www.cedar.buffalo.edu/NIJ/data/) |

✅ **Why highly recommended:**
- **Directly applicable** to signature fraud detection on cheques
- Perfect for **ViT-based pattern recognition**
- Ideal for **Explainable AI (Grad-CAM)** heatmaps
- Well-established benchmark in research papers

⭐ **HIGHLY RECOMMENDED FOR YOUR PROJECT**

**Integration code:**
```python
# data/signature_dataset/
# ├── genuine/
# └── forged/

from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

signature_dataset = datasets.ImageFolder('data/signature_dataset', transform=transform)
```

---

### 🖋️ 5. GPDS (Grupo de Procesado Digital de Señales)

| Property | Details |
|----------|---------|
| **Type** | Large signature verification dataset |
| **Size** | 4,000 individuals × 24 genuine + 30 forged each |
| **Total** | 160,000+ signature images |
| **Access** | Research request required |

✅ **Useful for:**
- Large-scale signature verification training
- Widely cited in **research papers**
- Robust model development

---

## 💳 3️⃣ Credit Card Fraud (Analytics Extension)

### 💰 6. Kaggle Credit Card Fraud Detection

| Property | Details |
|----------|---------|
| **Type** | Tabular transaction data |
| **Size** | 284,807 transactions (492 frauds) |
| **Features** | PCA-transformed (V1-V28) + Amount, Time |
| **Download** | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

✅ **Use this for:**
- **Risk scoring module** integration
- Fraud probability analytics dashboard
- **Multi-modal model** (Image + Transaction data)
- Building comprehensive fraud detection systems

**This makes your project very strong in Data Analytics!**

**Integration code:**
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load transaction data
df = pd.read_csv('data/creditcard.csv')

# Train risk scoring model
X = df.drop(['Class'], axis=1)
y = df['Class']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Combine with image prediction
def combined_fraud_score(image_pred, transaction_features):
    image_conf = image_pred['confidence']
    transaction_risk = model.predict_proba([transaction_features])[0][1]
    return 0.6 * image_conf + 0.4 * transaction_risk
```

---

## 🎥 4️⃣ ATM / Surveillance Fraud (Advanced Option)

### 📹 7. Kaggle Suspicious Activity Detection

| Property | Details |
|----------|---------|
| **Type** | Video/Image surveillance data |
| **Use Case** | Anomaly detection in ATM footage |
| **Download** | Various Kaggle datasets |

✅ **Used for:**
- **Suspicious activity detection** at ATMs
- Object recognition (masked faces, unusual behavior)
- Real-time fraud alerting systems

⚠️ More advanced but **very impressive** for a complete solution.

---

## 🏆 Recommended Dataset Combination (Best for Your Project)

Since your project implements: **ViT + SSL + XAI + Banking**

### 🔥 Optimal Dataset Strategy

| Priority | Dataset | Purpose | Impact |
|----------|---------|---------|--------|
| **Primary** | RVL-CDIP | Self-supervised pretraining for ViT | High |
| **Secondary** | CEDAR Signatures | Signature fraud detection | High |
| **Analytics** | Credit Card Fraud | Transaction risk scoring | Medium |
| **Advanced** | ATM Surveillance | Behavioral fraud detection | Bonus |

### 📈 What This Gives You

| Capability | Dataset Source |
|------------|----------------|
| ✅ Image-based fraud detection | Your synthetic + RVL-CDIP |
| ✅ Signature verification | CEDAR |
| ✅ Transaction analytics | Kaggle Credit Card |
| ✅ Risk scoring dashboard | Combined model |
| ✅ Explainable AI visualizations | Grad-CAM on all models |

---

## 🚀 Implementation Roadmap

### Phase 1: Improve ViT with RVL-CDIP Pretraining
```python
# 1. Download RVL-CDIP
from datasets import load_dataset
rvl_dataset = load_dataset("rvl_cdip", split="train")

# 2. Pretrain ViT with MAE on 400K documents
# 3. Fine-tune on your cheque dataset
```

### Phase 2: Add Signature Verification Module
```python
# 1. Download CEDAR dataset
# 2. Train separate ViT for signature detection
# 3. Extract signature region from cheque
# 4. Classify as genuine/forged
```

### Phase 3: Integrate Transaction Analytics
```python
# 1. Load credit card fraud dataset
# 2. Train XGBoost/RandomForest risk model
# 3. Combine image + transaction scores
# 4. Build unified fraud dashboard
```

### Phase 4: Deploy Complete System
```
┌─────────────────────────────────────────────────────────┐
│                 UNIFIED FRAUD DETECTION                  │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Cheque ViT  │  │ Signature   │  │ Transaction     │  │
│  │ Classifier  │  │ Verifier    │  │ Risk Scorer     │  │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         │                │                   │          │
│         └────────────────┼───────────────────┘          │
│                          ▼                              │
│              ┌───────────────────────┐                  │
│              │   FRAUD RISK SCORE    │                  │
│              │   (0-100%)            │                  │
│              └───────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

---

## 📚 Research Papers to Reference

1. **ViT**: "An Image is Worth 16x16 Words" - Dosovitskiy et al., 2020
2. **MAE**: "Masked Autoencoders Are Scalable Vision Learners" - He et al., 2021
3. **Document AI**: "LayoutLM: Pre-training of Text and Layout" - Xu et al., 2020
4. **Signature Verification**: "Writer-independent offline signature verification" - Hafemann et al., 2017
5. **Grad-CAM**: "Visual Explanations from Deep Networks" - Selvaraju et al., 2017

---

## 📁 Suggested Project Structure After Enhancement

```
bank_vit_project/
├── data/
│   ├── fraud_dataset/          # Current synthetic data
│   ├── rvl_cdip/               # Document pretraining
│   ├── cedar_signatures/       # Signature verification
│   └── transactions/           # Credit card fraud CSV
├── models/
│   ├── vit_model.py
│   ├── signature_vit.py        # NEW: Signature classifier
│   └── risk_scorer.py          # NEW: Transaction risk model
├── checkpoints/
│   ├── vit_rvl_pretrained.pth  # NEW: RVL-CDIP pretrained
│   ├── signature_vit.pth       # NEW: Signature model
│   └── risk_model.pkl          # NEW: XGBoost risk scorer
└── api.py                      # Unified API with all models
```

---

## ✅ Next Steps

1. **Download CEDAR** → Train signature verification ViT
2. **Download RVL-CDIP** → Pretrain ViT with MAE (improves accuracy significantly)
3. **Download Credit Card dataset** → Build risk scoring module
4. **Update API** → Combine all models into unified fraud score
5. **Update Dashboard** → Show signature analysis + transaction risk

---

*This document was generated as part of the Bank Fraud Detection ViT Project.*
*Last updated: February 28, 2026*
