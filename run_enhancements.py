"""
Advanced Enhancements for Bank Fraud Detection ViT Project
===========================================================
1. Grad-CAM Visualization for CNN/Hybrid models
2. K-Fold Cross-Validation
3. Advanced Data Augmentation (CutMix, MixUp, RandAugment)
4. Model Ensembling
5. ONNX Export & REST API preparation
6. Real Data Integration utilities
"""

import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
import matplotlib
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt

# Add project path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import project models
from models.hybrid_model import CNNBaseline, HybridCNNViT

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] Using {DEVICE}")

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw_images"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


###############################################################################
# SECTION 1: Grad-CAM Visualization for CNN/Hybrid Models
###############################################################################

class GradCAM:
    """Grad-CAM implementation for CNN models."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        assert self.gradients is not None, "Gradients not captured"
        assert self.activations is not None, "Activations not captured"
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def visualize_gradcam(model: nn.Module, target_layer: nn.Module, 
                      image_path: str, class_names: List[str],
                      save_path: Optional[str] = None):
    """Visualize Grad-CAM for a given image."""
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor: torch.Tensor = transform(image)  # type: ignore[assignment]
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(input_tensor)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = F.softmax(output, dim=1)[0, pred_class].item()
    
    # Resize CAM to image size
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224)))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image.resize((224, 224)))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Grad-CAM heatmap
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    img_array = np.array(image.resize((224, 224))) / 255.0
    from matplotlib.cm import ScalarMappable
    jet_cmap = matplotlib.colormaps.get_cmap('jet')  # type: ignore[attr-defined]
    heatmap = np.array(jet_cmap(cam_resized / 255.0))[:, :, :3]
    overlay = 0.6 * img_array + 0.4 * heatmap
    axes[2].imshow(overlay)
    axes[2].set_title(f'Prediction: {class_names[pred_class]} ({confidence:.2%})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved Grad-CAM: {save_path}")
    
    plt.close()
    return pred_class, confidence


###############################################################################
# SECTION 2: K-Fold Cross-Validation
###############################################################################

from sklearn.model_selection import StratifiedKFold

class ImageFolderDataset(Dataset):
    """Custom dataset for image folder structure."""
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def run_kfold_cv(model_class, model_kwargs: dict, n_folds: int = 5, 
                 epochs: int = 10, lr: float = 1e-4) -> Dict:
    """Run K-Fold Cross-Validation."""
    print(f"\n{'='*70}")
    print(f"  K-FOLD CROSS-VALIDATION ({n_folds} folds)")
    print(f"{'='*70}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = ImageFolderDataset(str(DATA_DIR), transform=transform)
    labels = [s[1] for s in dataset.samples]
    
    # K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), labels)):
        print(f"\n  Fold {fold + 1}/{n_folds}")
        print(f"  " + "-" * 40)
        
        # Create data loaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
        
        # Create model
        model = model_class(**model_kwargs).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for images, labels_batch in train_loader:
                images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels_batch).sum().item()
                train_total += labels_batch.size(0)
            
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_correct, val_total = 0, 0
            
            with torch.no_grad():
                for images, labels_batch in val_loader:
                    images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
                    outputs = model(images)
                    val_correct += (outputs.argmax(1) == labels_batch).sum().item()
                    val_total += labels_batch.size(0)
            
            val_acc = val_correct / val_total
            best_val_acc = max(best_val_acc, val_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"    Epoch [{epoch+1:2d}/{epochs}] Train: {train_acc:.4f} | Val: {val_acc:.4f}")
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': best_val_acc
        })
        print(f"  Fold {fold + 1} Best Val Acc: {best_val_acc:.4f}")
    
    # Summary
    mean_acc = np.mean([r['best_val_acc'] for r in fold_results])
    std_acc = np.std([r['best_val_acc'] for r in fold_results])
    
    print(f"\n  {'='*50}")
    print(f"  K-FOLD RESULTS: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  {'='*50}")
    
    return {
        'n_folds': n_folds,
        'fold_results': fold_results,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc
    }


###############################################################################
# SECTION 3: Advanced Data Augmentation
###############################################################################

class CutMix:
    """CutMix data augmentation."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        
        rand_index = torch.randperm(batch_size)
        
        # Get bounding box
        W, H = images.size(3), images.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return images, labels, labels[rand_index], lam


class MixUp:
    """MixUp data augmentation."""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        
        rand_index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[rand_index]
        
        return mixed_images, labels, labels[rand_index], lam


def get_advanced_transforms(use_randaugment: bool = True):
    """Get transforms with advanced augmentation."""
    from torchvision.transforms import autoaugment
    
    train_transforms = [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ]
    
    if use_randaugment:
        train_transforms.append(autoaugment.RandAugment(num_ops=2, magnitude=9))
    
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25)
    ])
    
    return transforms.Compose(train_transforms)


###############################################################################
# SECTION 4: Model Ensembling
###############################################################################

class EnsembleModel(nn.Module):
    """Ensemble of multiple models."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model, weight in zip(self.models, self.weights):
            out = model(x)
            outputs.append(F.softmax(out, dim=1) * weight)
        return torch.stack(outputs).sum(dim=0)


def create_ensemble(checkpoint_paths: List[str], model_classes: List, 
                    model_kwargs_list: List[dict]) -> EnsembleModel:
    """Create an ensemble from saved checkpoints."""
    models = []
    
    for ckpt_path, model_class, model_kwargs in zip(checkpoint_paths, model_classes, model_kwargs_list):
        model = model_class(**model_kwargs)
        
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        models.append(model)
    
    return EnsembleModel(models)


def evaluate_ensemble(ensemble: EnsembleModel, test_loader: DataLoader) -> Dict:
    """Evaluate ensemble model."""
    ensemble.eval()
    ensemble.to(DEVICE)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = ensemble(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }


###############################################################################
# SECTION 5: ONNX Export & REST API Preparation
###############################################################################

def export_to_onnx(model: nn.Module, save_path: str, input_shape: Tuple = (1, 3, 224, 224)):
    """Export PyTorch model to ONNX format."""
    model.eval()
    model.to(DEVICE)
    
    dummy_input = torch.randn(*input_shape).to(DEVICE)
    
    torch.onnx.export(
        model,
        (dummy_input,),
        save_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"  Exported ONNX model: {save_path}")
    print(f"  Size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")


def create_flask_api_code() -> str:
    """Generate Flask REST API code."""
    return '''"""
Flask REST API for Bank Fraud Detection
========================================
Run with: python api.py
"""

from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model (update path as needed)
MODEL_PATH = "checkpoints/cnn_best.pth"
CLASS_NAMES = ['genuine', 'fraud', 'tampered', 'forged']

# Initialize model
model = None

def load_model():
    global model
    from torchvision import models
    import torch.nn as nn
    
    # Create same model architecture as CNNBaseline
    backbone = models.resnet50(pretrained=False)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 4),
    )
    
    # Wrap in module to match checkpoint structure
    class CNNModel(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        def forward(self, x):
            return self.backbone(x)
    
    model = CNNModel(backbone)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        model = load_model()
    
    if 'file' not in request.files and 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Handle file upload or base64
        if 'file' in request.files:
            file = request.files['file']
            image = Image.open(file.stream).convert('RGB')
        else:
            image_data = base64.b64decode(request.json['image'])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()
        
        return jsonify({
            'prediction': CLASS_NAMES[pred_class],
            'confidence': round(confidence, 4),
            'probabilities': {
                name: round(prob.item(), 4) 
                for name, prob in zip(CLASS_NAMES, probs)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    global model
    if model is None:
        model = load_model()
    
    if 'images' not in request.json:
        return jsonify({'error': 'No images provided'}), 400
    
    results = []
    for i, img_b64 in enumerate(request.json['images']):
        try:
            image_data = base64.b64decode(img_b64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)[0]
                pred_class = probs.argmax().item()
                confidence = probs[pred_class].item()
            
            results.append({
                'index': i,
                'prediction': CLASS_NAMES[pred_class],
                'confidence': round(confidence, 4)
            })
        except Exception as e:
            results.append({'index': i, 'error': str(e)})
    
    return jsonify({'results': results})

if __name__ == '__main__':
    model = load_model()
    print(f"Model loaded from {MODEL_PATH}")
    app.run(host='0.0.0.0', port=5000, debug=False)
'''


###############################################################################
# SECTION 6: Real Data Integration
###############################################################################

def create_real_data_loader(data_dir: str, batch_size: int = 32) -> Tuple[DataLoader, List[str]]:
    """Create a data loader for real cheque images."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolderDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return loader, dataset.class_names


def analyze_real_data(data_dir: str):
    """Analyze real data distribution and quality."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"  Directory not found: {data_dir}")
        return None
    
    analysis = {
        'total_images': 0,
        'class_distribution': {},
        'image_sizes': [],
        'file_formats': {}
    }
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            images = list(class_dir.glob("*"))
            analysis['class_distribution'][class_name] = len(images)
            analysis['total_images'] += len(images)
            
            for img_path in images[:10]:  # Sample first 10
                try:
                    img = Image.open(img_path)
                    analysis['image_sizes'].append(img.size)
                    fmt = img_path.suffix.lower()
                    analysis['file_formats'][fmt] = analysis['file_formats'].get(fmt, 0) + 1
                except:
                    pass
    
    return analysis


###############################################################################
# MAIN EXECUTION
###############################################################################

def task_1_gradcam():
    """Generate Grad-CAM visualizations for CNN model."""
    print("\n" + "=" * 70)
    print("  TASK 1: Grad-CAM Visualization for CNN/Hybrid Models")
    print("=" * 70)
    
    # Load CNN model
    cnn_path = CHECKPOINTS_DIR / "cnn_best.pth"
    if not cnn_path.exists():
        print("  CNN checkpoint not found. Skipping Grad-CAM.")
        return
    
    # Create CNNBaseline model (matches saved checkpoint)
    model = CNNBaseline(pretrained=False, num_classes=4)
    
    checkpoint = torch.load(cnn_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(DEVICE)
    model.eval()
    
    # Target layer for Grad-CAM (last conv layer of ResNet50 backbone)
    target_layer: nn.Module = model.backbone.layer4[-1].conv3  # type: ignore[assignment]
    
    # Get sample images
    class_names = ['genuine', 'fraud', 'tampered', 'forged']
    
    for class_name in class_names:
        class_dir = DATA_DIR / class_name
        if class_dir.exists():
            # Support both jpg and png
            sample_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                sample_images.extend(list(class_dir.glob(ext))[:3])
            sample_images = sample_images[:3]
            
            for i, img_path in enumerate(sample_images):
                save_path = RESULTS_DIR / f"gradcam_{class_name}_{i+1}.png"
                visualize_gradcam(model, target_layer, str(img_path), 
                                class_names, str(save_path))
    
    print("  [TASK 1 COMPLETE] Grad-CAM visualizations saved to results/")


def task_2_kfold():
    """Run K-Fold Cross-Validation."""
    print("\n" + "=" * 70)
    print("  TASK 2: K-Fold Cross-Validation")
    print("=" * 70)
    
    # Use a simple CNN for CV (faster than ResNet50)
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.classifier = nn.Linear(128, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    results = run_kfold_cv(
        model_class=SimpleCNN,
        model_kwargs={'num_classes': 4},
        n_folds=5,
        epochs=10,
        lr=1e-3
    )
    
    # Save results
    with open(RESULTS_DIR / "kfold_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("  [TASK 2 COMPLETE] K-Fold results saved to results/kfold_results.json")


def task_3_augmentation_demo():
    """Demonstrate advanced augmentation."""
    print("\n" + "=" * 70)
    print("  TASK 3: Advanced Data Augmentation Demo")
    print("=" * 70)
    
    # Get sample image
    sample_dir = DATA_DIR / "genuine"
    if not sample_dir.exists():
        print("  Sample data not found. Skipping augmentation demo.")
        return
    
    sample_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        sample_images.extend(list(sample_dir.glob(ext)))
    if not sample_images:
        return
    
    # Load image
    img = Image.open(sample_images[0]).convert('RGB')
    
    # Apply different augmentations
    advanced_transform = get_advanced_transforms(use_randaugment=True)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Different augmentations
    augment_names = ['Aug 1', 'Aug 2', 'Aug 3', 'Aug 4', 'Aug 5', 'Aug 6', 'Aug 7']
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    for idx, name in enumerate(augment_names):
        row, col = (idx + 1) // 4, (idx + 1) % 4
        
        aug_tensor = advanced_transform(img)
        aug_img = inv_normalize(aug_tensor).permute(1, 2, 0).clamp(0, 1).numpy()
        
        axes[row, col].imshow(aug_img)
        axes[row, col].set_title(name)
        axes[row, col].axis('off')
    
    plt.suptitle('Advanced Data Augmentation (RandAugment + ColorJitter + RandomErasing)', fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "augmentation_demo.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  Saved augmentation demo: results/augmentation_demo.png")
    print("  [TASK 3 COMPLETE]")


def task_4_ensemble():
    """Create and evaluate ensemble model."""
    print("\n" + "=" * 70)
    print("  TASK 4: Model Ensembling")
    print("=" * 70)
    
    # Check available checkpoints
    cnn_path = CHECKPOINTS_DIR / "cnn_best.pth"
    hybrid_path = CHECKPOINTS_DIR / "hybrid_best.pth"
    
    if not cnn_path.exists():
        print("  CNN checkpoint not found. Skipping ensemble.")
        return
    
    # Create test loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolderDataset(str(DATA_DIR), transform=transform)
    
    # Use last 20% as test
    test_size = int(0.2 * len(dataset))
    test_indices = list(range(len(dataset) - test_size, len(dataset)))
    test_subset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    # Create CNN model using project CNNBaseline class
    cnn_model = CNNBaseline(pretrained=False, num_classes=4)
    checkpoint = torch.load(cnn_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        cnn_model.load_state_dict(checkpoint)
    cnn_model.eval()
    
    # Single model baseline
    print("  Evaluating CNN alone...")
    ensemble_single = EnsembleModel([cnn_model], [1.0])
    single_results = evaluate_ensemble(ensemble_single, test_loader)
    print(f"  CNN Accuracy: {single_results['accuracy']:.4f}")
    
    # Try with hybrid if available
    if hybrid_path.exists():
        print("  Loading Hybrid model for ensemble...")
        
        # Load hybrid using project HybridCNNViT class
        hybrid_model = HybridCNNViT(num_classes=4)
        checkpoint = torch.load(hybrid_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            hybrid_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            hybrid_model.load_state_dict(checkpoint)
        hybrid_model.eval()
        
        # Ensemble
        ensemble_combined = EnsembleModel([cnn_model, hybrid_model], [0.6, 0.4])
        ensemble_results = evaluate_ensemble(ensemble_combined, test_loader)
        print(f"  Ensemble (CNN + Hybrid) Accuracy: {ensemble_results['accuracy']:.4f}")
        
        # Save results
        results = {
            'cnn_only': single_results,
            'ensemble': ensemble_results
        }
    else:
        results = {'cnn_only': single_results}
    
    with open(RESULTS_DIR / "ensemble_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("  [TASK 4 COMPLETE] Ensemble results saved")


def task_5_onnx_api():
    """Export ONNX and create API code."""
    print("\n" + "=" * 70)
    print("  TASK 5: ONNX Export & REST API")
    print("=" * 70)
    
    # Export CNN to ONNX
    cnn_path = CHECKPOINTS_DIR / "cnn_best.pth"
    
    if cnn_path.exists():
        print("  Exporting CNN to ONNX...")
        
        model = CNNBaseline(pretrained=False, num_classes=4)
        checkpoint = torch.load(cnn_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        
        onnx_path = CHECKPOINTS_DIR / "cnn_model.onnx"
        export_to_onnx(model, str(onnx_path))
    
    # Create Flask API code
    print("  Generating Flask API code...")
    api_code = create_flask_api_code()
    api_path = BASE_DIR / "api.py"
    
    with open(api_path, 'w') as f:
        f.write(api_code)
    
    print(f"  API code saved: {api_path}")
    print("  Run API with: python api.py")
    print("  [TASK 5 COMPLETE]")


def task_6_real_data():
    """Real data integration utilities."""
    print("\n" + "=" * 70)
    print("  TASK 6: Real Data Integration")
    print("=" * 70)
    
    # Analyze current data
    print("  Analyzing current dataset...")
    analysis = analyze_real_data(str(DATA_DIR))
    
    if analysis:
        print(f"\n  Dataset Analysis:")
        print(f"  Total images: {analysis['total_images']}")
        print(f"  Class distribution:")
        for cls, count in analysis['class_distribution'].items():
            print(f"    - {cls}: {count}")
        print(f"  File formats: {analysis['file_formats']}")
        
        # Save analysis
        with open(RESULTS_DIR / "data_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
    
    # Create data integration guide
    guide = """
# Real Data Integration Guide

## Directory Structure
Place your real cheque images in the following structure:

```
data/real_images/
├── genuine/
│   ├── cheque_001.jpg
│   ├── cheque_002.jpg
│   └── ...
├── fraud/
│   ├── cheque_001.jpg
│   └── ...
├── tampered/
│   └── ...
└── forged/
    └── ...
```

## Image Requirements
- Supported formats: PNG, JPG, JPEG
- Recommended size: At least 224x224 pixels
- Color: RGB (3 channels)

## Running Inference
```python
from run_enhancements import create_real_data_loader
from torchvision import models
import torch

# Load model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('checkpoints/cnn_best.pth'))
model.eval()

# Load real data
loader, class_names = create_real_data_loader('data/real_images')

# Run inference
for images, labels in loader:
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        for pred in predictions:
            print(f"Predicted: {class_names[pred]}")
```

## Fine-tuning on Real Data
1. Prepare your dataset as shown above
2. Split into train/val/test (e.g., 70/15/15)
3. Use transfer learning from the pretrained CNN
4. Train with a lower learning rate (e.g., 1e-5)
"""
    
    with open(BASE_DIR / "REAL_DATA_GUIDE.md", 'w') as f:
        f.write(guide)
    
    print("  Created: REAL_DATA_GUIDE.md")
    print("  [TASK 6 COMPLETE]")


def main():
    """Run all enhancement tasks."""
    print("\n" + "#" * 70)
    print("  ADVANCED ENHANCEMENTS PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)
    
    start_time = time.time()
    
    # Run all tasks
    task_1_gradcam()
    task_2_kfold()
    task_3_augmentation_demo()
    task_4_ensemble()
    task_5_onnx_api()
    task_6_real_data()
    
    total_time = time.time() - start_time
    
    print("\n" + "#" * 70)
    print("  ALL ENHANCEMENTS COMPLETE!")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)
    
    print("\n  Generated files:")
    print("    - results/gradcam_*.png (Grad-CAM visualizations)")
    print("    - results/kfold_results.json (Cross-validation)")
    print("    - results/augmentation_demo.png (Aug demo)")
    print("    - results/ensemble_results.json (Ensemble metrics)")
    print("    - checkpoints/cnn_model.onnx (ONNX export)")
    print("    - api.py (Flask REST API)")
    print("    - REAL_DATA_GUIDE.md (Integration guide)")


if __name__ == "__main__":
    main()
