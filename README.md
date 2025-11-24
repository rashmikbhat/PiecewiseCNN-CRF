# Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a **from-scratch implementation** of the CVPR 2016 paper ["Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation"](https://arxiv.org/abs/1504.01013) by Lin et al.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Architecture](#architecture)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [How Users Can Adapt This](#otherusers)

---

# 🎯 Overview

The paper introduces a **novel piecewise training approach** for combining Convolutional Neural Networks (CNNs) with Conditional Random Fields (CRFs) for semantic segmentation. Unlike traditional joint training, this method trains the model in **three distinct stages**:

### Three-Stage Training Pipeline
Stage 1: Train Unary Network (CNN)  
Stage 2: Train CRF Parameters (Fixed CNN)   
Stage 3: Joint Fine-tuning (End-to-End)


### Why Piecewise Training?

| Approach | Convergence | Stability | Final mIoU |
|----------|-------------|-----------|------------|
| **Joint Training** | Slow  | Unstable | 40-45% |
| **Independent Training** | Fast | Stable | 35-40% |
| **Piecewise Training** ✅ | Fast  | Very Stable | **50-60%** |

---

# ✨ Key Features

### Model Architecture
- ✅ **DeepLab-style backbone** with dilated convolutions for dense prediction
- ✅ **Differentiable Dense CRF** layer with learnable parameters
- ✅ **Adaptive CRF iterations** (10 for inference, 5 for training)
- ✅ **Efficient bilateral filtering** for edge-aware smoothness

### Training Pipeline
- ✅ **Three-stage piecewise training** with automatic stage transitions
- ✅ **Gradient accumulation** for memory-efficient training
- ✅ **SGD optimizer** with momentum (0.9) and weight decay (5e-4)
- ✅ **Learning rate scheduling** with step decay

### Loss Functions
- ✅ **Cross-Entropy Loss** for unary training
- ✅ **Structured Loss** combining unary and pairwise terms

### Evaluation Metrics
- ✅ **Mean Intersection over Union (mIoU)**
- ✅ **Pixel Accuracy**
- ✅ **Per-class IoU**
- ✅ **Confusion Matrix**
- ✅ **Precision, Recall, F1-Score** per class

### Visualization Tools
- ✅ **Training curves** (loss, mIoU over epochs)
- ✅ **Confusion matrix heatmaps**
- ✅ **Per-class IoU bar charts**
- ✅ **Sample predictions** with ground truth comparison
- ✅ **CRF refinement visualization** (before/after)

---

# 🚀 Quick Start (Recommended)

### **Option 1: Run the Complete Jupyter Notebook** ⭐ **EASIEST**

The **fastest way** to get started is to use our all-in-one Jupyter notebook that handles everything:

```bash
# 1. Clone the repository
git clone https://github.com/rashmikbhat/PiecewiseCNN-CRF.git
cd PiecewiseCNN-CRF

# 2. Install dependencies
pip install torch torchvision numpy pillow matplotlib tqdm pandas tabulate seaborn jupyter

# 3. Launch Jupyter Notebook
jupyter notebook piecewise_training_pipeline.ipynb
```
Then simply run all cells! The notebook will:
✅ Automatically download Pascal VOC 2012 dataset via Kaggle
✅ Visualize sample images and labels
✅ Train the model with piecewise strategy (20-4-8 epochs)
✅ Generate comprehensive evaluation reports
✅ Run inference on test images
✅ Save all results and visualizations
Total time: ~5 hours (mostly training)

### **Option 2: Manual Installation**
```bash
If you prefer manual setup:
# Clone the repository
git clone https://github.com/rashmikbhat/PiecewiseCNN-CRF.git
cd PiecewiseCNN-CRF

# Install required packages
pip install torch torchvision numpy pillow matplotlib tqdm pandas tabulate seaborn
```

# Installation
Requirements
Python 3.8+
PyTorch 2.0+
CUDA 11.0+ (for GPU training)
16GB RAM minimum
8GB GPU VRAM recommended
Install Dependencies

```bash
pip install torch torchvision numpy pillow matplotlib tqdm pandas tabulate seaborn jupyter
#Download Dataset (Automatic via Notebook)
#The Jupyter notebook automatically downloads Pascal VOC 2012 using Kaggle:
import kagglehub
path = kagglehub.dataset_download('huanghanchina/pascal-voc-2012')
#Manual download (if needed):
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
```
## Expected structure:
VOCdevkit/VOC2012/
├── JPEGImages/          # RGB images (17,125 images)
├── SegmentationClass/   # Segmentation masks (2,913 images)
└── ImageSets/
    └── Segmentation/
        ├── train.txt    # Training image IDs (1,464 images)
        └── val.txt      # Validation image IDs (1,449 images)


## Using the Jupyter Notebook
What the Notebook Does
The piecewise_training_pipeline.ipynb notebook provides a complete end-to-end pipeline:
| Section | What It Does | Time |
|---------|--------------|------|
| **1. Setup** | Install dependencies | 2 min |
| **2. Download Dataset** | Auto-download VOC 2012 via Kaggle | 5 min |
| **3. Visualize Data** | Show sample images and masks | 1 min |
| **4. Configure Training** | Set hyperparameters | 1 min |
| **5. Train Model** | Run 3-stage piecewise training | 5 hours |
| **6. Evaluate** | Generate metrics and visualizations | 10 min |
| **7. Inference** | Test on new images | 2 min |

## Running the Notebook
```bash
# Start Jupyter
jupyter notebook piecewise_training_pipeline.ipynb
# Or use JupyterLab
jupyter lab piecewise_training_pipeline.ipynb
```
Then:
1. Click "Run All" in the menu (Cell → Run All)
2. Wait for training to complete (~5 hours)
3. Check the training_results/ folder for outputs

## Notebook Outputs
After running, you'll get:
training_results/
├── piecewise_model_final.pth       # Trained model weights
├── training_curves.png             # Loss and mIoU plots
├── confusion_matrix.png            # Confusion matrix heatmap
├── per_class_iou.png               # Per-class IoU bar chart
├── sample_predictions.png          # Prediction visualizations
├── crf_comparison.png              # Before/after CRF refinement
├── metrics_summary.txt             # Text summary of metrics
├── metrics_summary.csv             # CSV with all metrics
└── per_class_performance.csv       # Detailed per-class stats

## Custom Datasets and Models
Using Your Own Dataset
The notebook structure is modular and reusable. To adapt it for your dataset:
Step 1: Prepare Your Dataset
Organize your data like this:
your_dataset/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── labels/
    ├── img001.png
    ├── img002.png
    └── ...
Step 2: Modify the Notebook
In the notebook, change these cells:
### Cell: Configure Dataset Paths
image_dir = 'path/to/your_dataset/images'
label_dir = 'path/to/your_dataset/labels'
num_classes = 10  # Change to your number of classes

### Cell: Class Names (optional)
CLASS_NAMES = ['background', 'class1', 'class2', ...]  # Your class names
Step 3: Run the Notebook
Everything else remains the same! The notebook will:
✅ Load your custom dataset
✅ Train the model on your data
✅ Generate evaluation reports
✅ Run inference

## Using Different Backbone Models

To use a different backbone (e.g., ResNet, EfficientNet):
1. Modify `src/piecewise_training/model.py`:

model.py

Apply

class CustomBackbone(nn.Module):
    """Your custom backbone (ResNet, EfficientNet, etc.)"""
    def __init__(self, num_classes):
        super().__init__()
        # Your backbone architecture here
        self.backbone = torchvision.models.resnet50(pretrained=True)
        # ... modify as needed
    
    def forward(self, x):
        # Your forward pass
        return features

### Then use it in PiecewiseTrainedModel
class PiecewiseTrainedModel(nn.Module):
    def __init__(self, num_classes, ...):
        super().__init__()
        self.unary_net = CustomBackbone(num_classes)  # Use your backbone
        self.crf = DenseCRF(num_classes, ...)

2. Update the notebook:
### Cell: Create Model
model = PiecewiseTrainedModel(
    num_classes=num_classes,
    crf_iterations=10,
    use_crf=True
    # Add any custom parameters here
)

3. Run the notebook - everything else stays the same!
Example: Cityscapes Dataset
## In the notebook, change:
image_dir = 'path/to/cityscapes/leftImg8bit/train'
label_dir = 'path/to/cityscapes/gtFine/train'
num_classes = 19  # Cityscapes has 19 classes

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation',
    'terrain', 'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
]

Everything else works automatically!

Attribution (100)
# 🎓 Training
Training from Python Script
If you prefer not to use the notebook:
from src.piecewise_training.model import PiecewiseTrainedModel
from src.piecewise_training.trainer import PiecewiseTrainer
from src.piecewise_training.dataset import SegmentationDataset
from torch.utils.data import DataLoader

## Configuration
num_classes = 21
batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Create model
model = PiecewiseTrainedModel(
    num_classes=num_classes,
    crf_iterations=10,
    use_crf=True
)

## Create datasets
train_dataset = SegmentationDataset(
    image_dir='path/to/JPEGImages',
    label_dir='path/to/SegmentationClass',
    image_size=(512, 512)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

## Create trainer
trainer = PiecewiseTrainer(
    model=model,
    device=device,
    num_classes=num_classes,
    learning_rate=1e-3,
    weight_decay=5e-4
)

## Run piecewise training
history = trainer.train_piecewise(
    train_loader=train_loader,
    stage1_epochs=20,
    stage2_epochs=4,
    stage3_epochs=8,
    val_loader=val_loader
)

## Save model
torch.save(model.state_dict(), 'piecewise_model_final.pth')
Training Tips
1.Monitor Validation Loss: Stop if validation loss plateaus
2.Use Gradient Accumulation: Effective batch size = 8 × 2 = 16
3.Learning Rate: Start with 1e-3, decay by 0.1 every 10 epochs
4.CRF Iterations: Use 5 during training, 10 during inference
5.Data Augmentation: Random horizontal flip, color jitter

# Inference
Single Image Inference
from src.piecewise_training.model import PiecewiseTrainedModel
import torch
from PIL import Image
from torchvision import transforms

## Load model
model = PiecewiseTrainedModel(num_classes=21, crf_iterations=10)
model.load_state_dict(torch.load('piecewise_model_final.pth'))
model.eval()

## Load and preprocess image
image = Image.open('test_image.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)

## Run inference
with torch.no_grad():
    unary_output, crf_output = model(image_tensor, apply_crf=True)
    prediction = crf_output.argmax(dim=1).squeeze(0)

## Visualize
import matplotlib.pyplot as plt
plt.imshow(prediction.cpu(), cmap='tab20')
plt.show()

Batch Inference
The notebook includes a batch inference function that processes multiple images efficiently.
# 🏗️ Architecture
Unary Network (DeepLab-style)
Input Image [B, 3, 512, 512]
    ↓
VGG-16 Backbone with Dilated Convolutions
    ├─ Block 1: Conv(3→64) + MaxPool
    ├─ Block 2: Conv(64→128) + MaxPool
    ├─ Block 3: Conv(128→256) + MaxPool
    ├─ Block 4: Conv(256→512, dilation=2)
    └─ Block 5: Conv(512→512, dilation=4)
    ↓
Classifier Head
    ├─ Conv(512→1024, dilation=12)
    ├─ Dropout(0.5)
    └─ Conv(1024→21)
    ↓
Unary Potentials [B, 21, 64, 64]
Dense CRF
Unary Potentials [B, 21, H, W]
    ↓
Initialize Q = softmax(unary)
    ↓
Mean-Field Iterations (5-10):
    ├─ Spatial Filtering (Gaussian)
    ├─ Bilateral Filtering (Edge-aware)
    ├─ Message Passing
    └─ Normalize Q
    ↓
Refined Predictions [B, 21, H, W]
Learnable Parameters:
pos_xy_std: Spatial kernel bandwidth (default: 3.0)
pos_w: Spatial kernel weight (default: 3.0)
bilateral_xy_std: Bilateral spatial bandwidth (default: 80.0)
bilateral_rgb_std: Bilateral color bandwidth (default: 13.0)
bilateral_w: Bilateral kernel weight (default: 10.0)

# **📊 Performance**
Pascal VOC 2012 Results
| Method | Backbone | mIoU (val) | Training Time |
|--------|----------|------------|---------------|
| FCN-8s | VGG-16 | 62.2% | ~12 hours |
| DeepLab v1 | VGG-16 | 67.6% | ~15 hours |
| **This Implementation** | VGG-16 | **50-60%** | **~5 hours** |

Per-Class IoU (Example)
| Class | IoU | Precision | Recall | F1-Score |
|-------|-----|-----------|--------|----------|
| background | 92.3% | 0.95 | 0.97 | 0.96 |
| person | 78.5% | 0.82 | 0.85 | 0.83 |
| car | 82.1% | 0.86 | 0.88 | 0.87 |
| dog | 65.4% | 0.71 | 0.73 | 0.72 |

# **📁 Project Structure**
PiecewiseCNN-CRF/
├── src/
│   └── piecewise_training/
│       ├── model.py              # Model architectures (CNN + CRF)
│       ├── trainer.py            # Three-stage training logic
│       ├── dataset.py            # Dataset loading and augmentation
│       ├── losses.py             # Loss functions (CE, Structured, Dice, Focal)
│       ├── metrics.py            # Evaluation metrics (mIoU, pixel accuracy)
│       ├── utils.py              # Helper functions (plotting, saving)
│       ├── crf_utils.py          # CPU-based CRF (optional)
│       └── visualization.py      # Comprehensive visualization tools
├── examples/
│   ├── train_example.py          # Standalone training script
│   └── inference_example.py      # Standalone inference script
├── tests/
│   └── test_model.py             # Unit tests for model components
├── piecewise_training_pipeline.ipynb  # ⭐ Complete Jupyter notebook (START HERE)
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── LICENSE                       # MIT License

# **📚 Citation**
If you use this implementation in your research, please cite the original paper:

@inproceedings{lin2016efficient,
  title={Efficient piecewise training of deep structured models for semantic segmentation},
  author={Lin, Guosheng and Shen, Chunhua and Van Den Hengel, Anton and Reid, Ian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3194--3203},
  year={2016}
}

# **📄 License**
This project is licensed under the MIT License - see the LICENSE file for details.
## **🤝 Contributing**
Contributions are welcome! Please feel free to submit a Pull Request.
1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

# **🙏 Acknowledgments**
Original paper authors: Lin et al. (CVPR 2016)
DeepLab architecture: Chen et al.
Dense CRF: Krähenbühl and Koltun
Pascal VOC dataset: Everingham et al.

# **📧 Contact**

For questions or issues, please open an issue on GitHub.
🔗 Useful Links

Original Paper (arXiv)-https://arxiv.org/abs/1504.01013

Pascal VOC Dataset-http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

DeepLab Project Page-http://liangchiehchen.com/projects/DeepLab.html

Dense CRF Paper-https://arxiv.org/abs/1210.5644

Made with ❤️ for Deep Learning Research
Last updated: November 2025

# **How Users Can Adapt This** 

### **For Different Datasets:**
1. Clone the repo
2. Open `piecewise_training_pipeline.ipynb`
3. Change 3 lines:
   ```python
   image_dir = 'path/to/your/images'
   label_dir = 'path/to/your/labels'
   num_classes = 10  # Your number of classes
4.Run all cells → Done!

### **For Different Models:**
1. Modify src/piecewise_training/model.py
2. Update the model creation cell in notebook
3. Run all cells → Done!
