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
git clone https://github.com/rashmikbhat/PiecewiseCNN-CRF/.git
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
