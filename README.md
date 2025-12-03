# Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a **from-scratch implementation** of the CVPR 2016 paper ["Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation"](https://arxiv.org/abs/1504.01013) by Lin et al.


# 🎯 Overview

The paper introduces a **novel piecewise training approach** for combining Convolutional Neural Networks (CNNs) with Conditional Random Fields (CRFs) for semantic segmentation. Unlike traditional joint training, this method trains the model in **three distinct stages**:

### Three-Stage Training Pipeline
Stage 1: Train Unary Network (CNN)  
Stage 2: Train CRF Parameters (Fixed CNN)   
Stage 3: Joint Fine-tuning (End-to-End)
---

# ✨ Key Features

### Model Architecture
-  **DeepLab-style backbone** with dilated convolutions for dense prediction
-  **Differentiable Dense CRF** layer with learnable parameters
-  **Adaptive CRF iterations** (10 for inference, 5 for training)
-  **Efficient bilateral filtering** for edge-aware smoothness

### Training Pipeline
-  **Three-stage piecewise training** with automatic stage transitions
-  **Gradient accumulation** for memory-efficient training
-  **SGD optimizer** with momentum (0.9) and weight decay (5e-4)
-  **Learning rate scheduling** with step decay

### Loss Functions
-  **Cross-Entropy Loss** for unary training
-  **Structured Loss** combining unary and pairwise terms

### Evaluation Metrics
-  **Mean Intersection over Union (mIoU)**
-  **Pixel Accuracy**
-  **Per-class IoU**
-  **Confusion Matrix**
-  **Precision, Recall, F1-Score** per class

### Visualization Tools
-  **Training curves** (loss, mIoU over epochs)
-  **Confusion matrix heatmaps**
-  **Per-class IoU bar charts**
-  **Sample predictions** with ground truth comparison
-  **CRF refinement visualization** (before/after)

---

