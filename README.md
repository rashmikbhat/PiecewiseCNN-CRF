# Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation
This repository contains a from-scratch implementation of the CVPR 2016 paper "Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation" by Lin et al.

# Overview

The paper introduces a novel piecewise training approach for combining Convolutional Neural Networks (CNNs) with Conditional Random Fields (CRFs) for semantic segmentation. The key innovation is training the model in three stages:
Stage 1: Train the unary potential network (CNN) independently
Stage 2: Fix the unary network and train CRF parameters
Stage 3: Fine-tune the entire model end-to-end
This approach is more efficient than joint training and produces better results than training components separately.

# Key Features

DeepLab-style backbone with dilated convolutions for dense prediction
Differentiable Dense CRF layer with learnable parameters
Three-stage piecewise training pipeline
Comprehensive metrics including mIoU, pixel accuracy, and per-class IoU
Multiple loss functions (Cross-Entropy, Dice, Focal, Structured)
Efficient bilateral filtering for CRF message passing

# Installation
pip install torch torchvision numpy pillow matplotlib tqdm

# Usage

# Training
from src.piecewise_training.model import PiecewiseTrainedModel
from src.piecewise_training.trainer import PiecewiseTrainer

# Create model
model = PiecewiseTrainedModel(num_classes=21, crf_iterations=10)

# Create trainer
trainer = PiecewiseTrainer(
    model=model,
    device=device,
    num_classes=21,
    learning_rate=1e-3
)

# Run piecewise training
history = trainer.train_piecewise(
    train_loader=train_loader,
    stage1_epochs=20,
    stage2_epochs=5,
    stage3_epochs=10,
    val_loader=val_loader
)
# Inference

from src.piecewise_training.model import PiecewiseTrainedModel

model = PiecewiseTrainedModel(num_classes=21)
model.load_state_dict(torch.load('model.pth'))
model.eval()

with torch.no_grad():
    unary_output, crf_output = model(image, apply_crf=True)
    prediction = crf_output.argmax(dim=1)

# Architecture

Unary Network (DeepLab-style)
VGG-16 inspired backbone with dilated convolutions
Maintains spatial resolution through atrous convolutions
Produces dense unary potentials for each pixel

Dense CRF
Learnable Gaussian kernel parameters
Bilateral filtering for edge-aware smoothness
Mean-field inference for efficient optimization
Differentiable for end-to-end training

Training Strategy
The piecewise training approach offers several advantages:
1.Stability: Each component is trained separately before joint optimization
2.Efficiency: Faster convergence than joint training from scratch
3.Modularity: Easy to experiment with different unary networks or CRF configurations
4.Performance: Better final results than independent training

File Structure
project/
├── src/
│   └── piecewise_training/
│       ├── model.py          # Model architectures
│       ├── trainer.py        # Training logic
│       ├── losses.py         # Loss functions
│       ├── metrics.py        # Evaluation metrics
│       ├── dataset.py        # Dataset utilities
│       ├── utils.py          # Helper functions
│       └── crf_utils.py      # CRF utilities
├── examples/
│   ├── train_example.py      # Training example
│   └── inference_example.py  # Inference example
├── tests/
│   └── test_model.py         # Unit tests
└── README.md

Citation
If you use this implementation, please cite the original paper:
@inproceedings{lin2016efficient,
  title={Efficient piecewise training of deep structured models for semantic segmentation},
  author={Lin, Guosheng and Shen, Chunhua and Van Den Hengel, Anton and Reid, Ian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3194--3203},
  year={2016}
}