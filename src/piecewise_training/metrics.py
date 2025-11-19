import torch
import numpy as np
from typing import Dict, List


class SegmentationMetrics:
    """
    Comprehensive metrics for semantic segmentation.
    """
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new predictions.
        
        Args:
            predictions: Predicted labels [B, H, W] or [B, C, H, W]
            targets: Ground truth labels [B, H, W]
        """
        # Convert predictions to labels if needed
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)
        
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Create mask for valid pixels
        mask = targets != self.ignore_index
        
        # Update confusion matrix
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i, j] += np.sum(
                    (targets[mask] == i) & (predictions[mask] == j)
                )
        
        self.total_samples += mask.sum()
    
    def compute_iou(self) -> np.ndarray:
        """Compute IoU for each class."""
        iou_per_class = []
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            denominator = tp + fp + fn
            if denominator > 0:
                iou = tp / denominator
            else:
                iou = float('nan')
            
            iou_per_class.append(iou)
        
        return np.array(iou_per_class)
    
    def compute_miou(self) -> float:
        """Compute mean IoU."""
        iou = self.compute_iou()
        valid_iou = iou[~np.isnan(iou)]
        return valid_iou.mean() if len(valid_iou) > 0 else 0.0
    
    def compute_pixel_accuracy(self) -> float:
        """Compute overall pixel accuracy."""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / total if total > 0 else 0.0
    
    def compute_mean_accuracy(self) -> float:
        """Compute mean class accuracy."""
        class_accuracies = []
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            total = self.confusion_matrix[i, :].sum()
            
            if total > 0:
                acc = tp / total
                class_accuracies.append(acc)
        
        return np.mean(class_accuracies) if class_accuracies else 0.0
    
    def compute_frequency_weighted_iou(self) -> float:
        """Compute frequency weighted IoU."""
        iou = self.compute_iou()
        freq = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        
        valid_mask = ~np.isnan(iou)
        fwiou = (freq[valid_mask] * iou[valid_mask]).sum()
        
        return fwiou
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all computed metrics."""
        return {
            'pixel_accuracy': self.compute_pixel_accuracy(),
            'mean_accuracy': self.compute_mean_accuracy(),
            'miou': self.compute_miou(),
            'fwiou': self.compute_frequency_weighted_iou()
        }
    
    def get_per_class_iou(self, class_names: List[str] = None) -> Dict[str, float]:
        """Get IoU for each class."""
        iou = self.compute_iou()
        
        if class_names is None:
            class_names = [f'class_{i}' for i in range(self.num_classes)]
        
        return {name: iou_val for name, iou_val in zip(class_names, iou)}