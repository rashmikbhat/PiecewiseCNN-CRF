import torch
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from PIL import Image


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    """
    Compute IoU for each class.
    
    Args:
        pred: Predictions [H, W]
        target: Ground truth [H, W]
        num_classes: Number of classes
    Returns:
        IoU per class
    """
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return np.array(ious)


def visualize_segmentation(
    image: torch.Tensor,
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    num_classes: int,
    save_path: str = None
) -> None:
    """
    Visualize segmentation results.
    
    Args:
        image: Input image [3, H, W]
        prediction: Predicted segmentation [H, W]
        ground_truth: Ground truth segmentation [H, W]
        num_classes: Number of classes
        save_path: Path to save visualization
    """
    # Create color map
    colors = plt.cm.get_cmap('tab20', num_classes)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = image.cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Prediction
    pred_colored = colors(prediction.cpu().numpy())[:, :, :3]
    axes[1].imshow(pred_colored)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Ground truth
    gt_colored = colors(ground_truth.cpu().numpy())[:, :, :3]
    axes[2].imshow(gt_colored)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history: dict, save_path: str = None) -> None:
    """
    Plot training history across all stages.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    stages = ['stage1', 'stage2', 'stage3']
    stage_names = ['Stage 1: Unary', 'Stage 2: CRF', 'Stage 3: Joint']
    
    for idx, (stage, name) in enumerate(zip(stages, stage_names)):
        if stage not in history or not history[stage]:
            continue
        
        stage_history = history[stage]
        
        # Plot loss
        if 'train_loss' in stage_history:
            axes[0, idx].plot(stage_history['train_loss'], label='Train Loss')
        if 'val_loss' in stage_history:
            axes[0, idx].plot(stage_history['val_loss'], label='Val Loss')
        axes[0, idx].set_title(f'{name} - Loss')
        axes[0, idx].set_xlabel('Epoch')
        axes[0, idx].set_ylabel('Loss')
        axes[0, idx].legend()
        axes[0, idx].grid(True)
        
        # Plot mIoU
        if 'val_miou' in stage_history:
            axes[1, idx].plot(stage_history['val_miou'], label='Val mIoU')
            axes[1, idx].set_title(f'{name} - mIoU')
            axes[1, idx].set_xlabel('Epoch')
            axes[1, idx].set_ylabel('mIoU')
            axes[1, idx].legend()
            axes[1, idx].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()



class PolyLRScheduler:
    """
    Polynomial learning rate decay scheduler.
    Implements: lr = base_lr * (1 - iter/max_iter)^power
    
    This is the standard scheduler used in DeepLab and the piecewise training paper.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iterations: int,
        power: float = 0.9
    ):
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.power = power
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_iter = 0
    
    def step(self):
        """Update learning rate (call this EVERY iteration, not per epoch!)."""
        self.current_iter += 1
        
        # Compute decay factor: (1 - iter/max_iter)^power
        factor = (1 - self.current_iter / self.max_iterations) ** self.power
        
        # Update LR for all parameter groups
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * factor
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rates (for compatibility with PyTorch schedulers)."""
        return self.get_lr()
    
    def state_dict(self):
        """Return scheduler state for checkpointing."""
        return {
            'current_iter': self.current_iter,
            'base_lrs': self.base_lrs,
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.current_iter = state_dict['current_iter']
        self.base_lrs = state_dict['base_lrs']