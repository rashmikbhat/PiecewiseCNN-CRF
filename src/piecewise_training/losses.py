
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PiecewiseCRFLoss(nn.Module):
    """
    Piecewise training loss from Lin et al. CVPR 2016 (Equation 10).
    
    Decomposes CRF objective into independent per-potential likelihoods:
    - Unary likelihood: P^U(yp|x) = exp[-U(yp,xp)] / Σ exp[-U(y'p,xp)]
    - Pairwise likelihood: P^V(yp,yq|x) = exp[-V(yp,yq,xpq)] / Σ exp[-V(y'p,y'q,xpq)]
    
    This avoids expensive global partition function Z(x).
    """
    def __init__(
        self,
        num_classes: int,
        unary_weight: float = 1.0,
        pairwise_weight: float = 0.1,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.unary_weight = unary_weight
        self.pairwise_weight = pairwise_weight
        self.ignore_index = ignore_index
        
        # Unary likelihood: standard cross-entropy (Equation 8)
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
    
    def forward(
        self,
        unary_output: torch.Tensor,
        crf_output: torch.Tensor,
        target: torch.Tensor,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            unary_output: [B, C, H, W] - unary predictions
            crf_output: [B, C, H, W] - CRF refined predictions
            target: [B, H, W] - ground truth labels
            image: [B, 3, H, W] - input image (for pairwise term)
        
        Returns:
            Combined loss
        """
        # 1. Unary loss (on CRF output, not unary output!)
        loss_unary = self.ce_loss(crf_output, target)
        
        # 2. Pairwise smoothness loss
        loss_pairwise = self._compute_pairwise_loss(crf_output, image, target)
        
        # 3. Combine
        total_loss = (
            self.unary_weight * loss_unary +
            self.pairwise_weight * loss_pairwise
        )
        
        return total_loss
    
    def _compute_pairwise_loss(
        self,
        predictions: torch.Tensor,
        image: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise smoothness loss.
        Penalizes predictions that differ from neighbors with similar colors.
        
        Args:
            predictions: [B, C, H, W] - probability distributions
            image: [B, 3, H, W] - input image
            target: [B, H, W] - ground truth (for masking)
        
        Returns:
            Scalar pairwise loss
        """
        B, C, H, W = predictions.shape
        
        # Get predicted labels
        pred_labels = predictions.argmax(dim=1)  # [B, H, W]
        
        # Create valid mask (ignore background)
        valid_mask = (target != self.ignore_index)
        
        # Compute horizontal differences
        # Image similarity
        image_diff_x = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])  # [B, 3, H, W-1]
        image_sim_x = torch.exp(-image_diff_x.sum(dim=1))  # [B, H, W-1]
        
        # Label differences
        label_diff_x = (pred_labels[:, :, 1:] != pred_labels[:, :, :-1]).float()  # [B, H, W-1]
        
        # Mask
        mask_x = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]  # [B, H, W-1]
        
        # Pairwise loss (horizontal)
        loss_x = (image_sim_x * label_diff_x * mask_x).sum() / (mask_x.sum() + 1e-6)
        
        # Compute vertical differences
        image_diff_y = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])  # [B, 3, H-1, W]
        image_sim_y = torch.exp(-image_diff_y.sum(dim=1))  # [B, H-1, W]
        
        label_diff_y = (pred_labels[:, 1:, :] != pred_labels[:, :-1, :]).float()  # [B, H-1, W]
        
        mask_y = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]  # [B, H-1, W]
        
        loss_y = (image_sim_y * label_diff_y * mask_y).sum() / (mask_y.sum() + 1e-6)
        
        return loss_x + loss_y


class UnaryLoss(nn.Module):
    """
    Unary loss for Stage 1 training (Equation 8 from paper).
    
    This is simply: -log P^U(yp|x) = -log[exp(-U(yp,xp)) / Σ exp(-U(y'p,xp))]
    Which is equivalent to standard cross-entropy loss.
    """
    
    def __init__(
        self,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, H, W] - logits from unary network
            target: [B, H, W] - ground truth labels
        
        Returns:
            Scalar loss value
        """
        return self.ce_loss(predictions, target)