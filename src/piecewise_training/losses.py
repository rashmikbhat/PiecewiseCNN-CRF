import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredLoss(nn.Module):
    """
    Structured loss combining unary and pairwise terms.
    Implements the loss function from the paper.
    """
    def __init__(
        self,
        num_classes: int,
        unary_weight: float = 1.0,
        pairwise_weight: float = 0.1,
        ignore_index: int = 255
    ):
        super().__init__()
        self.num_classes = num_classes
        self.unary_weight = unary_weight
        self.pairwise_weight = pairwise_weight
        self.ignore_index = ignore_index
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        unary_output: torch.Tensor,
        crf_output: torch.Tensor,
        target: torch.Tensor,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute structured loss.
        
        Args:
            unary_output: Unary predictions [B, C, H, W]
            crf_output: CRF refined predictions [B, C, H, W]
            target: Ground truth labels [B, H, W]
            image: Input image [B, 3, H, W]
        Returns:
            Total loss
        """
        # Unary loss
        loss_unary = self.ce_loss(unary_output, target)
        
        # Pairwise loss (smoothness)
        loss_pairwise = self._pairwise_loss(crf_output, image, target)
        
        # Combined loss
        total_loss = (
            self.unary_weight * loss_unary +
            self.pairwise_weight * loss_pairwise
        )
        
        return total_loss
    
    def _pairwise_loss(
        self,
        predictions: torch.Tensor,
        image: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise smoothness loss.
        Encourages similar predictions for similar pixels.
        """
        B, C, H, W = predictions.shape
        
        # Get prediction probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Compute image gradients
        image_dx = image[:, :, :, 1:] - image[:, :, :, :-1]
        image_dy = image[:, :, 1:, :] - image[:, :, :-1, :]
        
        # Compute prediction gradients
        pred_dx = probs[:, :, :, 1:] - probs[:, :, :, :-1]
        pred_dy = probs[:, :, 1:, :] - probs[:, :, :-1, :]
        
        # Weight by image similarity (edge-aware)
        weight_x = torch.exp(-torch.sum(image_dx ** 2, dim=1, keepdim=True))
        weight_y = torch.exp(-torch.sum(image_dy ** 2, dim=1, keepdim=True))
        
        # Weighted smoothness loss
        loss_x = torch.mean(weight_x * torch.sum(pred_dx ** 2, dim=1, keepdim=True))
        loss_y = torch.mean(weight_y * torch.sum(pred_dy ** 2, dim=1, keepdim=True))
        
        return loss_x + loss_y


class DiceLoss(nn.Module):
    """
    Dice loss for semantic segmentation.
    Can be used as an alternative or additional loss.
    """
    def __init__(
        self,
        num_classes: int,
        smooth: float = 1.0,
        ignore_index: int = 255
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            predictions: Model predictions [B, C, H, W]
            target: Ground truth labels [B, H, W]
        Returns:
            Dice loss
        """
        # Get probabilities
        probs = F.softmax(predictions, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(
            target.clamp(0, self.num_classes - 1),
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()
        
        # Mask out ignore index
        mask = (target != self.ignore_index).unsqueeze(1).float()
        probs = probs * mask
        target_one_hot = target_one_hot * mask
        
        # Compute Dice coefficient
        intersection = torch.sum(probs * target_one_hot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            predictions: Model predictions [B, C, H, W]
            target: Ground truth labels [B, H, W]
        Returns:
            Focal loss
        """
        # Get log probabilities
        log_probs = F.log_softmax(predictions, dim=1)
        
        # Get probabilities
        probs = torch.exp(log_probs)
        
        # Gather target class probabilities
        target_clamped = target.clamp(0, predictions.shape[1] - 1)
        target_log_probs = log_probs.gather(1, target_clamped.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, target_clamped.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Compute loss
        loss = -self.alpha * focal_weight * target_log_probs
        
        # Mask out ignore index
        mask = (target != self.ignore_index).float()
        loss = loss * mask
        
        return loss.sum() / (mask.sum() + 1e-6)
