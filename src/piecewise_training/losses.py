import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F

class PiecewiseCRFLoss(nn.Module):
    """
    Piecewise training loss from the paper (Equation 10).
    
    Decomposes CRF objective into:
    1. Unary likelihood: P^U(yp|x) - Softmax over K classes
    2. Pairwise likelihood: P^V(yp,yq|x) - Softmax over spatial neighbors
    
    This avoids expensive global partition function Z(x).
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
        
        # Standard cross-entropy for unary term
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        unary_output: torch.Tensor,
        crf_output: torch.Tensor,
        target: torch.Tensor,
        image: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute piecewise CRF loss.
        
        Args:
            unary_output: Unary predictions [B, C, H, W]
            crf_output: CRF refined predictions [B, C, H, W]
            target: Ground truth labels [B, H, W]
            image: Input image [B, 3, H, W] (optional, for pairwise term)
        
        Returns:
            Total piecewise loss
        """
        # ✅ Unary likelihood term: -log P^U(yp|x)
        # This is just standard cross-entropy
        loss_unary = self.ce_loss(unary_output, target)
        
        # ✅ Pairwise likelihood term: -log P^V(yp,yq|x)
        # Computed on spatial neighbors
        if crf_output is not None:
            loss_pairwise = self._pairwise_likelihood_loss(
                crf_output, target, image
            )
        else:
            loss_pairwise = torch.tensor(0.0, device=unary_output.device)
        
        # ✅ Combined piecewise loss (Equation 10)
        total_loss = (
            self.unary_weight * loss_unary +
            self.pairwise_weight * loss_pairwise
        )
        
        return total_loss
    
    def _pairwise_likelihood_loss(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
        image: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute pairwise likelihood term: -log P^V(yp,yq|x).
        
        For each pixel p and its neighbor q, we compute:
        P^V(yp,yq|x) = exp[-V(yp,yq,x)] / Σ exp[-V(y'p,y'q,x)]
        
        This encourages spatial smoothness.
        """
        B, C, H, W = predictions.shape
        device = predictions.device
        
        # Get predicted class probabilities
        probs = F.softmax(predictions, dim=1)  # [B, C, H, W]
        
        # Create valid mask (ignore index 255)
        valid_mask = (target != self.ignore_index).float()  # [B, H, W]
        
        # ✅ Compute pairwise smoothness loss
        # Penalize disagreement between neighboring pixels
        
        # Horizontal neighbors (left-right)
        diff_h = torch.abs(probs[:, :, :, :-1] - probs[:, :, :, 1:])  # [B, C, H, W-1]
        mask_h = valid_mask[:, :, :-1] * valid_mask[:, :, 1:]  # [B, H, W-1]
        loss_h = (diff_h.sum(dim=1) * mask_h).sum() / (mask_h.sum() + 1e-6)
        
        # Vertical neighbors (top-bottom)
        diff_v = torch.abs(probs[:, :, :-1, :] - probs[:, :, 1:, :])  # [B, C, H-1, W]
        mask_v = valid_mask[:, :-1, :] * valid_mask[:, 1:, :]  # [B, H-1, W]
        loss_v = (diff_v.sum(dim=1) * mask_v).sum() / (mask_v.sum() + 1e-6)
        
        # ✅ Optional: Weight by image gradients (edge-aware smoothness)
        if image is not None:
            # Compute image gradients
            img_gray = image.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            
            grad_h = torch.abs(img_gray[:, :, :, :-1] - img_gray[:, :, :, 1:])
            grad_v = torch.abs(img_gray[:, :, :-1, :] - img_gray[:, :, 1:, :])
            
            # Weight smoothness by inverse of image gradient
            # (less smoothness at edges, more smoothness in flat regions)
            weight_h = torch.exp(-grad_h * 10)  # [B, 1, H, W-1]
            weight_v = torch.exp(-grad_v * 10)  # [B, 1, H-1, W]
            
            loss_h = (diff_h.sum(dim=1, keepdim=True) * weight_h).sum() / (mask_h.sum() + 1e-6)
            loss_v = (diff_v.sum(dim=1, keepdim=True) * weight_v).sum() / (mask_v.sum() + 1e-6)
        
        return loss_h + loss_v


class UnaryLoss(nn.Module):
    """
    Simple unary loss (Cross-Entropy) for Stage 1.
    """
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(predictions, target)
    

class StructuredLoss(nn.Module):
    """
    Structured loss implementing Equation 5 from the paper.
    Combines unary loss, pairwise smoothness, and structured hinge loss.
    """
    def __init__(
        self, 
        num_classes: int, 
        margin: float = 1.0,
        ignore_index: int = 255
    ):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self, 
        unary: torch.Tensor, 
        crf_output: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute structured loss.
        
        Args:
            unary: Unary predictions [B, C, H, W]
            crf_output: CRF refined predictions [B, C, H, W]
            target: Ground truth labels [B, H, W]
        
        Returns:
            Total structured loss
        """
        # Unary loss (standard cross-entropy)
        unary_loss = self.ce_loss(unary, target)
        
        # Pairwise smoothness loss
        pairwise_loss = self._pairwise_loss(crf_output)
        
        # Structured hinge loss
        structured_loss = self._structured_hinge_loss(unary, crf_output, target)
        
        # Combined loss with weights from the paper
        total_loss = unary_loss + 0.1 * pairwise_loss + 0.5 * structured_loss
        
        return total_loss
    
    def _pairwise_loss(self, output: torch.Tensor) -> torch.Tensor:
        """
        Encourage spatial smoothness in predictions.
        Penalizes large gradients in the output.
        """
        # Compute horizontal and vertical gradients
        dx = output[:, :, :, 1:] - output[:, :, :, :-1]
        dy = output[:, :, 1:, :] - output[:, :, :-1, :]
        
        # L1 smoothness loss
        return (dx.abs().mean() + dy.abs().mean())
    
    def _structured_hinge_loss(
        self, 
        unary: torch.Tensor, 
        crf_output: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Structured SVM loss (Equation 5 from paper).
        L(y, ŷ) = max(0, Δ(y, ŷ) + E(x, ŷ) - E(x, y))
        """
        B, C, H, W = unary.shape
        
        # Create mask for valid pixels (ignore background)
        mask = (target != self.ignore_index).float()
        num_valid = mask.sum() + 1e-6
        
        # Ground truth energy: E(x, y)
        target_clamped = torch.clamp(target, 0, C - 1)  # Clamp to valid range
        target_one_hot = F.one_hot(target_clamped, num_classes=C).permute(0, 3, 1, 2).float()
        E_gt = -(unary * target_one_hot * mask.unsqueeze(1)).sum() / num_valid
        
        # Predicted energy: E(x, ŷ)
        E_pred = -(unary * crf_output * mask.unsqueeze(1)).sum() / num_valid
        
        # Hamming distance: Δ(y, ŷ)
        pred_labels = crf_output.argmax(dim=1)
        delta = ((pred_labels != target).float() * mask).sum() / num_valid
        
        # Hinge loss: max(0, margin + delta + E_pred - E_gt)
        loss = torch.clamp(self.margin + delta + E_pred - E_gt, min=0)
        
        return loss


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
