import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class DeepLabV1Backbone(nn.Module):
    """
    Simplified DeepLab-like backbone for unary potentials.
    Based on VGG-16 with dilated convolutions.
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        
        # VGG-16 inspired feature extractor with dilated convolutions
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 5 - with dilation
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        
        # Classifier for unary potentials
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(1024, num_classes, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, 3, H, W]
        Returns:
            Unary potentials [B, num_classes, H/8, W/8]
        """
        features = self.features(x)
        unary = self.classifier(features)
        return unary



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DenseCRF(nn.Module):
    """
    Differentiable Dense CRF layer for pairwise potentials.
    Implements mean-field inference as described in the paper.
    
    ✅ OPTIMIZED: Adaptive iterations for faster training
    """
    def __init__(
        self,
        num_classes: int,
        num_iterations: int = 10,
        pos_xy_std: float = 3.0,
        pos_w: float = 3.0,
        bilateral_xy_std: float = 80.0,
        bilateral_rgb_std: float = 13.0,
        bilateral_w: float = 10.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations  # Full iterations for inference
        
        # ✅ Learnable CRF parameters (as per paper)
        self.pos_xy_std = nn.Parameter(torch.tensor(pos_xy_std))
        self.pos_w = nn.Parameter(torch.tensor(pos_w))
        self.bilateral_xy_std = nn.Parameter(torch.tensor(bilateral_xy_std))
        self.bilateral_rgb_std = nn.Parameter(torch.tensor(bilateral_rgb_std))
        self.bilateral_w = nn.Parameter(torch.tensor(bilateral_w))
        
    def forward(
        self,
        unary: torch.Tensor,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Dense CRF refinement.
        
        Args:
            unary: Unary potentials [B, C, H, W]
            image: Input image [B, 3, H, W]
        
        Returns:
            Refined predictions [B, C, H, W]
        """
        B, C, H, W = unary.shape
        
        # ✅ ADAPTIVE ITERATIONS: Use fewer during training for speed
        if self.training:
            iterations = max(3, self.num_iterations // 2)  # Half iterations during training
        else:
            iterations = self.num_iterations  # Full iterations during inference
        
        # Initialize Q with softmax of unary potentials
        Q = F.softmax(unary, dim=1)
        
        # Upsample image to match unary resolution
        if image.shape[2:] != unary.shape[2:]:
            image = F.interpolate(
                image, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Mean-field iterations
        for _ in range(iterations):
            # Message passing
            Q = self._message_passing_step(Q, image, unary)
        
        return Q
    
    def _message_passing_step(
        self,
        Q: torch.Tensor,
        image: torch.Tensor,
        unary: torch.Tensor
    ) -> torch.Tensor:
        """
        Single mean-field iteration.
        
        ✅ OPTIMIZED: Simplified bilateral filtering
        """
        B, C, H, W = Q.shape
        
        # Spatial smoothness (appearance-independent)
        spatial_out = self._spatial_filter(Q)
        
        # Bilateral smoothness (appearance-dependent)
        bilateral_out = self._bilateral_filter(Q, image)
        
        # Combine messages
        messages = self.pos_w * spatial_out + self.bilateral_w * bilateral_out
        
        # Compatibility transform (simple subtraction)
        messages = -messages
        
        # Add unary potentials
        Q_new = unary + messages
        
        # Normalize
        Q_new = F.softmax(Q_new, dim=1)
        
        return Q_new
    
    def _spatial_filter(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Spatial Gaussian filter.
        
        ✅ OPTIMIZED: Use average pooling for speed
        """
        # Simple spatial smoothing using average pooling
        kernel_size = int(self.pos_xy_std.item() * 2) + 1
        kernel_size = max(3, min(kernel_size, 7))  # Clamp to [3, 7]
        
        padding = kernel_size // 2
        
        smoothed = F.avg_pool2d(
            Q, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )
        
        return smoothed
    
    def _bilateral_filter(self, Q: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Bilateral filter (appearance-dependent smoothing).
        
        ✅ OPTIMIZED: Simplified implementation for speed
        """
        B, C, H, W = Q.shape
        
        # Compute image gradients for edge detection
        image_dx = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
        image_dy = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
        
        # Edge weights (high at edges, low in smooth regions)
        edge_weight_x = torch.exp(-image_dx.sum(dim=1, keepdim=True) / self.bilateral_rgb_std)
        edge_weight_y = torch.exp(-image_dy.sum(dim=1, keepdim=True) / self.bilateral_rgb_std)
        
        # Pad to match original size
        edge_weight_x = F.pad(edge_weight_x, (0, 1, 0, 0), value=1.0)
        edge_weight_y = F.pad(edge_weight_y, (0, 0, 0, 1), value=1.0)
        
        # Apply edge-aware smoothing
        Q_x = Q[:, :, :, 1:] * edge_weight_x + Q[:, :, :, :-1] * (1 - edge_weight_x)
        Q_y = Q[:, :, 1:, :] * edge_weight_y + Q[:, :, :-1, :] * (1 - edge_weight_y)
        
        # Pad back
        Q_x = F.pad(Q_x, (1, 0, 0, 0), value=0)
        Q_y = F.pad(Q_y, (0, 0, 1, 0), value=0)
        
        # Combine
        filtered = (Q_x + Q_y) / 2.0
        
        return filtered


class PiecewiseTrainedModel(nn.Module):
    """
    Complete piecewise training model combining CNN and CRF.
    """
    def __init__(
        self,
        num_classes: int,
        crf_iterations: int = 10,
        use_crf: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_crf = use_crf
        
        # Unary potential network
        self.unary_net = DeepLabV1Backbone(num_classes)
        
        # CRF for pairwise potentials
        if use_crf:
            self.crf = DenseCRF(num_classes, num_iterations=crf_iterations)
        
    def forward(
        self,
        image: torch.Tensor,
        apply_crf: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            image: Input image [B, 3, H, W]
            apply_crf: Whether to apply CRF refinement
        Returns:
            Tuple of (unary_output, crf_output)
        """
        # Get unary potentials
        unary = self.unary_net(image)
        
        # Upsample to original resolution
        unary_upsampled = F.interpolate(
            unary,
            size=image.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Apply CRF if requested
        crf_output = None
        if apply_crf and self.use_crf:
            crf_output = self.crf(unary_upsampled, image)
        
        return unary_upsampled, crf_output