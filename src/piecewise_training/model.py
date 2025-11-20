import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class DeepLabV1Backbone(nn.Module):
    """
    DeepLab-style backbone with dilated convolutions.
    Based on VGG-16 architecture.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        
        # VGG-16 style encoder with dilated convolutions
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
            
            # Block 4 (with dilation)
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            
            # Block 5 (with dilation)
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=12, dilation=12),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(1024, num_classes, 1)
        )

        # ✅ ADD PROPER INITIALIZATION
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights properly to prevent dead neurons.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # ✅ CRITICAL: Initialize final layer with small weights
        # This prevents the model from being too confident initially
        final_conv = self.classifier[-1]
        nn.init.normal_(final_conv.weight, mean=0, std=0.01)
        nn.init.constant_(final_conv.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x



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
        self.num_iterations = num_iterations
        
        # ✅ Learnable CRF parameters
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
        
        # ✅ ADAPTIVE ITERATIONS: Use fewer during training
        if self.training:
            iterations = max(3, self.num_iterations // 2)
        else:
            iterations = self.num_iterations
        
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
            Q = self._message_passing_step(Q, image, unary)
        
        return Q
    
    def _message_passing_step(
        self,
        Q: torch.Tensor,
        image: torch.Tensor,
        unary: torch.Tensor
    ) -> torch.Tensor:
        """Single mean-field iteration."""
        # Spatial smoothness
        spatial_out = self._spatial_filter(Q)
        
        # Bilateral smoothness
        bilateral_out = self._bilateral_filter(Q, image)
        
        # Combine messages
        messages = self.pos_w * spatial_out + self.bilateral_w * bilateral_out
        
        # Compatibility transform
        messages = -messages
        
        # Add unary potentials
        Q_new = unary + messages
        
        # Normalize
        Q_new = F.softmax(Q_new, dim=1)
        
        return Q_new
    
    def _spatial_filter(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Spatial Gaussian filter using average pooling.
        """
        kernel_size = int(self.pos_xy_std.item() * 2) + 1
        kernel_size = max(3, min(kernel_size, 7))  # Clamp to [3, 7]
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Ensure odd
        
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
        
        ✅ FIXED: Proper dimension handling
        """
        B, C, H, W = Q.shape
        
        # ✅ Simple edge-aware smoothing using convolution
        # Compute image gradients
        kernel_x = torch.tensor([[-1, 0, 1]], dtype=image.dtype, device=image.device).view(1, 1, 1, 3)
        kernel_y = torch.tensor([[-1], [0], [1]], dtype=image.dtype, device=image.device).view(1, 1, 3, 1)
        
        # Compute gradients for each channel
        grad_x = F.conv2d(image, kernel_x.repeat(3, 1, 1, 1), padding=(0, 1), groups=3)
        grad_y = F.conv2d(image, kernel_y.repeat(3, 1, 1, 1), padding=(1, 0), groups=3)
        
        # Edge magnitude
        edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8).sum(dim=1, keepdim=True)
        
        # Edge weight (high at edges = less smoothing)
        edge_weight = torch.exp(-edge_mag / self.bilateral_rgb_std)
        
        # ✅ Apply Gaussian smoothing weighted by edge information
        # Use depthwise convolution for efficiency
        kernel_size = 5
        padding = kernel_size // 2
        
        # Create Gaussian kernel
        sigma = self.bilateral_xy_std
        x = torch.arange(kernel_size, dtype=Q.dtype, device=Q.device) - kernel_size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # 2D Gaussian kernel
        gauss_2d = gauss_1d.view(-1, 1) * gauss_1d.view(1, -1)
        gauss_2d = gauss_2d.view(1, 1, kernel_size, kernel_size)
        
        # Apply depthwise convolution to each class
        smoothed = F.conv2d(
            Q, 
            gauss_2d.repeat(C, 1, 1, 1), 
            padding=padding, 
            groups=C
        )
        
        # ✅ Blend based on edge weights (preserve edges)
        # Expand edge_weight to match Q dimensions
        edge_weight_expanded = edge_weight.expand_as(Q)
        
        # Mix original and smoothed based on edge strength
        filtered = Q * (1 - edge_weight_expanded) + smoothed * edge_weight_expanded
        
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
        
        # Unary network (CNN)
        self.unary_net = DeepLabV1Backbone(num_classes)
        
        # CRF layer
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