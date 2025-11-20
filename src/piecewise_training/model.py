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


class DenseCRF(nn.Module):
    """
    Dense CRF layer with learnable parameters.
    Implements mean-field approximation for inference.
    """
    
    def __init__(self, num_classes: int, num_iterations: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        
        # Learnable compatibility transform (Potts model initialization)
        self.compatibility = nn.Parameter(torch.eye(num_classes))
        
        # Learnable kernel weights
        self.spatial_weight = nn.Parameter(torch.tensor(3.0))
        self.bilateral_weight = nn.Parameter(torch.tensor(5.0))
        
    def forward(self, unary: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Perform mean-field inference.
        
        Args:
            unary: Unary potentials [B, C, H, W]
            image: Input image [B, 3, H, W] (for bilateral filtering)
            
        Returns:
            Refined predictions [B, C, H, W]
        """
        B, C, H, W = unary.shape
        assert C == self.num_classes, f"Expected {self.num_classes} classes, got {C}"
        
        # Initialize Q with softmax of unary potentials
        Q = F.softmax(unary, dim=1)  # [B, C, H, W]
        
        # Mean-field iterations
        for iteration in range(self.num_iterations):
            # 1. Message passing
            Q_spatial = self._spatial_message_passing(Q)
            Q_bilateral = self._bilateral_message_passing(Q, image)
            
            # 2. Combine messages with learnable weights
            messages = (
                torch.clamp(self.spatial_weight, min=0) * Q_spatial +
                torch.clamp(self.bilateral_weight, min=0) * Q_bilateral
            )
            
            # 3. Apply compatibility transform
            # Reshape for batch matrix multiplication
            messages_flat = messages.reshape(B, C, H * W)  # [B, C, H*W]
            
            # Apply compatibility: [C, C] x [B, C, H*W] -> [B, C, H*W]
            # Use bmm with broadcasting
            compat_transform = torch.bmm(
                self.compatibility.unsqueeze(0).expand(B, -1, -1),  # [B, C, C]
                messages_flat                                        # [B, C, H*W]
            )
            
            # Reshape back
            compat_transform = compat_transform.reshape(B, C, H, W)
            
            # 4. Update Q with compatibility-transformed messages
            Q = F.softmax(unary - compat_transform, dim=1)
        
        return Q
    
    def _spatial_message_passing(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Spatial Gaussian kernel message passing.
        Approximated with average pooling.
        """
        kernel_size = 5
        padding = kernel_size // 2
        
        # Per-channel average pooling (approximates Gaussian convolution)
        Q_smooth = F.avg_pool2d(
            Q, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )
        
        return Q_smooth
    
    def _bilateral_message_passing(
        self, 
        Q: torch.Tensor, 
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Bilateral filtering (edge-aware smoothing).
        Uses image gradients to preserve edges.
        """
        B, C, H, W = Q.shape
        
        # Convert to grayscale for edge detection
        gray = (
            0.299 * image[:, 0:1] + 
            0.587 * image[:, 1:2] + 
            0.114 * image[:, 2:3]
        )
        
        # Compute edge weights using Sobel filters
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            dtype=image.dtype, 
            device=image.device
        ).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            dtype=image.dtype, 
            device=image.device
        ).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        
        # Edge magnitude
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # Edge-preserving weight (higher at edges)
        edge_weight = torch.exp(-edge_magnitude)
        
        # Apply edge-aware smoothing
        Q_weighted = Q * edge_weight
        Q_bilateral = F.avg_pool2d(Q_weighted, kernel_size=3, stride=1, padding=1)
        
        # Normalize
        norm = F.avg_pool2d(edge_weight, kernel_size=3, stride=1, padding=1) + 1e-6
        Q_bilateral = Q_bilateral / norm
        
        return Q_bilateral


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