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
    Differentiable Dense CRF layer for pairwise potentials.
    Implements mean-field inference as described in the paper.
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
        
        # Gaussian kernel parameters (learnable)
        self.pos_xy_std = nn.Parameter(torch.tensor(pos_xy_std))
        self.pos_w = nn.Parameter(torch.tensor(pos_w))
        self.bilateral_xy_std = nn.Parameter(torch.tensor(bilateral_xy_std))
        self.bilateral_rgb_std = nn.Parameter(torch.tensor(bilateral_rgb_std))
        self.bilateral_w = nn.Parameter(torch.tensor(bilateral_w))
        
        # Compatibility transform (learnable)
        self.compatibility = nn.Parameter(
            torch.eye(num_classes) * -1.0 + torch.ones(num_classes, num_classes)
        )
        
    def _spatial_features(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create spatial coordinate features."""
        y_coords = torch.arange(height, dtype=torch.float32, device=device)
        x_coords = torch.arange(width, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        return torch.stack([xx, yy], dim=0)  # [2, H, W]
    
    def _gaussian_kernel(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        std: torch.Tensor
    ) -> torch.Tensor:
        """Compute Gaussian kernel between feature vectors."""
        # features: [D, N]
        diff = features1.unsqueeze(2) - features2.unsqueeze(1)  # [D, N, N]
        dist_sq = (diff ** 2).sum(dim=0)  # [N, N]
        return torch.exp(-dist_sq / (2 * std ** 2))
    
    def _message_passing(
        self,
        Q: torch.Tensor,
        spatial_kernel: torch.Tensor,
        bilateral_kernel: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform message passing step.
        
        Args:
            Q: Current Q distribution [B, C, H, W]
            spatial_kernel: Spatial Gaussian kernel [H*W, H*W]
            bilateral_kernel: Bilateral Gaussian kernel [H*W, H*W]
        Returns:
            Messages [B, C, H, W]
        """
        B, C, H, W = Q.shape
        Q_flat = Q.view(B, C, -1)  # [B, C, H*W]
        
        # Spatial message
        spatial_msg = torch.matmul(Q_flat, spatial_kernel)  # [B, C, H*W]
        spatial_msg = spatial_msg * self.pos_w
        
        # Bilateral message
        bilateral_msg = torch.matmul(Q_flat, bilateral_kernel)  # [B, C, H*W]
        bilateral_msg = bilateral_msg * self.bilateral_w
        
        # Combine messages
        messages = spatial_msg + bilateral_msg
        return messages.view(B, C, H, W)
    
    def forward(
        self,
        unary: torch.Tensor,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform mean-field inference.
        
        Args:
            unary: Unary potentials [B, C, H, W]
            image: Input image [B, 3, H, W] for bilateral filtering
        Returns:
            Refined predictions [B, C, H, W]
        """
        B, C, H, W = unary.shape
        device = unary.device
        
        # Upsample unary to match image size if needed
        if unary.shape[2:] != image.shape[2:]:
            unary = F.interpolate(unary, size=image.shape[2:], mode='bilinear', align_corners=False)
            H, W = image.shape[2:]
        
        # Initialize Q with softmax of unary potentials
        Q = F.softmax(unary, dim=1)
        
        # Prepare features for kernels (simplified for efficiency)
        # In practice, use permutohedral lattice for efficiency
        spatial_coords = self._spatial_features(H, W, device)  # [2, H, W]
        
        # For efficiency, we'll use a simplified version with local neighborhoods
        # Full implementation would use permutohedral lattice
        
        # Mean-field iterations
        for iteration in range(self.num_iterations):
            # Save current Q
            Q_prev = Q.clone()
            
            # Apply compatibility transform
            Q_compat = torch.einsum('bc,bchw->bchw', self.compatibility, Q)
            
            # Message passing (simplified - using convolutions for local neighborhoods)
            # Spatial filtering
            spatial_filter = self._create_spatial_filter(device)
            spatial_msg = F.conv2d(
                Q_compat,
                spatial_filter,
                padding=spatial_filter.shape[-1] // 2,
                groups=C
            ) * self.pos_w
            
            # Bilateral filtering (simplified)
            bilateral_msg = self._bilateral_filter(Q_compat, image) * self.bilateral_w
            
            # Combine messages
            messages = spatial_msg + bilateral_msg
            
            # Update Q
            Q = unary - messages
            Q = F.softmax(Q, dim=1)
            
        return Q
    
    def _create_spatial_filter(self, device: torch.device) -> torch.Tensor:
        """Create Gaussian spatial filter."""
        kernel_size = 5
        sigma = self.pos_xy_std
        
        # Create 1D Gaussian
        x = torch.arange(kernel_size, dtype=torch.float32, device=device)
        x = x - kernel_size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # Create 2D Gaussian
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        
        # Expand for all channels
        kernel = gauss_2d.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(self.num_classes, 1, 1, 1)
        
        return kernel
    
    def _bilateral_filter(self, Q: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Simplified bilateral filtering."""
        # This is a simplified version - full implementation would use
        # permutohedral lattice for efficiency
        B, C, H, W = Q.shape
        
        # Use depthwise separable convolution as approximation
        spatial_kernel = self._create_spatial_filter(image.device)
        
        # Apply spatial filtering
        filtered = F.conv2d(
            Q,
            spatial_kernel,
            padding=spatial_kernel.shape[-1] // 2,
            groups=C
        )
        
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