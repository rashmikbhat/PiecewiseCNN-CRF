import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Callable
import os


class SegmentationDataset(Dataset):
    """
    Generic segmentation dataset.
    """
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (512, 512)
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_size = image_size
        
        # Get list of images
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load label
        label_name = img_name.replace('.jpg', '.png')
        label_path = os.path.join(self.label_dir, label_name)
        label = Image.open(label_path)
        
        # Resize
        image = image.resize(self.image_size, Image.BILINEAR)
        label = label.resize(self.image_size, Image.NEAREST)
        
        # Convert to tensors
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(np.array(label)).long()
        
        # Apply transforms
        if self.transform:
            image, label = self.transform(image, label)
        
        # Normalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        return image, label


class RandomHorizontalFlip:
    """Random horizontal flip augmentation."""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        return image, label


class RandomScale:
    """Random scale augmentation."""
    def __init__(self, scale_range: Tuple[float, float] = (0.5, 2.0)):
        self.scale_range = scale_range
    
    def __call__(
        self,
        image: torch.Tensor,
        label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        
        C, H, W = image.shape
        new_H, new_W = int(H * scale), int(W * scale)
        
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        label = torch.nn.functional.interpolate(
            label.unsqueeze(0).unsqueeze(0).float(),
            size=(new_H, new_W),
            mode='nearest'
        ).squeeze(0).squeeze(0).long()
        
        return image, label