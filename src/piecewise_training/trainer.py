import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import numpy as np
from tqdm import tqdm
from src.piecewise_training.losses import StructuredLoss


class PiecewiseTrainer:
    """
    Implements the piecewise training strategy from the paper.
    
    Training proceeds in stages:
    1. Train unary network (CNN) alone
    2. Fix unary network, train CRF parameters
    3. Fine-tune entire model end-to-end
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4
    ):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # ✅ Use different losses for different stages
        self.unary_criterion = nn.CrossEntropyLoss(ignore_index=255)  # Stage 1
        self.structured_criterion = StructuredLoss(num_classes=num_classes)  # Stage 2 & 3
        
    def _get_optimizer(self, parameters, lr: Optional[float] = None):
        """Create optimizer for given parameters."""
        if lr is None:
            lr = self.learning_rate
        return optim.SGD(
            parameters,
            lr=lr,
            momentum=0.9,
            weight_decay=self.weight_decay
        )
    
    
    def train_stage1_unary(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, list]:
        """
        Stage 1: Train only the unary network (CNN).
        """
        print("Stage 1: Training Unary Network")
        
        # Freeze CRF parameters if they exist
        if hasattr(self.model, 'crf'):
            for param in self.model.crf.parameters():
                param.requires_grad = False
        
        # Optimizer for unary network only
        optimizer = self._get_optimizer(self.model.unary_net.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        history = {'train_loss': [], 'val_loss': [], 'val_miou': []}
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass (unary only)
                unary_output, _ = self.model(images, apply_crf=False)
                
                # ✅ Use unary_criterion (not criterion!)
                loss = self.unary_criterion(unary_output, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, use_crf=False)
                history['val_loss'].append(val_metrics['loss'])
                history['val_miou'].append(val_metrics['miou'])
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val mIoU={val_metrics['miou']:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}")
            
            scheduler.step()
        
        return history
    
    
    def train_stage2_crf(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, list]:
        """
        Stage 2: Fix unary network, train CRF parameters with structured loss.
        """
        print("\nStage 2: Training CRF Parameters with Structured Loss")
        
        if not hasattr(self.model, 'crf'):
            print("Model has no CRF, skipping stage 2")
            return {}
        
        # Freeze unary network
        for param in self.model.unary_net.parameters():
            param.requires_grad = False
        
        # Unfreeze CRF parameters
        for param in self.model.crf.parameters():
            param.requires_grad = True
        
        # Optimizer for CRF only
        optimizer = self._get_optimizer(
            self.model.crf.parameters(),
            lr=self.learning_rate * 0.1  # Lower learning rate for CRF
        )
        
        history = {'train_loss': [], 'val_loss': [], 'val_miou': []}
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with CRF
                unary_output, crf_output = self.model(images, apply_crf=True)
                
                # ✅ Use structured loss (not CrossEntropy!)
                loss = self.structured_criterion(unary_output, crf_output, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, use_crf=True)
                history['val_loss'].append(val_metrics['loss'])
                history['val_miou'].append(val_metrics['miou'])
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val mIoU={val_metrics['miou']:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}")
        
        return history
    
    
    def train_stage3_joint(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, list]:
        """
        Stage 3: Fine-tune entire model end-to-end with structured loss.
        """
        print("\nStage 3: Joint Fine-tuning with Structured Loss")
        
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Optimizer for entire model
        optimizer = self._get_optimizer(
            self.model.parameters(),
            lr=self.learning_rate * 0.01  # Very low learning rate for fine-tuning
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        history = {'train_loss': [], 'val_loss': [], 'val_miou': []}
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with CRF
                unary_output, crf_output = self.model(images, apply_crf=True)
                
                # ✅ Use structured loss for joint training
                if crf_output is not None:
                    loss = self.structured_criterion(unary_output, crf_output, labels)
                else:
                    loss = self.unary_criterion(unary_output, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, use_crf=True)
                history['val_loss'].append(val_metrics['loss'])
                history['val_miou'].append(val_metrics['miou'])
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val mIoU={val_metrics['miou']:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}")
            
            scheduler.step()
        
        return history
    
    
    def validate(
        self,
        val_loader: DataLoader,
        use_crf: bool = True
    ) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                unary_output, crf_output = self.model(images, apply_crf=use_crf)
                output = crf_output if (use_crf and crf_output is not None) else unary_output
                
                # ✅ Use unary_criterion for validation
                loss = self.unary_criterion(output, labels)
                total_loss += loss.item()
                
                # Compute confusion matrix
                pred = output.argmax(dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                # Update confusion matrix
                mask = labels_np != 255  # Ignore index
                for i in range(self.num_classes):
                    for j in range(self.num_classes):
                        confusion_matrix[i, j] += np.sum(
                            (labels_np[mask] == i) & (pred[mask] == j)
                        )
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        miou = self._compute_miou(confusion_matrix)
        
        return {'loss': avg_loss, 'miou': miou}
    
    def _compute_miou(self, confusion_matrix: np.ndarray) -> float:
        """Compute mean Intersection over Union."""
        iou_per_class = []
        for i in range(self.num_classes):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            
            denominator = tp + fp + fn
            if denominator > 0:
                iou = tp / denominator
                iou_per_class.append(iou)
        
        return np.mean(iou_per_class) if iou_per_class else 0.0
    
    def train_piecewise(
        self,
        train_loader: DataLoader,
        stage1_epochs: int = 20,
        stage2_epochs: int = 5,
        stage3_epochs: int = 10,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, Dict[str, list]]:
        """
        Complete piecewise training pipeline.
        """
        history = {}
        
        # Stage 1: Train unary network
        history['stage1'] = self.train_stage1_unary(
            train_loader, stage1_epochs, val_loader
        )
        
        # Stage 2: Train CRF parameters
        if hasattr(self.model, 'crf'):
            history['stage2'] = self.train_stage2_crf(
                train_loader, stage2_epochs, val_loader
            )
        
        # Stage 3: Joint fine-tuning
        history['stage3'] = self.train_stage3_joint(
            train_loader, stage3_epochs, val_loader
        )
        
        return history