import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import numpy as np
from tqdm import tqdm
from src.piecewise_training.losses import PiecewiseCRFLoss, UnaryLoss


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
        weight_decay: float = 5e-4,
        patience: int = 5,  # ✅ Add early stopping patience
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience  # ✅ Store patience

        if class_weights is not None:
            class_weights = class_weights.to(device)
            print(f"✅ Using class weights (min={class_weights.min():.4f}, max={class_weights.max():.4f})")
        else:
            print("⚠️  WARNING: No class weights provided - may struggle with imbalanced data!")
        
        self.unary_loss = UnaryLoss(ignore_index=255,class_weights=class_weights)  # Stage 1
        self.piecewise_loss = PiecewiseCRFLoss(  # Stage 2 & 3
            num_classes=num_classes,
            unary_weight=1.0,
            pairwise_weight=0.1,
            ignore_index=255,
            class_weights=class_weights
        )
        
        # Alias for structured loss (same as piecewise loss)
        self.structured_criterion = self.piecewise_loss
        
        # For validation (simple cross-entropy)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=255)

    def _early_stopping_check(
        self,
        val_miou_history: list,
        patience: int = 5
    ) -> bool:
        """
        Check if training should stop early.
        
        Returns:
            True if should stop, False otherwise
        """
        if len(val_miou_history) < patience + 1:
            return False
        
        # Check if mIoU hasn't improved in last 'patience' epochs
        recent_miou = val_miou_history[-patience:]
        best_recent = max(recent_miou)
        best_overall = max(val_miou_history[:-patience])
        
        if best_recent <= best_overall:
            print(f"\n⚠️ Early stopping: No improvement in {patience} epochs")
            return True
        
        return False
        
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
        """Stage 1: Train unary network with UnaryLoss (Cross-Entropy)."""
        print("Stage 1: Training Unary Network")
        
        if hasattr(self.model, 'crf'):
            for param in self.model.crf.parameters():
                param.requires_grad = False
        
        optimizer = self._get_optimizer(
            self.model.unary_net.parameters(),
            lr=1e-4
        )
        
        # ✅ FIX: Add learning rate warmup
        def lr_lambda(epoch):
            if epoch < 5:  # Warmup for first 5 epochs
                return (epoch + 1) / 5
            else:
                return 0.1 ** ((epoch - 5) // 10)  # Decay every 10 epochs
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_miou': [],
            'val_acc': []
        }
        
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
                
                # Compute loss
                loss = self.unary_loss(unary_output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
                history['val_acc'].append(val_metrics['pixel_acc'])
                
                print(f"Epoch {epoch+1}: "
                    f"Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val mIoU={val_metrics['miou']:.4f}, "
                    f"Val Acc={val_metrics['pixel_acc']:.4f}, "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}")  # ✅ Show LR
                
                # Early stopping check
                if self._early_stopping_check(history['val_miou'], patience=self.patience):
                    print(f"Stopping at epoch {epoch+1}")
                    break
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
        """Stage 2: Train CRF parameters with PiecewiseCRFLoss."""
        print("\nStage 2: Training CRF Parameters with Piecewise Loss")
        
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
            lr=1e-3
        )
        
        history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': [],
        'val_acc': []  # ✅ Add this
        }
        
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
                
                # ✅ Use PiecewiseCRFLoss (unary + pairwise terms)
                loss = self.piecewise_loss(
                    unary_output, crf_output, labels, images
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.crf.parameters(), max_norm=1.0)
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
                history['val_acc'].append(val_metrics['pixel_acc'])  # ✅ Add this
                
                print(f"Epoch {epoch+1}: "
                    f"Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val mIoU={val_metrics['miou']:.4f}, "
                    f"Val Acc={val_metrics['pixel_acc']:.4f}")  # ✅ Add this
                
                # ✅ ADD EARLY STOPPING CHECK HERE
                if self._early_stopping_check(history['val_miou'], patience=self.patience):
                    print(f"Stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}")
        
        return history
    
    
    
    def train_stage3_joint(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, list]:
        """Stage 3: Joint fine-tuning with PiecewiseCRFLoss."""
        print("\nStage 3: Joint Fine-tuning with Piecewise Loss")
        
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Optimizer for entire model
        optimizer = self._get_optimizer(
            self.model.parameters(),
            lr=self.learning_rate * 0.01  # Very low learning rate for fine-tuning
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # ✅ Gradient accumulation for effective larger batch size
        accumulation_steps = 2  # Effective batch size = 8 × 2 = 16
        
        history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': [],
        'val_acc': []  # ✅ Add this
        }
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            optimizer.zero_grad()  # ✅ Initialize gradients once per epoch
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass with CRF
                unary_output, crf_output = self.model(images, apply_crf=True)
                
                # ✅ FIXED: Use piecewise loss (not structured_criterion)
                if crf_output is not None:
                    loss = self.piecewise_loss(unary_output, crf_output, labels, images)
                else:
                    loss = self.unary_loss(unary_output, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            # ✅ Handle remaining gradients if batch count not divisible by accumulation_steps
            if len(train_loader) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, use_crf=True)
                history['val_loss'].append(val_metrics['loss'])
                history['val_miou'].append(val_metrics['miou'])
                history['val_acc'].append(val_metrics['pixel_acc'])  # ✅ Add this
                
                print(f"Epoch {epoch+1}: "
                    f"Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val mIoU={val_metrics['miou']:.4f}, "
                    f"Val Acc={val_metrics['pixel_acc']:.4f}")  # ✅ Add this
                
                # ✅ ADD EARLY STOPPING CHECK HERE
                if self._early_stopping_check(history['val_miou'], patience=self.patience):
                    print(f"Stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}")
            
            scheduler.step()
        
        return history
    
    
    
    def validate(
        self,
        val_loader: DataLoader,
        use_crf: bool = True
    ) -> Dict[str, float]:
        """
        Validate the model on validation set.
        
        Args:
            val_loader: Validation data loader
            use_crf: Whether to use CRF refinement
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                unary_output, crf_output = self.model(images, apply_crf=use_crf)
                
                # Use CRF output if available, otherwise unary
                if use_crf and crf_output is not None:
                    output = crf_output
                else:
                    output = unary_output
                
                # Compute loss
                loss = self.criterion(output, labels)
                total_loss += loss.item()
                
                # Get predictions
                predictions = output.argmax(dim=1)  # [B, H, W]
                
                # ✅ FIX: Update confusion matrix correctly
                # Flatten predictions and labels
                pred_flat = predictions.cpu().numpy().flatten()
                label_flat = labels.cpu().numpy().flatten()
                
                # Remove ignore index (255)
                valid_mask = label_flat != 255
                pred_flat = pred_flat[valid_mask]
                label_flat = label_flat[valid_mask]
                
                # Update confusion matrix
                for true_label in range(self.num_classes):
                    for pred_label in range(self.num_classes):
                        mask = (label_flat == true_label) & (pred_flat == pred_label)
                        confusion_matrix[true_label, pred_label] += mask.sum()
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        miou = self._compute_miou(confusion_matrix)
        
        # ✅ FIX: Compute pixel accuracy
        pixel_acc = confusion_matrix.diagonal().sum() / confusion_matrix.sum()
        
        return {
            'loss': avg_loss,
            'miou': miou,
            'pixel_acc': pixel_acc,
            'confusion_matrix': confusion_matrix
        }


    def _compute_miou(self, confusion_matrix: np.ndarray) -> float:
        """
        Compute mean IoU from confusion matrix.
        
        Args:
            confusion_matrix: [num_classes, num_classes]
        
        Returns:
            Mean IoU across all classes
        """
        iou_per_class = []
        
        for i in range(self.num_classes):
            # True positives
            tp = confusion_matrix[i, i]
            
            # False positives + False negatives
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            
            # Compute IoU
            denominator = tp + fp + fn
            
            if denominator > 0:
                iou = tp / denominator
                iou_per_class.append(iou)
            else:
                # ✅ FIX: Skip classes with no samples (don't use NaN)
                continue
        
        # ✅ FIX: Return mean of valid IoUs only
        if len(iou_per_class) > 0:
            return np.mean(iou_per_class)
        else:
            return 0.0
    
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