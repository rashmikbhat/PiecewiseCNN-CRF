import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import numpy as np
from tqdm import tqdm
from src.piecewise_training.losses import PiecewiseCRFLoss, UnaryLoss
from src.piecewise_training.utils import PolyLRScheduler


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
        """
        Stage 1: Train unary network with poly LR schedule.
        
        Paper settings (Section 3.2):
        - Base LR: 0.001
        - Poly power: 0.9
        - Optimizer: SGD with momentum 0.9
        - LR decay: lr = base_lr × (1 - iter/max_iter)^0.9
        """
        print("=" * 70)
        print("Stage 1: Training Unary Network (Following CVPR 2016 Paper)")
        print("=" * 70)
        print(f"   Base LR: {self.learning_rate}")
        print(f"   Optimizer: SGD with momentum 0.9")
        print(f"   LR Schedule: Poly (power=0.9)")
        print(f"   Epochs: {num_epochs}")
        print()
        
        # Freeze CRF if it exists
        if hasattr(self.model, 'crf'):
            for param in self.model.crf.parameters():
                param.requires_grad = False
        
        # ✅ Create SGD optimizer (as per paper)
        optimizer = self._get_optimizer(self.model.unary_net.parameters())
        
        # ✅ Create poly scheduler (CRITICAL: per-iteration, not per-epoch!)
        max_iterations = num_epochs * len(train_loader)
        scheduler = PolyLRScheduler(
            optimizer,
            max_iterations=max_iterations,
            power=0.9
        )
        
        print(f"   Total iterations: {max_iterations}")
        print(f"   Iterations per epoch: {len(train_loader)}")
        print()
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_miou': [],
            'val_acc': [],
            'lr': []
        }
        
        best_miou = 0.0
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass (unary only)
                unary_output, _ = self.model(images, apply_crf=False)
                
                # ✅ Use unary loss (Equation 8 from paper)
                loss = self.unary_loss(unary_output, labels)
                
                # Backward pass
                loss.backward()
                
                # ✅ Gradient clipping (helps stability)
                torch.nn.utils.clip_grad_norm_(self.model.unary_net.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # ✅ CRITICAL: Step poly scheduler EVERY iteration!
                scheduler.step()
                
                train_loss += loss.item()
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.6f}'
                })
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, use_crf=False)
                history['val_loss'].append(val_metrics['loss'])
                history['val_miou'].append(val_metrics['miou'])
                history['val_acc'].append(val_metrics['pixel_acc'])
                
                # ✅ Single-line compact format
                print(f"Epoch {epoch+1}/{num_epochs}: "
                    f"Train Loss={avg_train_loss:.4f} | "
                    f"Val Loss={val_metrics['loss']:.4f} | "
                    f"Val mIoU={val_metrics['miou']:.4f} | "
                    f"Val Acc={val_metrics.get('accuracy', 0):.4f} | "
                    f"LR={scheduler.get_last_lr()[0]:.6f}")
                
                # Track best model
                if val_metrics['miou'] > best_miou:
                    best_miou = val_metrics['miou']
                    #print(f"   New best mIoU: {best_miou:.4f}")
                
                print()
            else:
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}")
        
        print(f"Stage 1 Complete! Best Val mIoU: {best_miou:.4f}")
        print("=" * 70)
        print()
        
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
        lr=self.learning_rate * 0.1  # Lower learning rate for CRF
        )
        
        # ✅ Use plateau scheduler for Stage 2 (CRF is sensitive)
        scheduler = self._get_scheduler(optimizer, num_epochs, mode='plateau')
        
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
                
                # ✅ Step poly scheduler EVERY iteration
                scheduler.step()

                train_loss += loss.item()
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.6f}'
                })
                    
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, use_crf=True)
                history['val_loss'].append(val_metrics['loss'])
                history['val_miou'].append(val_metrics['miou'])
                history['val_acc'].append(val_metrics['pixel_acc'])
                
                # ✅ Single-line compact format
                print(f"Epoch {epoch+1}/{num_epochs}: "
                    f"Train Loss={avg_train_loss:.4f} | "
                    f"Val Loss={val_metrics['loss']:.4f} | "
                    f"Val mIoU={val_metrics['miou']:.4f} | "
                    f"Val Acc={val_metrics.get('accuracy', 0):.4f} | "
                    f"LR={scheduler.get_last_lr()[0]:.6f}")
                
                if val_metrics['miou'] > best_miou:
                    best_miou = val_metrics['miou']
                    #print(f"   ✅ New best mIoU: {best_miou:.4f}")
                
                # Early stopping
                if self._early_stopping_check(history['val_miou'], patience=self.patience):
                    print(f"Stopping at epoch {epoch+1}")
                    break
                
                print()
            else:
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}")
        
        print(f"Stage 2 Complete! Best Val mIoU: {best_miou:.4f}")
        print("=" * 70)
        print()
        
        return history
    
    
    
    def train_stage3_joint(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, list]:
        """
        Stage 3: Joint fine-tuning with poly LR schedule.
        
        Paper settings:
        - Base LR: 0.00001 (100x lower than Stage 1)
        - Poly power: 0.9
        - Optimizer: SGD with momentum 0.9
        """
        print("=" * 70)
        print("Stage 3: Joint Fine-tuning (Following CVPR 2016 Paper)")
        print("=" * 70)
        
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        # ✅ Very low learning rate for fine-tuning
        finetune_lr = self.learning_rate * 0.01  # 0.00001 if base is 0.001
        optimizer = self._get_optimizer(
            self.model.parameters(),
            lr=finetune_lr
        )
        
        # ✅ Use poly scheduler
        max_iterations = num_epochs * len(train_loader)
        scheduler = PolyLRScheduler(
            optimizer,
            max_iterations=max_iterations,
            power=0.9
        )
        
        print(f"   Base LR: {finetune_lr}")
        print(f"   Optimizer: SGD with momentum 0.9")
        print(f"   LR Schedule: Poly (power=0.9)")
        print(f"   Total iterations: {max_iterations}")
        print()
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_miou': [],
            'val_acc': [],
            'lr': []
        }
        
        best_miou = 0.0
        
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
                
                # ✅ Use piecewise loss
                if crf_output is not None:
                    loss = self.piecewise_loss(unary_output, crf_output, labels, images)
                else:
                    loss = self.unary_loss(unary_output, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # ✅ Step poly scheduler EVERY iteration
                scheduler.step()
                
                train_loss += loss.item()
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.6f}'
                })
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, use_crf=True)
                history['val_loss'].append(val_metrics['loss'])
                history['val_miou'].append(val_metrics['miou'])
                history['val_acc'].append(val_metrics['pixel_acc'])
                
                # ✅ Single-line compact format
                print(f"Epoch {epoch+1}/{num_epochs}: "
                    f"Loss={avg_train_loss:.4f} | "
                    f"Val Loss={val_metrics['loss']:.4f} | "
                    f"Val mIoU={val_metrics['miou']:.4f} | "
                    f"Val Acc={val_metrics.get('accuracy', 0):.4f} | "
                    f"LR={scheduler.get_last_lr()[0]:.6f}")
                
                if val_metrics['miou'] > best_miou:
                    best_miou = val_metrics['miou']
                    #print(f"   ✅ New best mIoU: {best_miou:.4f}")
                
                # Early stopping
                if self._early_stopping_check(history['val_miou'], patience=self.patience):
                    print(f"Stopping at epoch {epoch+1}")
                    break
                
                print()
            else:
                print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}")
            
        print(f"Stage 3 Complete! Best Val mIoU: {best_miou:.4f}")
        print("=" * 70)
        print()
        
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

         # ✅ ADD: Pixel accuracy tracking
        correct_pixels = 0
        total_pixels = 0
        
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

                # ✅ ADD: Compute pixel accuracy
                valid_mask = (labels != 255)
                correct_pixels += ((predictions == labels) & valid_mask).sum().item()
                total_pixels += valid_mask.sum().item()
                
                # Update confusion matrix for mIoU
                for pred, label in zip(predictions.cpu().numpy().flatten(), 
                                    labels.cpu().numpy().flatten()):
                    if label != 255:  # Ignore background
                        confusion_matrix[label, pred] += 1
        
        avg_loss = total_loss / len(val_loader)
        miou = self._compute_miou(confusion_matrix)
        
        # ✅ ADD: Compute pixel accuracy
        pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'miou': miou,
            'pixel_acc': pixel_accuracy,  # ✅ ADD THIS
            'confusion_matrix':confusion_matrix
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