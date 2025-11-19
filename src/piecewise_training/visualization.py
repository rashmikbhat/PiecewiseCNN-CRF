
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
from tabulate import tabulate


class ComprehensiveVisualizer:
    """
    Complete visualization suite for semantic segmentation training.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class {i}' for i in range(num_classes)]
        self.colors = plt.cm.get_cmap('tab20', num_classes)
    
    def generate_full_report(
        self,
        history: Dict,
        final_metrics: Dict,
        confusion_matrix: np.ndarray,
        sample_predictions: List[Dict],
        save_dir: str = 'results'
    ):
        """
        Generate complete training report with all visualizations.
        
        Args:
            history: Training history from piecewise training
            final_metrics: Final evaluation metrics
            confusion_matrix: Confusion matrix from validation
            sample_predictions: List of dicts with 'image', 'pred', 'gt'
            save_dir: Directory to save all outputs
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE TRAINING REPORT")
        print("="*70)
        
        # 1. Training curves
        print("📊 Plotting training curves...")
        self.plot_training_curves(
            history,
            save_path=save_path / 'training_curves.png'
        )
        
        # 2. Metrics table
        print("📋 Generating metrics table...")
        self.generate_metrics_table(
            history,
            final_metrics,
            save_path=save_path / 'metrics_summary.txt'
        )
        
        # 3. Confusion matrix
        print("🔢 Plotting confusion matrix...")
        self.plot_confusion_matrix(
            confusion_matrix,
            save_path=save_path / 'confusion_matrix.png'
        )
        
        # 4. Per-class IoU
        print("📊 Plotting per-class IoU...")
        iou_per_class = self._compute_iou_from_cm(confusion_matrix)
        self.plot_per_class_iou(
            iou_per_class,
            save_path=save_path / 'per_class_iou.png'
        )
        
        # 5. Sample predictions
        print("🖼️  Visualizing sample predictions...")
        self.visualize_predictions_grid(
            sample_predictions,
            save_path=save_path / 'sample_predictions.png'
        )
        
        # 6. CRF comparison
        print("🔍 Generating CRF comparison...")
        self.plot_crf_comparison(
            sample_predictions,
            save_path=save_path / 'crf_comparison.png'
        )
        
        print(f"\n✅ Report generated successfully in: {save_path}")
        print("="*70 + "\n")
    
    def plot_training_curves(self, history: Dict, save_path: str = None):
        """Enhanced training curves with all stages."""
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        stages = ['stage1', 'stage2', 'stage3']
        stage_names = ['Stage 1: Unary Network', 'Stage 2: CRF Parameters', 'Stage 3: Joint Fine-tuning']
        
        for idx, (stage, name) in enumerate(zip(stages, stage_names)):
            if stage not in history:
                continue
            
            stage_history = history[stage]
            
            # Loss subplot
            ax_loss = fig.add_subplot(gs[0, idx])
            if 'train_loss' in stage_history:
                epochs = range(1, len(stage_history['train_loss']) + 1)
                ax_loss.plot(epochs, stage_history['train_loss'], 
                           label='Train', marker='o', linewidth=2)
            if 'val_loss' in stage_history:
                epochs = range(1, len(stage_history['val_loss']) + 1)
                ax_loss.plot(epochs, stage_history['val_loss'], 
                           label='Val', marker='s', linewidth=2)
            ax_loss.set_title(f'{name}\nLoss', fontweight='bold')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
            
            # mIoU subplot
            ax_miou = fig.add_subplot(gs[1, idx])
            if 'val_miou' in stage_history:
                epochs = range(1, len(stage_history['val_miou']) + 1)
                ax_miou.plot(epochs, stage_history['val_miou'], 
                           label='Val mIoU', marker='o', linewidth=2, color='green')
                ax_miou.set_title('mIoU', fontweight='bold')
                ax_miou.set_xlabel('Epoch')
                ax_miou.set_ylabel('mIoU')
                ax_miou.set_ylim(0, 1.0)
                ax_miou.legend()
                ax_miou.grid(True, alpha=0.3)
            
            # Pixel accuracy subplot (if available)
            ax_acc = fig.add_subplot(gs[2, idx])
            if 'val_acc' in stage_history:
                epochs = range(1, len(stage_history['val_acc']) + 1)
                ax_acc.plot(epochs, stage_history['val_acc'], 
                          label='Val Acc', marker='o', linewidth=2, color='orange')
                ax_acc.set_title('Pixel Accuracy', fontweight='bold')
                ax_acc.set_xlabel('Epoch')
                ax_acc.set_ylabel('Accuracy')
                ax_acc.set_ylim(0, 1.0)
                ax_acc.legend()
                ax_acc.grid(True, alpha=0.3)
        
        plt.suptitle('Piecewise Training Progress', fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_metrics_table(
        self,
        history: Dict,
        final_metrics: Dict,
        save_path: str = None
    ):
        """Generate formatted metrics table."""
        # Collect metrics from each stage
        rows = []
        
        for stage in ['stage1', 'stage2', 'stage3']:
            if stage not in history:
                continue
            
            stage_history = history[stage]
            row = {'Stage': stage.replace('stage', 'Stage ')}
            
            # Get final epoch metrics
            if 'train_loss' in stage_history and stage_history['train_loss']:
                row['Train Loss'] = stage_history['train_loss'][-1]
            if 'val_loss' in stage_history and stage_history['val_loss']:
                row['Val Loss'] = stage_history['val_loss'][-1]
            if 'val_miou' in stage_history and stage_history['val_miou']:
                row['Val mIoU'] = stage_history['val_miou'][-1]
            if 'val_acc' in stage_history and stage_history['val_acc']:
                row['Val Acc'] = stage_history['val_acc'][-1]
            
            rows.append(row)
        
        # Add final metrics
        if final_metrics:
            final_row = {'Stage': 'Final Test'}
            final_row.update(final_metrics)
            rows.append(final_row)
        
        df = pd.DataFrame(rows)
        
        # Format table
        table_str = "\n" + "="*80 + "\n"
        table_str += "TRAINING METRICS SUMMARY\n"
        table_str += "="*80 + "\n"
        table_str += tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f', showindex=False)
        table_str += "\n" + "="*80 + "\n"
        
        print(table_str)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(table_str)
            df.to_csv(str(save_path).replace('.txt', '.csv'), index=False)
            print(f"   Saved to: {save_path}")
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save_path: str = None,
        normalize: bool = True
    ):
        """Plot confusion matrix heatmap."""
        if normalize:
            cm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-10)
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            cm = confusion_matrix
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(14, 12))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'},
            square=True
        )
        
        plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
        plt.ylabel('True Class', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_per_class_iou(self, iou_per_class: np.ndarray, save_path: str = None):
        """Plot per-class IoU as bar chart."""
        plt.figure(figsize=(16, 6))
        
        # Filter valid classes
        valid_mask = ~np.isnan(iou_per_class)
        valid_iou = iou_per_class[valid_mask]
        valid_names = [self.class_names[i] for i in range(len(self.class_names)) if valid_mask[i]]
        
        # Create bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(valid_iou)))
        bars = plt.bar(range(len(valid_iou)), valid_iou, color=colors, edgecolor='black', linewidth=1.5)
        
        # Customize
        plt.xlabel('Class', fontsize=12, fontweight='bold')
        plt.ylabel('IoU', fontsize=12, fontweight='bold')
        plt.title('Per-Class IoU Performance', fontsize=14, fontweight='bold', pad=15)
        plt.xticks(range(len(valid_iou)), valid_names, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, iou in zip(bars, valid_iou):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{iou:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add mean line
        mean_iou = np.nanmean(iou_per_class)
        plt.axhline(y=mean_iou, color='red', linestyle='--', linewidth=2.5,
                   label=f'Mean IoU: {mean_iou:.3f}', alpha=0.8)
        plt.legend(fontsize=11, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_predictions_grid(
        self,
        sample_predictions: List[Dict],
        save_path: str = None,
        max_samples: int = 6
    ):
        """Visualize grid of predictions."""
        n_samples = min(len(sample_predictions), max_samples)
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, sample in enumerate(sample_predictions[:n_samples]):
            image = sample['image']
            pred = sample['pred']
            gt = sample['gt']
            
            # Image
            img_np = image.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title('Input Image', fontweight='bold')
            axes[idx, 0].axis('off')
            
            # Ground truth
            gt_colored = self.colors(gt.cpu().numpy())[:, :, :3]
            axes[idx, 1].imshow(gt_colored)
            axes[idx, 1].set_title('Ground Truth', fontweight='bold')
            axes[idx, 1].axis('off')
            
            # Prediction
            pred_colored = self.colors(pred.cpu().numpy())[:, :, :3]
            axes[idx, 2].imshow(pred_colored)
            
            # Compute IoU
            iou = self._compute_sample_iou(pred, gt)
            axes[idx, 2].set_title(f'Prediction (mIoU: {iou:.3f})', fontweight='bold')
            axes[idx, 2].axis('off')
        
        plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_crf_comparison(
        self,
        sample_predictions: List[Dict],
        save_path: str = None,
        max_samples: int = 3
    ):
        """Compare unary vs CRF predictions."""
        n_samples = min(len(sample_predictions), max_samples)
        fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, sample in enumerate(sample_predictions[:n_samples]):
            if 'unary_pred' not in sample or 'crf_pred' not in sample:
                continue
            
            image = sample['image']
            gt = sample['gt']
            unary_pred = sample['unary_pred']
            crf_pred = sample['crf_pred']
            
            # Image
            img_np = image.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title('Input', fontweight='bold')
            axes[idx, 0].axis('off')
            
            # Ground truth
            gt_colored = self.colors(gt.cpu().numpy())[:, :, :3]
            axes[idx, 1].imshow(gt_colored)
            axes[idx, 1].set_title('Ground Truth', fontweight='bold')
            axes[idx, 1].axis('off')
            
            # Unary
            unary_colored = self.colors(unary_pred.cpu().numpy())[:, :, :3]
            axes[idx, 2].imshow(unary_colored)
            unary_iou = self._compute_sample_iou(unary_pred, gt)
            axes[idx, 2].set_title(f'Unary (CNN)\nmIoU: {unary_iou:.3f}', fontweight='bold')
            axes[idx, 2].axis('off')
            
            # CRF
            crf_colored = self.colors(crf_pred.cpu().numpy())[:, :, :3]
            axes[idx, 3].imshow(crf_colored)
            crf_iou = self._compute_sample_iou(crf_pred, gt)
            improvement = crf_iou - unary_iou
            color = 'green' if improvement > 0 else 'red'
            axes[idx, 3].set_title(
                f'CRF Refined\nmIoU: {crf_iou:.3f} ({improvement:+.3f})',
                fontweight='bold',
                color=color
            )
            axes[idx, 3].axis('off')
        
        plt.suptitle('CRF Refinement Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _compute_iou_from_cm(self, confusion_matrix: np.ndarray) -> np.ndarray:
        """Compute IoU from confusion matrix."""
        iou_per_class = []
        for i in range(self.num_classes):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            
            if tp + fp + fn == 0:
                iou = float('nan')
            else:
                iou = tp / (tp + fp + fn)
            
            iou_per_class.append(iou)
        
        return np.array(iou_per_class)
    
    def _compute_sample_iou(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute mIoU for a single sample."""
        ious = []
        pred_np = pred.cpu().numpy()
        gt_np = gt.cpu().numpy()
        
        for cls in range(self.num_classes):
            pred_mask = pred_np == cls
            gt_mask = gt_np == cls
            
            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()
            
            if union > 0:
                ious.append(intersection / union)
        
        return np.mean(ious) if ious else 0.0


# Convenience function
def create_full_report(
    history: Dict,
    model,
    val_loader,
    device,
    num_classes: int,
    class_names: List[str],
    save_dir: str = 'results'
):
    """
    One-line function to generate complete report.
    
    Usage:
        create_full_report(
            history=history,
            model=model,
            val_loader=val_loader,
            device=device,
            num_classes=21,
            class_names=PASCAL_VOC_CLASSES,
            save_dir='training_results'
        )
    """
    from .metrics import SegmentationMetrics
    
    visualizer = ComprehensiveVisualizer(num_classes, class_names)
    
    # Collect validation metrics
    print("Evaluating model on validation set...")
    metrics_calculator = SegmentationMetrics(num_classes)
    confusion_matrix = np.zeros((num_classes, num_classes))
    sample_predictions = []
    
    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            unary_output, crf_output = model(images, apply_crf=True)
            unary_pred = unary_output.argmax(1)
            crf_pred = crf_output.argmax(1) if crf_output is not None else unary_pred
            
            # Update metrics
            metrics_calculator.update(crf_pred, labels)
            
            # Update confusion matrix
            for i in range(num_classes):
                for j in range(num_classes):
                    mask = labels != 255
                    confusion_matrix[i, j] += ((labels[mask] == i) & (crf_pred[mask] == j)).sum().item()
            
            # Collect samples for visualization
            if idx < 10:  # Save first 10 batches
                for b in range(min(3, images.shape[0])):
                    sample_predictions.append({
                        'image': images[b],
                        'gt': labels[b],
                        'unary_pred': unary_pred[b],
                        'crf_pred': crf_pred[b],
                        'pred': crf_pred[b]
                    })