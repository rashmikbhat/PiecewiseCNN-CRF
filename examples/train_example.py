import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/user/project/src')

from src.piecewise_training.model import PiecewiseTrainedModel
from src.piecewise_training.trainer import PiecewiseTrainer
from src.piecewise_training.dataset import SegmentationDataset, RandomHorizontalFlip
from src.piecewise_training.utils import plot_training_history


def main():
    # Configuration
    num_classes = 21  # Pascal VOC
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = PiecewiseTrainedModel(
        num_classes=num_classes,
        crf_iterations=10,
        use_crf=True
    )
    
    # Create datasets
    train_dataset = SegmentationDataset(
        image_dir='/path/to/train/images',
        label_dir='/path/to/train/labels',
        transform=RandomHorizontalFlip(p=0.5),
        image_size=(512, 512)
    )
    
    val_dataset = SegmentationDataset(
        image_dir='/path/to/val/images',
        label_dir='/path/to/val/labels',
        image_size=(512, 512)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
	num_workers=4,
        pin_memory=True
    )
    
    # Create trainer
    trainer = PiecewiseTrainer(
        model=model,
        device=device,
        num_classes=num_classes,
        learning_rate=1e-3,
        weight_decay=5e-4
    )
    
    # Piecewise training
    print("Starting piecewise training...")
    history = trainer.train_piecewise(
        train_loader=train_loader,
        stage1_epochs=20,  # Train unary network
        stage2_epochs=5,   # Train CRF parameters
        stage3_epochs=10,  # Joint fine-tuning
        val_loader=val_loader
    )
    
    # Plot training history
    plot_training_history(history, save_path='training_history.png')
    
    # Save final model
    torch.save(model.state_dict(), 'piecewise_model_final.pth')
    print("Training completed!")


if __name__ == '__main__':
    main()