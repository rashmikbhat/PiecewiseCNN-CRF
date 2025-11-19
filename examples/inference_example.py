import torch
import numpy as np
from PIL import Image
import sys
sys.path.append('/home/user/project/src')

from src.piecewise_training.model import PiecewiseTrainedModel
from src.piecewise_training.utils import visualize_segmentation


def load_image(image_path: str, size: tuple = (512, 512)) -> torch.Tensor:
    """Load and preprocess image."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size, Image.BILINEAR)
    
    # Convert to tensor
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std
    
    return image.unsqueeze(0)  # Add batch dimension


def inference(
    model_path: str,
    image_path: str,
    num_classes: int = 21,
    use_crf: bool = True
):
    """Run inference on a single image."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = PiecewiseTrainedModel(
        num_classes=num_classes,
        crf_iterations=10,
        use_crf=use_crf
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load image
    image = load_image(image_path).to(device)
    
    # Inference
    with torch.no_grad():
        unary_output, crf_output = model(image, apply_crf=use_crf)
        
        # Get predictions
        if use_crf and crf_output is not None:
            prediction = crf_output.argmax(dim=1).squeeze(0)
        else:
            prediction = unary_output.argmax(dim=1).squeeze(0)
    
    return prediction.cpu()


def main():
    model_path = 'piecewise_model_final.pth'
    image_path = '/path/to/test/image.jpg'
    
    # Run inference without CRF
    print("Running inference without CRF...")
    pred_no_crf = inference(model_path, image_path, use_crf=False)
    
    # Run inference with CRF
    print("Running inference with CRF...")
    pred_with_crf = inference(model_path, image_path, use_crf=True)
    
    # Visualize results
    image = load_image(image_path).squeeze(0)
    
    visualize_segmentation(
        image=image,
        prediction=pred_no_crf,
        ground_truth=pred_with_crf,  # Using CRF result as comparison
        num_classes=21,
        save_path='inference_result.png'
    )
    
    print("Inference completed!")


if __name__ == '__main__':
    main()

