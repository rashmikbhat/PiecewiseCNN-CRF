import torch
import pytest
import sys
sys.path.append('/home/user/project/src')

from src.piecewise_training.model import DeepLabV1Backbone, DenseCRF, PiecewiseTrainedModel


def test_deeplab_backbone():
    """Test DeepLab backbone."""
    model = DeepLabV1Backbone(num_classes=21)
    x = torch.randn(2, 3, 512, 512)
    
    output = model(x)
    
    # Check output shape (should be downsampled by 8x)
    assert output.shape == (2, 21, 64, 64)
    print("✓ DeepLab backbone test passed")


def test_dense_crf():
    """Test Dense CRF."""
    crf = DenseCRF(num_classes=21, num_iterations=5)
    unary = torch.randn(2, 21, 64, 64)
    image = torch.randn(2, 3, 512, 512)
    
    output = crf(unary, image)
    
    # Check output shape
    assert output.shape == (2, 21, 512, 512)
    
    # Check that output is a valid probability distribution
    assert torch.allclose(output.sum(dim=1), torch.ones(2, 512, 512), atol=1e-5)
    print("✓ Dense CRF test passed")


def test_piecewise_model():
    """Test complete piecewise model."""
    model = PiecewiseTrainedModel(num_classes=21, use_crf=True)
    image = torch.randn(2, 3, 512, 512)
    
    unary_output, crf_output = model(image, apply_crf=True)
    
    # Check shapes
    assert unary_output.shape == (2, 21, 512, 512)
    assert crf_output.shape == (2, 21, 512, 512)
    
    # Test without CRF
    unary_only, crf_none = model(image, apply_crf=False)
    assert crf_none is None
    print("✓ Piecewise model test passed")


def test_model_gradients():
    """Test that gradients flow properly."""
    model = PiecewiseTrainedModel(num_classes=21, use_crf=True)
    image = torch.randn(2, 3, 512, 512, requires_grad=True)
    target = torch.randint(0, 21, (2, 512, 512))
    
    unary_output, crf_output = model(image, apply_crf=True)
    
    # Compute loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(crf_output, target)
    
    # Backward
    loss.backward()
    
    # Check that gradients exist
    assert image.grad is not None
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    print("✓ Gradient flow test passed")


if __name__ == '__main__':
    test_deeplab_backbone()
    test_dense_crf()
    test_piecewise_model()
    test_model_gradients()
    print("\n✓ All tests passed!")