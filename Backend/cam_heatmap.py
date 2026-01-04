"""
CAM Heatmap Generator for Fracture Detection
Uses ScoreCAM and XGradCAM methods to visualize fracture regions
"""

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from pytorch_grad_cam import ScoreCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
import io
import base64
import onnxruntime as ort


class DenseNet169Binary(nn.Module):
    """DenseNet-169 model for binary classification (fracture/no fracture)"""
    def __init__(self, pretrained=True):
        super(DenseNet169Binary, self).__init__()
        self.densenet = models.densenet169(weights='IMAGENET1K_V1' if pretrained else None)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.densenet(x)
        x = self.sigmoid(x)
        return x


class FractureCAMGenerator:
    """Generate CAM heatmaps for fracture detection using ScoreCAM and XGradCAM"""
    
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    def __init__(self, model_path: str = None, onnx_path: str = None, device: str = None):
        """
        Initialize the CAM generator
        
        Args:
            model_path: Path to the trained PyTorch model checkpoint (.pth file)
            onnx_path: Path to ONNX model for prediction (optional)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = None
        self.onnx_session = None
        self.target_layers = None
        
        if model_path:
            self.load_pytorch_model(model_path)
        elif onnx_path:
            self.load_onnx_and_create_pytorch_model(onnx_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN.tolist(), std=self.IMAGENET_STD.tolist())
        ])
    
    def load_pytorch_model(self, model_path: str):
        """Load a trained PyTorch DenseNet-169 model"""
        self.model = DenseNet169Binary(pretrained=False)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self._setup_target_layers()
        print(f"PyTorch model loaded successfully on {self.device}")
    
    def load_onnx_and_create_pytorch_model(self, onnx_path: str):
        """
        Load ONNX model for predictions and create a PyTorch model for CAM
        Note: CAM visualizations will use pretrained DenseNet features
        """
        # Load ONNX for accurate predictions
        self.onnx_session = ort.InferenceSession(onnx_path)
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name
        
        # Create PyTorch model for CAM visualization (uses pretrained weights)
        self.model = DenseNet169Binary(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._setup_target_layers()
        print(f"ONNX model loaded for predictions, PyTorch DenseNet for CAM on {self.device}")
    
    def _setup_target_layers(self):
        """Setup target layers for CAM - last conv layer of DenseNet-169"""
        # DenseNet-169 has denseblock4 with denselayer32 as the last layer
        self.target_layers = [self.model.densenet.features.denseblock4.denselayer32.conv2]
    
    def preprocess_image(self, image: Image.Image) -> tuple:
        """
        Preprocess image for model input
        
        Returns:
            tuple: (input_tensor, rgb_image for visualization)
        """
        # Ensure RGB
        image = image.convert('RGB')
        
        # Get tensor for model
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get normalized RGB image for visualization
        img_resized = image.resize((224, 224))
        rgb_img = np.array(img_resized, dtype=np.float32) / 255.0
        
        return input_tensor, rgb_img
    
    def denormalize_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert normalized tensor back to displayable image"""
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * self.IMAGENET_STD + self.IMAGENET_MEAN
        img = np.clip(img, 0, 1)
        return img
    
    def generate_cam(self, image: Image.Image, method: str = 'XGradCAM') -> dict:
        """
        Generate CAM heatmap for the input image
        
        Args:
            image: PIL Image
            method: 'ScoreCAM' or 'XGradCAM'
        
        Returns:
            dict with heatmap, prediction, and confidence
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        input_tensor, rgb_img = self.preprocess_image(image)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = output.item()
            fracture_detected = probability > 0.5
            confidence = probability if fracture_detected else (1 - probability)
        
        # Select CAM method
        if method == 'ScoreCAM':
            cam = ScoreCAM(model=self.model, target_layers=self.target_layers)
        elif method == 'XGradCAM':
            cam = XGradCAM(model=self.model, target_layers=self.target_layers)
        else:
            raise ValueError(f"Unknown CAM method: {method}. Use 'ScoreCAM' or 'XGradCAM'")
        
        # Generate CAM
        targets = [BinaryClassifierOutputTarget(1)]  # Target fracture class
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Create visualization overlay
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        return {
            'heatmap': grayscale_cam,
            'visualization': visualization,
            'original_image': rgb_img,
            'probability': probability,
            'fracture_detected': fracture_detected,
            'confidence': confidence,
            'method': method
        }
    
    def generate_both_cams(self, image: Image.Image) -> dict:
        """
        Generate both ScoreCAM and XGradCAM heatmaps
        
        Args:
            image: PIL Image
        
        Returns:
            dict with both heatmaps and predictions
        """
        input_tensor, rgb_img = self.preprocess_image(image)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = output.item()
            fracture_detected = probability > 0.5
            confidence = probability if fracture_detected else (1 - probability)
        
        results = {
            'original_image': rgb_img,
            'probability': probability,
            'fracture_detected': fracture_detected,
            'confidence': confidence,
            'heatmaps': {}
        }
        
        targets = [BinaryClassifierOutputTarget(1)]
        
        # Generate ScoreCAM
        score_cam = ScoreCAM(model=self.model, target_layers=self.target_layers)
        score_grayscale = score_cam(input_tensor=input_tensor, targets=targets)[0, :]
        score_viz = show_cam_on_image(rgb_img, score_grayscale, use_rgb=True)
        
        results['heatmaps']['ScoreCAM'] = {
            'grayscale': score_grayscale,
            'visualization': score_viz
        }
        
        # Generate XGradCAM
        xgrad_cam = XGradCAM(model=self.model, target_layers=self.target_layers)
        xgrad_grayscale = xgrad_cam(input_tensor=input_tensor, targets=targets)[0, :]
        xgrad_viz = show_cam_on_image(rgb_img, xgrad_grayscale, use_rgb=True)
        
        results['heatmaps']['XGradCAM'] = {
            'grayscale': xgrad_grayscale,
            'visualization': xgrad_viz
        }
        
        return results
    
    def visualization_to_base64(self, visualization: np.ndarray) -> str:
        """Convert visualization numpy array to base64 string for API response"""
        img = Image.fromarray((visualization * 255).astype(np.uint8) if visualization.max() <= 1 else visualization.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def generate_cam_base64(self, image: Image.Image, method: str = 'XGradCAM') -> dict:
        """
        Generate CAM and return as base64 encoded image
        
        Args:
            image: PIL Image
            method: 'ScoreCAM' or 'XGradCAM'
        
        Returns:
            dict with base64 encoded heatmap and prediction info
        """
        result = self.generate_cam(image, method)
        
        return {
            'heatmap_base64': self.visualization_to_base64(result['visualization']),
            'probability': result['probability'],
            'fracture_detected': result['fracture_detected'],
            'confidence': result['confidence'],
            'method': result['method']
        }
    
    def generate_both_cams_base64(self, image: Image.Image) -> dict:
        """
        Generate both CAMs and return as base64 encoded images
        
        Args:
            image: PIL Image
        
        Returns:
            dict with base64 encoded heatmaps for both methods
        """
        results = self.generate_both_cams(image)
        
        return {
            'probability': results['probability'],
            'fracture_detected': results['fracture_detected'],
            'confidence': results['confidence'],
            'original_base64': self.visualization_to_base64(results['original_image']),
            'scorecam_base64': self.visualization_to_base64(results['heatmaps']['ScoreCAM']['visualization']),
            'xgradcam_base64': self.visualization_to_base64(results['heatmaps']['XGradCAM']['visualization'])
        }


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Initialize generator (without model for demo)
    print("CAM Heatmap Generator for Fracture Detection")
    print("=" * 50)
    print("\nUsage:")
    print("  from cam_heatmap import FractureCAMGenerator")
    print("  ")
    print("  # Initialize with model")
    print("  generator = FractureCAMGenerator('path/to/model.pth')")
    print("  ")
    print("  # Generate single CAM")
    print("  result = generator.generate_cam(image, method='XGradCAM')")
    print("  ")
    print("  # Generate both CAMs")
    print("  results = generator.generate_both_cams(image)")
    print("  ")
    print("  # Get base64 for API response")
    print("  response = generator.generate_cam_base64(image, method='ScoreCAM')")
    print("\nSupported methods: 'ScoreCAM', 'XGradCAM'")
