import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

class MuraDataset(Dataset):
    def __init__(self, csv_file, base_path, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None, names=['path', 'label'])
        self.base_path = base_path
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        self.classes = []
        
        for idx, row in self.annotations.iterrows():
            path = row['path']
            if path.startswith('MURA-v1.1/'):
                path = path[10:]
            directory_path = os.path.join(base_path, path)
            
            class_name = self.get_class_from_path(row['path'])
            image_files = glob.glob(os.path.join(directory_path, '*.png'))
            
            for img_file in image_files:
                self.image_paths.append(img_file)
                self.labels.append(row['label'])
                self.classes.append(class_name)
    
    def __len__(self):
        return len(self.image_paths)
    
    def get_class_from_path(self, path):
        parts = path.strip('/').split('/')
        if len(parts) >= 3:
            return parts[2]
        return None
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, self.classes[idx], img_path
    
    def get_unique_classes(self):
        return list(set(self.classes))

class DenseNet169Binary(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet169Binary, self).__init__()
        self.densenet = models.densenet169(pretrained=pretrained)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.densenet(x)
        x = self.sigmoid(x)
        return x

def get_val_transform():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

def load_model(checkpoint_path, device):
    model = DenseNet169Binary(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint

def get_sample_images_per_class(dataset, num_samples=1):
    """Get one sample image per unique class"""
    class_samples = {}
    
    for idx in range(len(dataset)):
        _, label, class_name, img_path = dataset[idx]
        
        if class_name not in class_samples:
            class_samples[class_name] = {
                'idx': idx,
                'label': label,
                'path': img_path
            }
        
        if len(class_samples) == len(dataset.get_unique_classes()):
            break
    
    return class_samples

def apply_cam_method(model, input_tensor, cam_method, target_layers):
    """Apply CAM method and return visualization"""
    if cam_method == 'GradCAM':
        cam = GradCAM(model=model, target_layers=target_layers)
    elif cam_method == 'GradCAM++':
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    elif cam_method == 'ScoreCAM':
        cam = ScoreCAM(model=model, target_layers=target_layers)
    elif cam_method == 'XGradCAM':
        cam = XGradCAM(model=model, target_layers=target_layers)
    else:
        raise ValueError(f"Unknown CAM method: {cam_method}")
    
    targets = [BinaryClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    return grayscale_cam

def denormalize_image(tensor):
    """Convert normalized tensor back to image"""
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * imagenet_std + imagenet_mean
    img = np.clip(img, 0, 1)
    
    return img

def create_cam_visualization_grid(model, checkpoint_path, dataset, device, output_dir):
    """Create 7x4 grid: 7 classes (columns) x 4 CAM methods (rows)"""
    model_name = os.path.basename(checkpoint_path).replace('.pth', '')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    val_loss = checkpoint['val_loss']
    
    cam_methods = ['GradCAM', 'GradCAM++', 'ScoreCAM', 'XGradCAM']
    target_layers = [model.densenet.features.denseblock4.denselayer16.conv2]
    
    class_samples = get_sample_images_per_class(dataset)
    sorted_classes = sorted(class_samples.keys())
    
    num_classes = len(sorted_classes)
    num_methods = len(cam_methods)
    
    fig_width = num_classes * 6
    fig_height = num_methods * 6
    
    fig, axes = plt.subplots(num_methods, num_classes, figsize=(fig_width, fig_height), dpi=300)
    
    if num_classes == 1:
        axes = axes.reshape(-1, 1)
    if num_methods == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\nGenerating CAM visualizations for {model_name}...")
    
    for col_idx, class_name in enumerate(tqdm(sorted_classes, desc="Processing classes")):
        sample_info = class_samples[class_name]
        img_tensor, true_label, _, img_path = dataset[sample_info['idx']]
        
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred_label = 1 if output.item() > 0.5 else 0
        
        rgb_img = denormalize_image(img_tensor)
        
        for row_idx, cam_method in enumerate(cam_methods):
            grayscale_cam = apply_cam_method(model, input_tensor, cam_method, target_layers)
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            ax = axes[row_idx, col_idx]
            ax.imshow(visualization)
            ax.axis('off')
            
            if row_idx == 0:
                title = f"{class_name.upper()}\nTrue: {true_label} | Pred: {pred_label}"
                ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            
            if col_idx == 0:
                ax.text(-0.1, 0.5, cam_method, transform=ax.transAxes, 
                       fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)
    
    plt.suptitle(f"{model_name}_CAM_Methods_Comparison (Val Loss: {val_loss:.4f})", 
                 fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = os.path.join(output_dir, f"{model_name}_CAM_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"✓ Saved: {output_path}")

def main():
    DATASET_PATH = "/home/rs/24CS91R03/Antareep/MURA-v1.1"
    SAVE_DIR = "/home/rs/24CS91R03/Antareep/trained_models"
    OUTPUT_DIR = "/home/rs/24CS91R03/Antareep/CAM_outputs"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ensemble_info_path = os.path.join(SAVE_DIR, 'ensemble_info.json')
    if not os.path.exists(ensemble_info_path):
        print(f"ERROR: No trained models found at {ensemble_info_path}")
        return
    
    with open(ensemble_info_path, 'r') as f:
        ensemble_info = json.load(f)
    
    model_checkpoints = ensemble_info['top_5_checkpoints']
    
    print(f"\nFound {len(model_checkpoints)} trained models")
    for i, ckpt in enumerate(model_checkpoints, 1):
        print(f"  {i}. {ckpt}")
    
    val_csv = os.path.join(DATASET_PATH, "valid_labeled_studies.csv")
    if not os.path.exists(val_csv):
        print(f"ERROR: Validation CSV not found at {val_csv}")
        return
    
    val_transform = get_val_transform()
    val_dataset = MuraDataset(val_csv, DATASET_PATH, transform=val_transform)
    
    print(f"\nValidation dataset size: {len(val_dataset)}")
    print(f"Unique classes: {val_dataset.get_unique_classes()}")
    
    for checkpoint_name in model_checkpoints:
        checkpoint_path = os.path.join(SAVE_DIR, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing: {checkpoint_name}")
        print(f"{'='*80}")
        
        model, _ = load_model(checkpoint_path, device)
        
        create_cam_visualization_grid(
            model=model,
            checkpoint_path=checkpoint_path,
            dataset=val_dataset,
            device=device,
            output_dir=OUTPUT_DIR
        )
    
    print(f"\n{'='*80}")
    print("CAM VISUALIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
