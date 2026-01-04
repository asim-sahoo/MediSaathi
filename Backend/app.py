from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import base64
import os

app = FastAPI(title="MediSaathi X-Ray Analysis API", version="1.0.0")

# CORS Configuration for production
allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://medisaathi.vercel.app",
    "https://medisaathi-frontend.vercel.app",
    "https://medisaathi-api.onrender.com",
]
# Add custom frontend URL from environment
if os.getenv("FRONTEND_URL"):
    allowed_origins.append(os.getenv("FRONTEND_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if os.getenv("RENDER") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX model at startup
try:
    session = ort.InferenceSession("densenet169_fracture.onnx")
    input_name = session.get_inputs()[0].name
    print(f"Model loaded successfully. Input name: {input_name}")
    print(f"Expected input shape: {session.get_inputs()[0].shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# CAM Generator - lazy loaded
cam_generator = None

def get_cam_generator():
    """Lazy load the CAM generator to avoid slow startup"""
    global cam_generator
    if cam_generator is None:
        try:
            from cam_heatmap import FractureCAMGenerator
            cam_generator = FractureCAMGenerator(onnx_path="densenet169_fracture.onnx")
            print("CAM Generator loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load CAM generator: {e}")
            cam_generator = "unavailable"
    return cam_generator if cam_generator != "unavailable" else None

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for DenseNet-169 model
    
    Args:
        image: PIL Image in RGB format
        
    Returns:
        numpy array of shape (1, 3, 224, 224) with normalized values
    """
    # Resize to 224x224
    image = image.resize((224, 224), Image.BILINEAR)
    
    # Convert to numpy array (H, W, C) with values 0-255
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Apply ImageNet normalization
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    
    # Transpose to (C, H, W) format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension (1, C, H, W)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.post("/detect")
async def detect_fracture(file: UploadFile = File(...)):
    """
    Detect fractures in X-ray images using DenseNet-169 ONNX model
    
    Args:
        file: Uploaded image file
        
    Returns:
        Dictionary with probability, fracture_detected flag, and confidence
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess image
        input_array = preprocess_image(image)
        
        # Run inference
        outputs = session.run(None, {input_name: input_array})
        
        # Extract probability (output shape is (1, 1))
        probability = float(outputs[0].flatten()[0])
        
        # Binary prediction at threshold 0.5
        fracture_detected = probability > 0.3
        
        # Calculate confidence
        confidence = probability if fracture_detected else (1 - probability)
        
        return {
            "probability": probability,
            "fracture_detected": fracture_detected,
            "confidence": confidence
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/detect/heatmap")
async def detect_fracture_with_heatmap(
    file: UploadFile = File(...),
    method: str = Query("XGradCAM", description="CAM method: 'ScoreCAM' or 'XGradCAM'")
):
    """
    Detect fractures and generate CAM heatmap visualization
    
    Args:
        file: Uploaded X-ray image file
        method: CAM method to use ('ScoreCAM' or 'XGradCAM')
        
    Returns:
        Dictionary with prediction results and base64 encoded heatmap
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get prediction from ONNX model
        input_array = preprocess_image(image)
        outputs = session.run(None, {input_name: input_array})
        probability = float(outputs[0].flatten()[0])
        fracture_detected = probability > 0.3
        confidence = probability if fracture_detected else (1 - probability)
        
        # Try to generate CAM heatmap
        heatmap_base64 = None
        cam_method_used = None
        
        generator = get_cam_generator()
        if generator:
            try:
                result = generator.generate_cam_base64(image, method=method)
                heatmap_base64 = result['heatmap_base64']
                cam_method_used = method
            except Exception as cam_error:
                print(f"CAM generation failed: {cam_error}")
        
        return {
            "probability": probability,
            "fracture_detected": fracture_detected,
            "confidence": confidence,
            "heatmap_base64": heatmap_base64,
            "cam_method": cam_method_used
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/detect/heatmap/both")
async def detect_fracture_with_both_heatmaps(file: UploadFile = File(...)):
    """
    Detect fractures and generate both ScoreCAM and XGradCAM heatmaps
    
    Args:
        file: Uploaded X-ray image file
        
    Returns:
        Dictionary with prediction results and base64 encoded heatmaps for both methods
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get prediction from ONNX model
        input_array = preprocess_image(image)
        outputs = session.run(None, {input_name: input_array})
        probability = float(outputs[0].flatten()[0])
        fracture_detected = probability > 0.3
        confidence = probability if fracture_detected else (1 - probability)
        
        # Generate original image base64
        img_resized = image.resize((224, 224))
        buffer = io.BytesIO()
        img_resized.save(buffer, format='PNG')
        buffer.seek(0)
        original_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Try to generate CAM heatmaps
        scorecam_base64 = None
        xgradcam_base64 = None
        
        generator = get_cam_generator()
        if generator:
            try:
                results = generator.generate_both_cams_base64(image)
                scorecam_base64 = results['scorecam_base64']
                xgradcam_base64 = results['xgradcam_base64']
            except Exception as cam_error:
                print(f"CAM generation failed: {cam_error}")
        
        return {
            "probability": probability,
            "fracture_detected": fracture_detected,
            "confidence": confidence,
            "original_base64": original_base64,
            "scorecam_base64": scorecam_base64,
            "xgradcam_base64": xgradcam_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint for basic connectivity check"""
    return {"message": "MediSaathi X-Ray Analysis API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint - fast response for Render health checks"""
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.get("/health/detailed")
async def health_check_detailed():
    """Detailed health check with CAM availability (slower)"""
    cam_available = get_cam_generator() is not None
    return {
        "status": "healthy",
        "model_loaded": True,
        "cam_available": cam_available
    }
