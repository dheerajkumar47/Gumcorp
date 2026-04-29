import torch
import cv2
from ultralytics import YOLO
import sys

def check_env():
    print("Python Version:", sys.version)
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    print("PyTorch CUDA Build:", torch.version.cuda)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU: not available to PyTorch; install CUDA-enabled PyTorch to use YOLO on GPU.")
    
    print("\nLoading YOLOv8n model...")
    try:
        model = YOLO('models/yolov8n.pt')
        model.to(device)
        print("Model loaded successfully.")
        
        # Test on a blank image
        import numpy as np
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(blank, verbose=False, device=device)
        print(f"Inference test successful on {device}.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_env()
