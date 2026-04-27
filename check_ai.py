import torch
import cv2
from ultralytics import YOLO
import sys

def check_env():
    print("Python Version:", sys.version)
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    
    print("\nLoading YOLOv8n model...")
    try:
        model = YOLO('models/yolov8n.pt')
        print("Model loaded successfully.")
        
        # Test on a blank image
        import numpy as np
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(blank, verbose=False)
        print("Inference test successful.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_env()
