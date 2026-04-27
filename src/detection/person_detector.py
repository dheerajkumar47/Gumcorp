import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict

class PersonDetector:
    def __init__(self, model_path: str = "models/yolov8n.pt", 
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        """
        Initialize the person detector using YOLOv8.
        
        Args:
            model_path: Path to the YOLOv8 model weights
            confidence_threshold: Minimum confidence for detection
            iou_threshold: IoU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = [0]  # person class in COCO dataset
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in a frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of detections, each as {"bbox": (x1, y1, x2, y2), "center": (cx, cy), "confidence": conf}
        """
        # Run YOLOv8 inference
        results = self.model(frame, 
                            conf=self.confidence_threshold,
                            iou=self.iou_threshold,
                            classes=self.classes,
                            verbose=False)[0]
        
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # (n, 4)
            confidences = results.boxes.conf.cpu().numpy()  # (n,)
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "confidence": float(conf)
                })
                
        return detections
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and confidence scores on the frame.
        
        Args:
            frame: Input image frame
            detections: List of detections from detect() method
            
        Returns:
            Frame with detections drawn
        """
        output_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw center point
            cx, cy = det["center"]
            cv2.circle(output_frame, (cx, cy), 3, (0, 0, 255), -1)
            # Draw confidence score
            label = f"Person: {conf:.2f}"
            cv2.putText(output_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return output_frame