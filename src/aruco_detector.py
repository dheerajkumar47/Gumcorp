import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

class ArUcoDetector:
    def __init__(self, dictionary_type: str = "DICT_4X4_50", 
                 marker_size_mm: float = 50.0):
        """
        Initialize the ArUco marker detector.
        
        Args:
            dictionary_type: Type of ArUco dictionary to use
            marker_size_mm: Physical size of the marker in millimeters
        """
        # Get the ArUco dictionary
        aruco_dict = getattr(cv2.aruco, dictionary_type)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Tuned for factory use: smaller markers at distance, shadowed environments
        self.aruco_params.adaptiveThreshWinSizeMin = 3    # default 7 — catches small/far markers
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 4
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.minMarkerPerimeterRate = 0.02   # default 0.05 — allows smaller markers
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.04
        self.aruco_params.minCornerDistanceRate = 0.02
        self.aruco_params.errorCorrectionRate = 0.9       # slightly more forgiving bit errors

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_size_mm = marker_size_mm
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect ArUco markers in a frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of detected markers, each as dict with:
            - id: marker ID
            - corners: 4 corner points
            - center: center point
            - rvec: rotation vector
            - tvec: translation vector
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        detections = []
        if ids is not None:
            # Try to estimate pose, but handle case where function doesn't exist
            rvecs = None
            tvecs = None
            
            # Check if estimatePoseSingleMarkers is available
            if hasattr(cv2.aruco, 'estimatePoseSingleMarkers'):
                try:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, self.marker_size_mm, 
                        np.eye(3), np.zeros((4, 1))
                    )
                except Exception:
                    pass  # Skip pose estimation if it fails
            
            for i, marker_id in enumerate(ids.flatten()):
                # Calculate center point
                corner_points = corners[i][0]  # 4x2 array
                center = np.mean(corner_points, axis=0).astype(int)
                
                detection = {
                    'id': int(marker_id),
                    'corners': corner_points.astype(int),
                    'center': tuple(center),
                    'rvec': rvecs[i][0] if rvecs is not None else None,
                    'tvec': tvecs[i][0] if tvecs is not None else None
                }
                detections.append(detection)
                
        return detections
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Dict]) -> np.ndarray:
        """
        Draw detected ArUco markers on the frame.
        
        Args:
            frame: Input image frame
            detections: List of detections from detect() method
            
        Returns:
            Frame with detections drawn
        """
        output_frame = frame.copy()
        for detection in detections:
            # Draw the marker boundary
            cv2.polylines(output_frame, [detection['corners']], True, (0, 255, 255), 2)
            
            # Draw the ID
            center = detection['center']
            cv2.putText(output_frame, f"ID: {detection['id']}", 
                       (center[0] - 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw center point
            cv2.circle(output_frame, center, 3, (0, 0, 255), -1)
            
        return output_frame