import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
import os

class HomographyMapper:
    def __init__(self, config_path: str = "config/camera_coverage_analysis.json"):
        """
        Initialize the homography mapper for camera-to-floor-plan transformation.
        
        Args:
            config_path: Path to the camera coverage JSON configuration file
        """
        self.config = self._load_config(config_path)
        self.homography_matrix = None
        self.is_calibrated = False
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default config if file not found
            return {
                "cameras": {}
            }
    
    def calibrate(self, camera_points: List[Tuple[float, float]], 
                  map_points: List[Tuple[float, float]]) -> bool:
        """
        Calculate homography matrix from corresponding points.
        
        Args:
            camera_points: List of (x, y) points in camera image
            map_points: List of (x, y) points in floor plan coordinates
            
        Returns:
            True if calibration successful, False otherwise
        """
        if len(camera_points) < 4 or len(map_points) < 4:
            raise ValueError("At least 4 point pairs required for homography calculation")
            
        if len(camera_points) != len(map_points):
            raise ValueError("Number of camera points must equal number of map points")
            
        # Convert to numpy arrays
        src_pts = np.array(camera_points, dtype=np.float32)
        dst_pts = np.array(map_points, dtype=np.float32)
        
        # Calculate homography using RANSAC for robustness
        self.homography_matrix, mask = cv2.findHomography(
            src_pts, dst_pts, 
            cv2.RANSAC, 
            ransacReprojThreshold=3.0
        )
        
        self.is_calibrated = self.homography_matrix is not None
        return self.is_calibrated
    
    def map_point(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """
        Transform a point from camera coordinates to floor plan coordinates.
        
        Args:
            x: x-coordinate in camera image
            y: y-coordinate in camera image
            
        Returns:
            Transformed (X, Y) point in floor plan coordinates, or None if not calibrated
        """
        if not self.is_calibrated or self.homography_matrix is None:
            # Fallback if not calibrated: return raw coords for now
            return (float(x), float(y))
            
        # Convert point to homogeneous coordinates
        point_homogeneous = np.array([[x, y, 1.0]], dtype=np.float32).T
        
        # Apply homography transformation
        transformed_homogeneous = self.homography_matrix @ point_homogeneous
        
        # Convert back to Cartesian coordinates
        w = transformed_homogeneous[2, 0]
        if w == 0:
            return None
            
        X = transformed_homogeneous[0, 0] / w
        Y = transformed_homogeneous[1, 0] / w
        
        return (float(X), float(Y))
    
    def transform_points(self, points: List[Tuple[float, float]]) -> List[Optional[Tuple[float, float]]]:
        """
        Transform multiple points from camera to floor plan coordinates.
        
        Args:
            points: List of (x, y) points in camera image coordinates
            
        Returns:
            List of transformed points (None for points that couldn't be transformed)
        """
        return [self.map_point(x, y) for (x, y) in points]
    
    def get_homography_matrix(self) -> Optional[np.ndarray]:
        """Get the current homography matrix."""
        return self.homography_matrix.copy() if self.homography_matrix is not None else None