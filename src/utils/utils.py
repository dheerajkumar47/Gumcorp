import cv2
import numpy as np
from typing import Tuple, List, Optional

def draw_floor_plan(frame: np.ndarray, 
                  zones: dict,
                  tracks: dict,
                  track_histories: dict = None) -> np.ndarray:
    """
    Draw the floor plan overlay with zones and track positions.
    """
    output = frame.copy()
    height, width = frame.shape[:2]
    
    # Draw zones
    for zone_name, zone_data in zones.items():
        if 'polygon' in zone_data and zone_data['polygon']:
            polygon = np.array(zone_data['polygon'], dtype=np.int32)
            color = zone_data.get('color', (128, 128, 128))
            cv2.polylines(output, [polygon], True, color, 2)
            cv2.fillPoly(output, [polygon], color=(color[0]//2, color[1]//2, color[2]//2))
            center = np.mean(polygon, axis=0).astype(int)
            cv2.putText(output, zone_name, tuple(center), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw track positions
    for track_id, position in tracks.items():
        x, y = map(int, position)
        cv2.circle(output, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(output, (x, y), 10, (255, 255, 255), 1)
        cv2.putText(output, f"ID:{track_id}", (x + 10, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw track histories
    if track_histories:
        for track_id, history in track_histories.items():
            if len(history) > 1:
                for i in range(1, len(history)):
                    pt1 = tuple(map(int, history[i-1]))
                    pt2 = tuple(map(int, history[i]))
                    cv2.line(output, pt1, pt2, (0, 255, 0), 1)
    
    return output

def draw_activity_info(frame: np.ndarray, 
                   activities: dict) -> np.ndarray:
    """
    Draw activity information on the frame.
    """
    output = frame.copy()
    y_offset = 30
    
    for track_id, activity in activities.items():
        text = f"ID {track_id}: {activity}"
        cv2.putText(output, text, (10, y_offset),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30
    
    return output

def calculate_distance(p1: Tuple[float, float], 
                    p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def point_in_polygon(point: Tuple[float, float], 
                 polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting.
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def normalize_coordinates(points: List[Tuple[float, float]],
                     source_bounds: Tuple[float, float, float, float],
                     target_bounds: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    """
    Normalize coordinates from source bounds to target bounds.
    """
    sx_min, sy_min, sx_max, sy_max = source_bounds
    tx_min, ty_min, tx_max, ty_max = target_bounds
    
    sx_range = sx_max - sx_min if sx_max != sx_min else 1
    sy_range = sy_max - sy_min if sy_max != sy_min else 1
    tx_range = tx_max - tx_min
    ty_range = ty_max - ty_min
    
    normalized = []
    for x, y in points:
        nx = ((x - sx_min) / sx_range) * tx_range + tx_min
        ny = ((y - sy_min) / sy_range) * ty_range + ty_min
        normalized.append((nx, ny))
    
    return normalized

def get_bounding_box_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Get center point of a bounding box.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def compute_iou(box1: Tuple[float, float, float, float],
               box2: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union between two bounding boxes.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0