import math
from typing import List, Tuple


def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    """Ray casting algorithm to determine if a point is inside a polygon.
    Args:
        point: (x, y) coordinate.
        polygon: List of (x, y) vertices defining the polygon (must be closed or will be treated as closed).
    Returns:
        True if point is inside the polygon, False otherwise.
    """
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y + 1e-9) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def meters_to_pixels(meters: float, scale: float = 100.0) -> int:
    """Convert real‑world meters to pixel units.
    The default scale assumes 100 pixels per meter; adjust as needed.
    """
    return int(meters * scale)


def pixels_to_meters(pixels: int, scale: float = 100.0) -> float:
    """Convert pixel units back to meters using the same scale.
    """
    return pixels / scale
