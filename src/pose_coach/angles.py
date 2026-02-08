from typing import Tuple

import math
import numpy as np


def angle_at_point(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees between BA and BC.
    a, b, c are 2D points (x, y).
    """
    ba = a - b
    bc = c - b
    dot = float(np.dot(ba, bc))
    norm = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_val = max(-1.0, min(1.0, dot / norm))
    return math.degrees(math.acos(cos_val))


def angle_of_vector(v: np.ndarray) -> float:
    """Angle of vector vs +x axis in degrees (0 to 180 for our use)."""
    angle = math.degrees(math.atan2(v[1], v[0]))
    return angle


def horizontal_angle(p1: np.ndarray, p2: np.ndarray) -> float:
    """Angle of p2-p1 vector relative to horizontal (x-axis)."""
    return abs(angle_of_vector(p2 - p1))


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(np.dot(v1, v2))
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
    cos_val = max(-1.0, min(1.0, dot / denom))
    return math.degrees(math.acos(cos_val))