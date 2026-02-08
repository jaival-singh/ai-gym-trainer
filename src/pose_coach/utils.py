from typing import List, Optional, Tuple

import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise RuntimeError("mediapipe is required. Install via requirements.txt") from exc


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def init_pose_estimator(
    static_image_mode: bool,
    model_complexity: int,
    smooth_landmarks: bool,
    enable_segmentation: bool,
    min_detection_confidence: float,
    min_tracking_confidence: float,
):
    return mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=enable_segmentation,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_landmarks(results, frame_shape) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not results or not results.pose_landmarks:
        return None, None
    h, w = frame_shape[:2]
    points = []
    visibility = []
    for lm in results.pose_landmarks.landmark:
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        points.append([x_px, y_px])
        visibility.append(lm.visibility)
    return np.array(points, dtype=np.int32), np.array(visibility, dtype=np.float32)