from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class Analyzer(ABC):
    @abstractmethod
    def update(self, landmarks_px: Optional[np.ndarray], visibility: Optional[np.ndarray], frame_shape) -> list:
        """Process landmarks and return text lines to render on screen."""
        raise NotImplementedError

    @staticmethod
    def is_visible(visibility: Optional[np.ndarray], indices: List[int], threshold: float = 0.5) -> bool:
        if visibility is None:
            return False
        return all(visibility[i] >= threshold for i in indices)

    @staticmethod
    def safe_points(landmarks_px: Optional[np.ndarray], indices: List[int]) -> Optional[np.ndarray]:
        if landmarks_px is None:
            return None
        try:
            return landmarks_px[indices]
        except Exception:
            return None