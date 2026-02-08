from typing import List

import numpy as np

from ..feedback import Analyzer

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


class CalfRaiseCoach(Analyzer):
    def __init__(self):
        self.reps = 0
        self.state = "down"
        self.baseline_y = None

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Calf Raise"]
        if landmarks_px is None:
            tips.append("No person detected. Ensure lower body in frame.")
            tips.append(f"Reps: {self.reps}")
            return tips

        la = landmarks_px[LEFT_ANKLE].astype(float)
        ra = landmarks_px[RIGHT_ANKLE].astype(float)
        ankle_y = ((la[1] + ra[1]) / 2.0)

        if self.baseline_y is None:
            self.baseline_y = ankle_y

        rise = self.baseline_y - ankle_y
        at_top = rise > 12
        at_bottom = rise < 4

        if self.state == "down" and at_top:
            self.state = "up"
            self.reps += 1
        elif self.state == "up" and at_bottom:
            self.state = "down"

        tips.append(f"Reps: {self.reps} | Phase: {self.state}")
        tips.append("Go up on toes fully; pause and lower slowly.")
        return tips