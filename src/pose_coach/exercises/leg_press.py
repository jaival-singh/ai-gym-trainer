from typing import List

import numpy as np

from ..feedback import Analyzer
from ..angles import angle_at_point

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


class LegPressCoach(Analyzer):
    def __init__(self):
        self.reps = 0
        self.state = "start"

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Leg Press"]
        if landmarks_px is None:
            tips.append("No person detected. Ensure lower body in frame.")
            tips.append(f"Reps: {self.reps}")
            return tips

        lh = landmarks_px[LEFT_HIP].astype(float)
        rh = landmarks_px[RIGHT_HIP].astype(float)
        lk = landmarks_px[LEFT_KNEE].astype(float)
        rk = landmarks_px[RIGHT_KNEE].astype(float)
        la = landmarks_px[LEFT_ANKLE].astype(float)
        ra = landmarks_px[RIGHT_ANKLE].astype(float)

        hip = (lh + rh) / 2.0
        knee = (lk + rk) / 2.0
        ankle = (la + ra) / 2.0

        knee_angle = angle_at_point(hip, knee, ankle)
        knees_in = abs(lk[0] - lh[0]) < abs(la[0] - lh[0]) * 0.7 or abs(rk[0] - rh[0]) < abs(ra[0] - rh[0]) * 0.7

        at_bottom = knee_angle < 90
        at_top = knee_angle > 160

        if self.state in ("start", "down") and at_top:
            self.state = "up"
        elif self.state in ("start", "up") and at_bottom:
            self.state = "down"
            self.reps += 1

        tips.append(f"Reps: {self.reps} | Phase: {self.state}")
        tips.append("Track knees over toes; avoid valgus collapse." if knees_in else "Good knee tracking.")
        tips.append("Control ROM: full extension without locking, deep enough at bottom.")
        return tips