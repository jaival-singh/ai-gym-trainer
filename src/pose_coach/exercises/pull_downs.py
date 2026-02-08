from typing import List

import numpy as np

from ..feedback import Analyzer
from ..angles import angle_at_point

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


class PullDownCoach(Analyzer):
    def __init__(self):
        self.reps = 0
        self.state = "up"

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Pull-down"]
        if landmarks_px is None:
            tips.append("No person detected.")
            tips.append(f"Reps: {self.reps}")
            return tips

        ls = landmarks_px[LEFT_SHOULDER].astype(float)
        rs = landmarks_px[RIGHT_SHOULDER].astype(float)
        le = landmarks_px[LEFT_ELBOW].astype(float)
        re = landmarks_px[RIGHT_ELBOW].astype(float)
        lw = landmarks_px[LEFT_WRIST].astype(float)
        rw = landmarks_px[RIGHT_WRIST].astype(float)

        shoulder = (ls + rs) / 2.0
        elbow = (le + re) / 2.0
        wrist = (lw + rw) / 2.0

        elbow_angle = angle_at_point(shoulder, elbow, wrist)
        wide_grip = abs(lw[0] - rw[0]) > 0.6 * abs(rs[0] - ls[0])

        at_bottom = elbow_angle < 70
        at_top = elbow_angle > 150

        if self.state == "up" and at_bottom:
            self.state = "down"
            self.reps += 1
        elif self.state == "down" and at_top:
            self.state = "up"

        tips.append(f"Reps: {self.reps} | Phase: {self.state}")
        tips.append("Pull elbows down and back; avoid shrugging.")
        if not wide_grip:
            tips.append("Consider a slightly wider grip for lat engagement.")
        return tips