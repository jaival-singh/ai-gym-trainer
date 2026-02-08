from typing import List

import numpy as np

from ..feedback import Analyzer
from ..angles import angle_at_point

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_WRIST = 15
RIGHT_WRIST = 16


class PushupCoach(Analyzer):
    def __init__(self):
        self.reps = 0
        self.state = "up"

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Pushup"]
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
        lh = landmarks_px[LEFT_HIP].astype(float)
        rh = landmarks_px[RIGHT_HIP].astype(float)

        shoulder = (ls + rs) / 2.0
        elbow = (le + re) / 2.0
        wrist = (lw + rw) / 2.0
        hip = (lh + rh) / 2.0

        elbow_angle = angle_at_point(shoulder, elbow, wrist)
        hips_sagging = hip[1] > shoulder[1] + 0.08 * frame_shape[0]

        at_bottom = elbow_angle < 80
        at_top = elbow_angle > 150

        if self.state == "up" and at_bottom:
            self.state = "down"
        elif self.state == "down" and at_top:
            self.state = "up"
            self.reps += 1

        tips.append(f"Reps: {self.reps} | Phase: {self.state}")
        if hips_sagging:
            tips.append("Engage core; keep a straight line from shoulders to ankles.")
        tips.append("Chest to ~90° elbow bend; full lockout at top.")
        return tips