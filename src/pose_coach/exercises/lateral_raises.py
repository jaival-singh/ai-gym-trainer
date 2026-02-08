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


class LateralRaiseCoach(Analyzer):
    def __init__(self):
        self.reps = 0
        self.state = "down"

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Lateral Raise"]
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

        left_height_ok = abs(le[1] - ls[1]) < 0.08 * frame_shape[0]
        right_height_ok = abs(re[1] - rs[1]) < 0.08 * frame_shape[0]
        elbow_soft = abs(le[1] - lw[1]) < 0.15 * frame_shape[0] and abs(re[1] - rw[1]) < 0.15 * frame_shape[0]

        # use left arm for phase detection
        up_phase = left_height_ok and right_height_ok
        down_phase = (le[1] > ls[1] + 0.12 * frame_shape[0]) and (re[1] > rs[1] + 0.12 * frame_shape[0])
        if self.state == "down" and up_phase:
            self.state = "up"
            self.reps += 1
        elif self.state == "up" and down_phase:
            self.state = "down"

        tips.append(f"Reps: {self.reps} | Phase: {self.state}")
        if not left_height_ok or not right_height_ok:
            tips.append("Raise to shoulder height; control the top.")
        if not elbow_soft:
            tips.append("Keep a slight elbow bend; lead with elbows, not wrists.")
        if left_height_ok and right_height_ok and elbow_soft:
            tips.append("Great raises! Slow on the way down.")
        return tips