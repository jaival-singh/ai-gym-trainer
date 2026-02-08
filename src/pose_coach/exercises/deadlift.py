from typing import List

import numpy as np

from ..feedback import Analyzer
from ..angles import angle_between_vectors

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26


class DeadliftCoach(Analyzer):
    def __init__(self):
        self.reps = 0
        self.state = "down"

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Deadlift"]
        if landmarks_px is None:
            tips.append("No person detected. Ensure full body in frame.")
            tips.append(f"Reps: {self.reps}")
            return tips

        ls = landmarks_px[LEFT_SHOULDER].astype(float)
        rs = landmarks_px[RIGHT_SHOULDER].astype(float)
        lh = landmarks_px[LEFT_HIP].astype(float)
        rh = landmarks_px[RIGHT_HIP].astype(float)
        lk = landmarks_px[LEFT_KNEE].astype(float)
        rk = landmarks_px[RIGHT_KNEE].astype(float)

        shoulder = (ls + rs) / 2.0
        hip = (lh + rh) / 2.0
        knee = (lk + rk) / 2.0

        back_vec = shoulder - hip
        thigh_vec = knee - hip
        back_from_vertical = angle_between_vectors(back_vec, np.array([0.0, -1.0]))
        hip_above_knee = hip[1] < knee[1]

        at_top = back_from_vertical < 15 and hip_above_knee
        at_bottom = back_from_vertical > 45

        if self.state == "down" and at_top:
            self.state = "up"
            self.reps += 1
        elif self.state == "up" and at_bottom:
            self.state = "down"

        tips.append(f"Reps: {self.reps} | Phase: {self.state}")
        if back_from_vertical > 35:
            tips.append("Keep back flat; brace core and pack lats.")
        if hip[1] > shoulder[1]:
            tips.append("Hips and shoulders should rise together.")
        tips.append("Keep the bar close; push the floor away.")
        return tips