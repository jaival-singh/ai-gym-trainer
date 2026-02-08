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


class RDLCoach(Analyzer):
    def __init__(self):
        self.reps = 0
        self.state = "up"

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Romanian Deadlift"]
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

        torso_vec = shoulder - hip
        thigh_vec = knee - hip
        torso_from_horizontal = angle_between_vectors(torso_vec, np.array([1.0, 0.0]))
        knee_bend_small = angle_between_vectors(thigh_vec, np.array([0.0, 1.0])) < 25
        hip_hinge_ok = 20 <= torso_from_horizontal <= 70

        at_bottom = torso_from_horizontal > 50
        at_top = torso_from_horizontal < 20

        if self.state == "up" and at_bottom:
            self.state = "down"
        elif self.state == "down" and at_top:
            self.state = "up"
            self.reps += 1

        tips.append(f"Reps: {self.reps} | Phase: {self.state}")
        if not hip_hinge_ok:
            tips.append("Hinge at hips; push hips back and keep back flat.")
        if not knee_bend_small:
            tips.append("Keep a slight knee bend; avoid squatting the weight.")
        if hip_hinge_ok and knee_bend_small:
            tips.append("Good hinge. Keep bar close to legs.")

        return tips