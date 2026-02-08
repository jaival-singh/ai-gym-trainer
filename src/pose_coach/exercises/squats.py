from typing import List

import numpy as np

from ..feedback import Analyzer
from ..angles import angle_at_point, angle_between_vectors

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


class SquatCoach(Analyzer):
    def __init__(self):
        self.reps = 0
        self.state = "up"

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Squat"]
        if landmarks_px is None:
            tips.append("No person detected. Ensure full body in frame.")
            tips.append(f"Reps: {self.reps}")
            return tips

        lh = landmarks_px[LEFT_HIP].astype(float)
        rh = landmarks_px[RIGHT_HIP].astype(float)
        lk = landmarks_px[LEFT_KNEE].astype(float)
        rk = landmarks_px[RIGHT_KNEE].astype(float)
        la = landmarks_px[LEFT_ANKLE].astype(float)
        ra = landmarks_px[RIGHT_ANKLE].astype(float)
        ls = landmarks_px[LEFT_SHOULDER].astype(float)
        rs = landmarks_px[RIGHT_SHOULDER].astype(float)

        hip = (lh + rh) / 2.0
        knee = (lk + rk) / 2.0
        ankle = (la + ra) / 2.0
        shoulder = (ls + rs) / 2.0

        knee_angle = angle_at_point(hip, knee, ankle)
        hip_depth_px = hip[1] - knee[1]
        torso_vec = shoulder - hip
        torso_angle_from_vertical = angle_between_vectors(torso_vec, np.array([0.0, -1.0]))

        depth_ok = hip_depth_px > 0
        knee_ok = knee_angle > 90
        torso_ok = torso_angle_from_vertical < 35

        at_bottom = depth_ok and knee_angle < 100
        at_top = knee_angle > 160

        if self.state == "up" and at_bottom:
            self.state = "down"
        elif self.state == "down" and at_top:
            self.state = "up"
            self.reps += 1

        tips.append(f"Reps: {self.reps} | Phase: {self.state}")
        if not depth_ok:
            tips.append("Sit deeper: hip crease below knee.")
        if not knee_ok:
            tips.append("Avoid knees collapsing; keep tracking over toes.")
        if not torso_ok:
            tips.append("Keep chest up; avoid excessive forward lean.")
        if depth_ok and knee_ok and torso_ok:
            tips.append("Solid squat! Drive through heels.")

        return tips