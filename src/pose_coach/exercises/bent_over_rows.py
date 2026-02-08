from typing import List

import numpy as np

from ..feedback import Analyzer
from ..angles import angle_between_vectors, angle_at_point

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


class BentOverRowCoach(Analyzer):
    def __init__(self):
        self.reps = 0
        self.state = "down"

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Bent-over Row"]
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

        torso_vec = hip - shoulder
        torso_angle_from_horizontal = angle_between_vectors(torso_vec, np.array([1.0, 0.0]))
        hinge_ok = 20 <= torso_angle_from_horizontal <= 60

        elbow_angle = angle_at_point(shoulder, elbow, wrist)
        at_top = elbow_angle < 70
        at_bottom = elbow_angle > 140

        if self.state == "down" and at_top:
            self.state = "up"
            self.reps += 1
        elif self.state == "up" and at_bottom:
            self.state = "down"

        wrist_under_elbow = abs(wrist[0] - elbow[0]) < 0.4 * np.linalg.norm(elbow - shoulder)

        tips.append(f"Reps: {self.reps} | Phase: {self.state}")
        if not hinge_ok:
            tips.append("Hinge more at hips; torso ~30-45° to floor.")
        if not wrist_under_elbow:
            tips.append("Keep bar under elbows; pull to lower ribs.")
        if hinge_ok and wrist_under_elbow:
            tips.append("Strong rows. Squeeze lats at top.")

        return tips