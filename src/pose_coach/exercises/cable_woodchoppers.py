from typing import List

import numpy as np

from ..feedback import Analyzer
from ..angles import angle_between_vectors

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_WRIST = 15
RIGHT_WRIST = 16


class CableWoodchopperCoach(Analyzer):
    def __init__(self):
        self.reps = 0
        self.state = "start"

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Cable Woodchopper"]
        if landmarks_px is None:
            tips.append("No person detected.")
            tips.append(f"Reps: {self.reps}")
            return tips

        ls = landmarks_px[LEFT_SHOULDER].astype(float)
        rs = landmarks_px[RIGHT_SHOULDER].astype(float)
        lh = landmarks_px[LEFT_HIP].astype(float)
        rh = landmarks_px[RIGHT_HIP].astype(float)
        lw = landmarks_px[LEFT_WRIST].astype(float)
        rw = landmarks_px[RIGHT_WRIST].astype(float)

        shoulder = (ls + rs) / 2.0
        hip = (lh + rh) / 2.0
        hands = (lw + rw) / 2.0

        torso_vec = shoulder - hip
        hands_vec = hands - shoulder
        rotation = angle_between_vectors(torso_vec, np.array([0.0, -1.0]))
        hands_distance = np.linalg.norm(hands_vec)

        big_rotation = rotation > 25
        hands_far = hands_distance > 0.25 * frame_shape[1]

        if self.state in ("start", "eccentric") and big_rotation and hands_far:
            self.state = "concentric"
            self.reps += 1
        elif self.state == "concentric" and not big_rotation:
            self.state = "eccentric"

        tips.append(f"Reps: {self.reps} | Phase: {self.state}")
        tips.append("Rotate torso, not just arms; control the chop.")
        return tips