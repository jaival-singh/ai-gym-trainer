from typing import List

import numpy as np

from ..feedback import Analyzer

LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_KNEE = 25
RIGHT_KNEE = 26


class BoxJumpCoach(Analyzer):
    def __init__(self):
        self.jumps = 0
        self.state = "ground"
        self.ground_y = None

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Box Jump"]
        if landmarks_px is None:
            tips.append("No person detected.")
            tips.append(f"Jumps: {self.jumps}")
            return tips

        la = landmarks_px[LEFT_ANKLE].astype(float)
        ra = landmarks_px[RIGHT_ANKLE].astype(float)
        lk = landmarks_px[LEFT_KNEE].astype(float)
        rk = landmarks_px[RIGHT_KNEE].astype(float)

        ankle_y = (la[1] + ra[1]) / 2.0
        if self.ground_y is None or ankle_y > self.ground_y:
            self.ground_y = ankle_y

        airborne = ankle_y < self.ground_y - 8

        if self.state == "ground" and airborne:
            self.state = "air"
        elif self.state == "air" and not airborne:
            self.state = "ground"
            self.jumps += 1

        knee_bend_on_landing = (lk[1] + rk[1]) / 2.0 < self.ground_y - 12

        tips.append(f"Jumps: {self.jumps} | Phase: {self.state}")
        if not knee_bend_on_landing:
            tips.append("Absorb landing by bending knees and hips.")
        tips.append("Land softly on entire foot; avoid stiff knees.")
        return tips