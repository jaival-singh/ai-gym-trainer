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


class LungeCoach(Analyzer):
    def __init__(self):
        self.reps = 0
        self.state = "up"

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Lunge"]
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

        # Determine forward leg by knee-to-hip horizontal distance
        left_stride = abs(lk[0] - lh[0])
        right_stride = abs(rk[0] - rh[0])
        left_forward = left_stride > right_stride

        if left_forward:
            front_knee, front_ankle, front_hip = lk, la, lh
        else:
            front_knee, front_ankle, front_hip = rk, ra, rh

        knee_angle = angle_at_point(front_hip, front_knee, front_ankle)
        vertical_shin = abs(front_knee[0] - front_ankle[0]) < 0.2 * np.linalg.norm(front_hip - front_ankle)
        depth_ok = knee_angle < 110

        at_bottom = depth_ok
        at_top = knee_angle > 160

        if self.state == "up" and at_bottom:
            self.state = "down"
        elif self.state == "down" and at_top:
            self.state = "up"
            self.reps += 1

        tips.append(f"Reps: {self.reps} | Phase: {self.state}")
        if not vertical_shin:
            tips.append("Keep front shin vertical; knee over ankle, not past toes.")
        if not depth_ok:
            tips.append("Lower until front thigh approaches parallel.")
        if vertical_shin and depth_ok:
            tips.append("Nice lunge. Control the descent.")

        return tips