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
LEFT_KNEE = 25
RIGHT_KNEE = 26


class AbdominalsAndThighsRater(Analyzer):
    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Pose - Abdominals and Thighs"]
        if landmarks_px is None:
            tips.append("No person detected.")
            return tips

        ls = landmarks_px[LEFT_SHOULDER].astype(float)
        rs = landmarks_px[RIGHT_SHOULDER].astype(float)
        le = landmarks_px[LEFT_ELBOW].astype(float)
        re = landmarks_px[RIGHT_ELBOW].astype(float)
        lh = landmarks_px[LEFT_HIP].astype(float)
        rh = landmarks_px[RIGHT_HIP].astype(float)
        lk = landmarks_px[LEFT_KNEE].astype(float)
        rk = landmarks_px[RIGHT_KNEE].astype(float)

        elbows_out = abs(le[0] - ls[0]) > 0.2 * abs(rs[0] - ls[0]) and abs(re[0] - rs[0]) > 0.2 * abs(rs[0] - ls[0])
        knee_locked = abs(lk[1] - rk[1]) < 0.05 * frame_shape[0]
        hip_level = abs(lh[1] - rh[1]) < 0.05 * frame_shape[0]

        score = 0
        if elbows_out:
            score += 35
        if knee_locked:
            score += 30
        if hip_level:
            score += 35

        tips.append(f"Score: {score}/100")
        if not elbows_out:
            tips.append("Flare elbows to frame the abs.")
        if not knee_locked:
            tips.append("Flex quads and lock knees gently.")
        if not hip_level:
            tips.append("Square hips to the front.")
        if score >= 90:
            tips.append("Sharp abs and thigh pose! Hold steady.")
        return tips