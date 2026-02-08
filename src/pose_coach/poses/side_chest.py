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
LEFT_HIP = 23
RIGHT_HIP = 24


class SideChestRater(Analyzer):
    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Pose - Side Chest"]
        if landmarks_px is None:
            tips.append("No person detected.")
            return tips

        ls = landmarks_px[LEFT_SHOULDER].astype(float)
        rs = landmarks_px[RIGHT_SHOULDER].astype(float)
        le = landmarks_px[LEFT_ELBOW].astype(float)
        re = landmarks_px[RIGHT_ELBOW].astype(float)
        lw = landmarks_px[LEFT_WRIST].astype(float)
        rw = landmarks_px[RIGHT_WRIST].astype(float)
        lh = landmarks_px[LEFT_HIP].astype(float)
        rh = landmarks_px[RIGHT_HIP].astype(float)

        elbow_angle_left = angle_at_point(ls, le, lw)
        elbow_angle_right = angle_at_point(rs, re, rw)
        arms_flexed = (70 <= elbow_angle_left <= 120) and (70 <= elbow_angle_right <= 120)
        chest_up = ((ls + rs) / 2.0)[1] < ((lh + rh) / 2.0)[1] - 0.08 * frame_shape[0]

        score = 0
        if arms_flexed:
            score += 60
        if chest_up:
            score += 40

        tips.append(f"Score: {score}/100")
        if not arms_flexed:
            tips.append("Flex arms to frame the chest (~90-110°).")
        if not chest_up:
            tips.append("Lift chest and retract scapula.")
        if score >= 90:
            tips.append("Classic side chest! Hold and smile.")
        return tips