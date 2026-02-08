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


class SideTricepsRater(Analyzer):
    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Pose - Side Triceps"]
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
        extended = elbow_angle_left > 150 or elbow_angle_right > 150
        elbow_near_torso = abs(((le + re) / 2.0)[0] - ((ls + rs) / 2.0)[0]) < 0.12 * frame_shape[1]

        score = 0
        if extended:
            score += 60
        if elbow_near_torso:
            score += 40

        tips.append(f"Score: {score}/100")
        if not extended:
            tips.append("Extend the arm to showcase triceps definition.")
        if not elbow_near_torso:
            tips.append("Keep elbow close to torso for a tight pose.")
        if score >= 90:
            tips.append("Sharp side triceps! Hold steady.")
        return tips