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


class DoubleBicepsRater(Analyzer):
    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Pose - Double Biceps"]
        if landmarks_px is None:
            tips.append("No person detected. Step back and ensure upper body in frame.")
            return tips

        ls = landmarks_px[LEFT_SHOULDER].astype(float)
        rs = landmarks_px[RIGHT_SHOULDER].astype(float)
        le = landmarks_px[LEFT_ELBOW].astype(float)
        re = landmarks_px[RIGHT_ELBOW].astype(float)
        lw = landmarks_px[LEFT_WRIST].astype(float)
        rw = landmarks_px[RIGHT_WRIST].astype(float)
        lh = landmarks_px[LEFT_HIP].astype(float)
        rh = landmarks_px[RIGHT_HIP].astype(float)

        # Target: arms up and out, elbows roughly at shoulder level, elbows ~90-120 deg
        left_elbow_angle = angle_at_point(ls, le, lw)
        right_elbow_angle = angle_at_point(rs, re, rw)

        # Elbow height relative to shoulder
        left_elbow_level = abs(le[1] - ls[1]) < 0.12 * np.linalg.norm(ls - lh)
        right_elbow_level = abs(re[1] - rs[1]) < 0.12 * np.linalg.norm(rs - rh)

        # Symmetry: horizontal distances of elbows from shoulders
        left_span = abs(le[0] - ls[0])
        right_span = abs(re[0] - rs[0])
        symmetry = abs(left_span - right_span) / (max(1.0, (left_span + right_span) / 2.0)) < 0.25

        # Elbow flexion score
        left_flex_ok = 70 <= left_elbow_angle <= 120
        right_flex_ok = 70 <= right_elbow_angle <= 120

        # Composite score
        score = 0
        if left_elbow_level:
            score += 25
        if right_elbow_level:
            score += 25
        if left_flex_ok:
            score += 25
        if right_flex_ok:
            score += 25
        if symmetry:
            score += 10
        score = min(100, score)

        tips.append(f"Score: {score}/100")
        if not left_elbow_level or not right_elbow_level:
            tips.append("Lift elbows to shoulder height.")
        if not left_flex_ok or not right_flex_ok:
            tips.append("Flex elbows ~90-110° and squeeze biceps.")
        if not symmetry:
            tips.append("Match left/right spread for symmetry.")
        if left_elbow_level and right_elbow_level and left_flex_ok and right_flex_ok and symmetry:
            tips.append("Great pose! Hold and breathe.")

        return tips