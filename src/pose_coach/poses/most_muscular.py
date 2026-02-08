from typing import List

import numpy as np

from ..feedback import Analyzer

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14


class MostMuscularRater(Analyzer):
    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Pose - Most Muscular"]
        if landmarks_px is None:
            tips.append("No person detected.")
            return tips

        ls = landmarks_px[LEFT_SHOULDER].astype(float)
        rs = landmarks_px[RIGHT_SHOULDER].astype(float)
        le = landmarks_px[LEFT_ELBOW].astype(float)
        re = landmarks_px[RIGHT_ELBOW].astype(float)

        shoulder_center_x = ((ls + rs) / 2.0)[0]
        elbows_in = abs(le[0] - shoulder_center_x) < 0.2 * abs(rs[0] - ls[0]) and abs(re[0] - shoulder_center_x) < 0.2 * abs(rs[0] - ls[0])
        shoulders_forward = ((ls + rs) / 2.0)[1] < min(ls[1], rs[1]) + 0.05 * frame_shape[0]

        score = 0
        if elbows_in:
            score += 60
        if shoulders_forward:
            score += 40

        tips.append(f"Score: {score}/100")
        if not elbows_in:
            tips.append("Bring elbows in to crunch the chest and traps.")
        if not shoulders_forward:
            tips.append("Lean slightly forward and contract hard.")
        if score >= 90:
            tips.append("Beast mode! Hold the most muscular.")
        return tips