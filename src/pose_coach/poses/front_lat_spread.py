from typing import List

import numpy as np

from ..feedback import Analyzer

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


class FrontLatSpreadRater(Analyzer):
    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Pose - Front Lat Spread"]
        if landmarks_px is None:
            tips.append("No person detected.")
            return tips

        ls = landmarks_px[LEFT_SHOULDER].astype(float)
        rs = landmarks_px[RIGHT_SHOULDER].astype(float)
        le = landmarks_px[LEFT_ELBOW].astype(float)
        re = landmarks_px[RIGHT_ELBOW].astype(float)
        lh = landmarks_px[LEFT_HIP].astype(float)
        rh = landmarks_px[RIGHT_HIP].astype(float)

        shoulder_width = abs(rs[0] - ls[0])
        elbow_flare_left = abs(le[0] - ls[0]) > 0.4 * shoulder_width
        elbow_flare_right = abs(re[0] - rs[0]) > 0.4 * shoulder_width
        elbows_level = abs(le[1] - re[1]) < 0.08 * frame_shape[0]

        score = 0
        if elbow_flare_left:
            score += 30
        if elbow_flare_right:
            score += 30
        if elbows_level:
            score += 40

        tips.append(f"Score: {score}/100")
        if not elbow_flare_left or not elbow_flare_right:
            tips.append("Flare elbows to widen lats.")
        if not elbows_level:
            tips.append("Keep elbows at even height.")
        if score >= 90:
            tips.append("Impressive width! Hold the spread.")
        return tips