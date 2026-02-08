from typing import List

import numpy as np

from ..feedback import Analyzer

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


class QuarterTurnsRater(Analyzer):
    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Pose - Quarter Turns"]
        if landmarks_px is None:
            tips.append("No person detected.")
            return tips

        ls = landmarks_px[LEFT_SHOULDER].astype(float)
        rs = landmarks_px[RIGHT_SHOULDER].astype(float)
        lh = landmarks_px[LEFT_HIP].astype(float)
        rh = landmarks_px[RIGHT_HIP].astype(float)

        shoulders_level = abs(ls[1] - rs[1]) < 0.05 * frame_shape[0]
        hips_level = abs(lh[1] - rh[1]) < 0.05 * frame_shape[0]
        torso_upright = abs(((ls + rs) / 2.0)[0] - ((lh + rh) / 2.0)[0]) < 0.06 * frame_shape[1]

        score = 0
        if shoulders_level:
            score += 35
        if hips_level:
            score += 35
        if torso_upright:
            score += 30

        tips.append(f"Score: {score}/100")
        if not shoulders_level:
            tips.append("Level your shoulders.")
        if not hips_level:
            tips.append("Square and level hips.")
        if not torso_upright:
            tips.append("Stand tall; avoid leaning.")
        if score >= 90:
            tips.append("Clean quarter turn. Hold steady.")
        return tips