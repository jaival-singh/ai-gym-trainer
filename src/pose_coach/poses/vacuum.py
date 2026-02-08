from typing import List

import numpy as np

from ..feedback import Analyzer

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


class VacuumPoseRater(Analyzer):
    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Pose - Vacuum"]
        if landmarks_px is None:
            tips.append("No person detected.")
            return tips

        ls = landmarks_px[LEFT_SHOULDER].astype(float)
        rs = landmarks_px[RIGHT_SHOULDER].astype(float)
        lh = landmarks_px[LEFT_HIP].astype(float)
        rh = landmarks_px[RIGHT_HIP].astype(float)

        shoulder_y = ((ls + rs) / 2.0)[1]
        hip_y = ((lh + rh) / 2.0)[1]

        ribcage_lift = hip_y - shoulder_y > 0.3 * frame_shape[0]
        stomach_draw_in = hip_y - shoulder_y > 0.35 * frame_shape[0]

        score = 0
        if ribcage_lift:
            score += 50
        if stomach_draw_in:
            score += 50

        tips.append(f"Score: {score}/100")
        if not ribcage_lift:
            tips.append("Lift ribcage by expanding chest upward.")
        if not stomach_draw_in:
            tips.append("Pull stomach in tightly to emphasize vacuum.")
        if score >= 90:
            tips.append("Classic vacuum! Hold steady.")
        return tips