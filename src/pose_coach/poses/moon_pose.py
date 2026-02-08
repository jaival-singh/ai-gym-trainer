from typing import List

import numpy as np

from ..feedback import Analyzer

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


class MoonPoseRater(Analyzer):
    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Pose - Moon Pose"]
        if landmarks_px is None:
            tips.append("No person detected.")
            return tips

        ls = landmarks_px[LEFT_SHOULDER].astype(float)
        rs = landmarks_px[RIGHT_SHOULDER].astype(float)
        lw = landmarks_px[LEFT_WRIST].astype(float)
        rw = landmarks_px[RIGHT_WRIST].astype(float)
        lh = landmarks_px[LEFT_HIP].astype(float)
        rh = landmarks_px[RIGHT_HIP].astype(float)

        torso_tilt = abs(((ls + rs) / 2.0)[0] - ((lh + rh) / 2.0)[0]) > 0.08 * frame_shape[1]
        arm_reach = abs(lw[1] - rw[1]) > 0.15 * frame_shape[0]

        score = 0
        if torso_tilt:
            score += 50
        if arm_reach:
            score += 50

        tips.append(f"Score: {score}/100")
        if not torso_tilt:
            tips.append("Add a side bend to highlight obliques.")
        if not arm_reach:
            tips.append("Reach with the upper arm to elongate the line.")
        if score >= 90:
            tips.append("Beautiful moon pose! Hold the line.")
        return tips