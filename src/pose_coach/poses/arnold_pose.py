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


class ArnoldPoseRater(Analyzer):
    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Pose - Arnold"]
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

        # Approximate: one arm overhead, other flexed across torso (simplified)
        # Check left arm overhead: elbow above shoulder and wrist above elbow
        left_overhead = (le[1] < ls[1]) and (lw[1] < le[1])
        right_overhead = (re[1] < rs[1]) and (rw[1] < re[1])

        # Flexed arm: elbow bent ~90-120 deg
        left_elbow_angle = angle_at_point(ls, le, lw)
        right_elbow_angle = angle_at_point(rs, re, rw)
        left_flexed = 70 <= left_elbow_angle <= 120
        right_flexed = 70 <= right_elbow_angle <= 120

        # Choose configuration that best matches Arnold pose
        # Case A: left overhead, right flexed; Case B: right overhead, left flexed
        case_a = left_overhead and right_flexed
        case_b = right_overhead and left_flexed

        score = 0
        if case_a or case_b:
            score = 70
        if (left_overhead and left_flexed) or (right_overhead and right_flexed):
            # If the same arm is both overhead and flexed, it's less ideal
            score = max(score, 40)

        # Symmetry: shoulder heights relatively level
        symmetry = abs(ls[1] - rs[1]) < 0.1 * np.linalg.norm(ls - lh)
        if symmetry:
            score += 15

        # Elbow positions relative to torso width
        torso_width = abs(rs[0] - ls[0]) + 1e-6
        elbows_out = (abs(le[0] - ls[0]) > 0.3 * torso_width) and (abs(re[0] - rs[0]) > 0.3 * torso_width)
        if elbows_out:
            score += 15

        score = min(100, score)

        tips.append(f"Score: {score}/100")
        if not (case_a or case_b):
            tips.append("Lift one arm overhead; flex the other across torso (~90-110°).")
        if not symmetry:
            tips.append("Keep shoulders level for a clean look.")
        if not elbows_out:
            tips.append("Flare elbows slightly to enhance silhouette.")
        if score >= 90:
            tips.append("Iconic! Great Arnold pose.")

        return tips