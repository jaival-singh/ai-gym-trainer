from typing import List

import numpy as np

from ..feedback import Analyzer
from ..angles import angle_at_point

# MediaPipe landmark indices for readability
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


class BicepCurlCoach(Analyzer):
    def __init__(self):
        self.reps = 0
        self.state = "down"  # down -> up
        self.side_hint = "auto"

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Bicep Curl"]
        if landmarks_px is None:
            tips.append("No person detected. Step back and ensure full body in frame.")
            tips.append(f"Reps: {self.reps}")
            return tips

        # Choose side with better visibility
        left_vis = float(visibility[LEFT_ELBOW]) if visibility is not None else 0.0
        right_vis = float(visibility[RIGHT_ELBOW]) if visibility is not None else 0.0
        use_left = left_vis >= right_vis

        if use_left:
            shoulder, elbow, wrist, hip = (
                landmarks_px[LEFT_SHOULDER],
                landmarks_px[LEFT_ELBOW],
                landmarks_px[LEFT_WRIST],
                landmarks_px[LEFT_HIP],
            )
            side = "Left"
        else:
            shoulder, elbow, wrist, hip = (
                landmarks_px[RIGHT_SHOULDER],
                landmarks_px[RIGHT_ELBOW],
                landmarks_px[RIGHT_WRIST],
                landmarks_px[RIGHT_HIP],
            )
            side = "Right"

        # Angles
        elbow_angle = angle_at_point(shoulder.astype(float), elbow.astype(float), wrist.astype(float))
        shoulder_angle = angle_at_point(hip.astype(float), shoulder.astype(float), elbow.astype(float))

        # Rep counting thresholds
        at_bottom = elbow_angle > 150
        at_top = elbow_angle < 50

        if self.state == "down" and at_top:
            self.state = "up"
            self.reps += 1
        elif self.state == "up" and at_bottom:
            self.state = "down"

        # Form checks
        # 1) Elbow should stay near torso: shoulder-elbow vertical alignment (x-distance small compared to upper arm length)
        upper_arm_len = np.linalg.norm(shoulder - elbow) + 1e-6
        elbow_torso_dx = abs(elbow[0] - shoulder[0])
        elbow_stable = elbow_torso_dx < 0.6 * upper_arm_len

        # 2) Shoulder should remain stable (avoid swinging): shoulder angle shouldn't exceed ~60 deg at top
        shoulder_stable = shoulder_angle < 70

        # 3) Full ROM: At bottom, elbow_angle should exceed 150; at top, below 50
        rom_ok = at_top or at_bottom

        tips.append(f"Side: {side} | Reps: {self.reps} | Phase: {self.state}")

        if not elbow_stable:
            tips.append("Keep your elbow close; avoid drifting forward/back.")
        if not shoulder_stable:
            tips.append("Avoid swinging shoulders; isolate the biceps.")
        if not rom_ok:
            tips.append("Use full range: extend at bottom and squeeze at top.")

        if elbow_stable and shoulder_stable and rom_ok:
            tips.append("Great form! Keep it up.")

        return tips