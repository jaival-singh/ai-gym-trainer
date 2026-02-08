from typing import List

import numpy as np

from ..feedback import Analyzer
from ..angles import angle_at_point, angle_between_vectors

# Landmarks
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26


class BarbellRowCoach(Analyzer):
    def __init__(self):
        self.reps = 0
        self.state = "down"  # bar down -> pull up

    def update(self, landmarks_px, visibility, frame_shape) -> list:
        tips: List[str] = ["Mode: Exercise - Barbell Row"]
        if landmarks_px is None:
            tips.append("No person detected. Step back and ensure full body in frame.")
            tips.append(f"Reps: {self.reps}")
            return tips

        left_side_vis = float(visibility[LEFT_ELBOW]) if visibility is not None else 0.0
        right_side_vis = float(visibility[RIGHT_ELBOW]) if visibility is not None else 0.0
        use_left = left_side_vis >= right_side_vis

        if use_left:
            shoulder = landmarks_px[LEFT_SHOULDER].astype(float)
            elbow = landmarks_px[LEFT_ELBOW].astype(float)
            wrist = landmarks_px[LEFT_WRIST].astype(float)
            hip = landmarks_px[LEFT_HIP].astype(float)
            knee = landmarks_px[LEFT_KNEE].astype(float)
            other_shoulder = landmarks_px[RIGHT_SHOULDER].astype(float)
        else:
            shoulder = landmarks_px[RIGHT_SHOULDER].astype(float)
            elbow = landmarks_px[RIGHT_ELBOW].astype(float)
            wrist = landmarks_px[RIGHT_WRIST].astype(float)
            hip = landmarks_px[RIGHT_HIP].astype(float)
            knee = landmarks_px[RIGHT_KNEE].astype(float)
            other_shoulder = landmarks_px[LEFT_SHOULDER].astype(float)

        # Hip hinge: torso angle vs horizontal. We approximate torso vector as shoulder->hip
        torso_vec = hip - shoulder
        horizontal_vec = np.array([1.0, 0.0])
        torso_angle_from_horizontal = angle_between_vectors(torso_vec, horizontal_vec)
        # Neutral hinge typically ~ 20-45 deg above horizontal (i.e., torso leaned forward)
        hip_hinge_ok = 20 <= torso_angle_from_horizontal <= 60

        # Neutral spine: shoulders roughly level (small shoulder-to-shoulder slope)
        shoulder_slope = abs(other_shoulder[1] - shoulder[1]) / (abs(other_shoulder[0] - shoulder[0]) + 1e-6)
        neutral_spine = shoulder_slope < 0.4

        # Elbow path: wrist under elbow at top; vertical pull
        elbow_angle = angle_at_point(shoulder, elbow, wrist)
        bar_up = elbow_angle < 70
        bar_down = elbow_angle > 140
        if self.state == "down" and bar_up:
            self.state = "up"
            self.reps += 1
        elif self.state == "up" and bar_down:
            self.state = "down"

        # Wrist under elbow check
        wrist_under_elbow = abs(wrist[0] - elbow[0]) < 0.4 * np.linalg.norm(elbow - shoulder)

        tips.append(f"Reps: {self.reps} | Phase: {self.state}")

        if not hip_hinge_ok:
            tips.append("Hinge more at the hips; keep torso ~30-45° to the floor.")
        if not neutral_spine:
            tips.append("Keep spine neutral; avoid twisting or rounding.")
        if not wrist_under_elbow:
            tips.append("Pull elbows back; keep wrist under elbow for a straight path.")

        if hip_hinge_ok and neutral_spine and wrist_under_elbow:
            tips.append("Nice rows! Strong positioning.")

        return tips