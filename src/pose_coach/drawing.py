from typing import List, Optional

import cv2

from .utils import mp_drawing, mp_drawing_styles


def draw_landmarks_and_info(frame, results, overlay_lines: Optional[list], fps: float):
    output = frame.copy()
    if results and results.pose_landmarks:
        mp_drawing.draw_landmarks(
            output,
            results.pose_landmarks,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    # Draw overlay text lines
    if overlay_lines:
        y = 30
        for line in overlay_lines:
            cv2.putText(output, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y += 28

    cv2.putText(output, f"FPS: {fps:.1f}", (output.shape[1] - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return output