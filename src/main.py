import argparse
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .pose_coach.utils import init_pose_estimator, extract_landmarks
from .pose_coach.drawing import draw_landmarks_and_info
from .pose_coach.exercises.bicep_curl import BicepCurlCoach
from .pose_coach.exercises.barbell_row import BarbellRowCoach
from .pose_coach.poses.double_biceps import DoubleBicepsRater
from .pose_coach.poses.arnold_pose import ArnoldPoseRater


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose Coach: Real-time exercise assistant and pose rater")
    parser.add_argument("--mode", choices=["exercise", "pose"], required=True, help="Run exercise assistant or pose rater")
    parser.add_argument("--exercise", choices=[
        "bicep_curl","barbell_row","squat","lunge","rdl","leg_press","calf_raise",
        "pushup","pull_down","bench_press","bent_over_row","lateral_raise","deadlift",
        "box_jump","cable_woodchopper"
    ], help="Exercise to coach in exercise mode")
    parser.add_argument("--pose", choices=[
        "double_biceps","arnold","quarter_turns","front_lat_spread","side_chest",
        "back_double_biceps","rear_lat_spread","side_triceps","ab_thigh","most_muscular",
        "vacuum","moon_pose"
    ], help="Pose to rate in pose mode")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def get_coach_or_rater(args):
    if args.mode == "exercise":
        if args.exercise == "bicep_curl":
            return BicepCurlCoach()
        if args.exercise == "barbell_row":
            return BarbellRowCoach()
        if args.exercise == "squat":
            from .pose_coach.exercises.squats import SquatCoach
            return SquatCoach()
        if args.exercise == "lunge":
            from .pose_coach.exercises.lunges import LungeCoach
            return LungeCoach()
        if args.exercise == "rdl":
            from .pose_coach.exercises.rdl import RDLCoach
            return RDLCoach()
        if args.exercise == "leg_press":
            from .pose_coach.exercises.leg_press import LegPressCoach
            return LegPressCoach()
        if args.exercise == "calf_raise":
            from .pose_coach.exercises.calf_raises import CalfRaiseCoach
            return CalfRaiseCoach()
        if args.exercise == "pushup":
            from .pose_coach.exercises.pushups import PushupCoach
            return PushupCoach()
        if args.exercise == "pull_down":
            from .pose_coach.exercises.pull_downs import PullDownCoach
            return PullDownCoach()
        if args.exercise == "bench_press":
            from .pose_coach.exercises.bench_press import BenchPressCoach
            return BenchPressCoach()
        if args.exercise == "bent_over_row":
            from .pose_coach.exercises.bent_over_rows import BentOverRowCoach
            return BentOverRowCoach()
        if args.exercise == "lateral_raise":
            from .pose_coach.exercises.lateral_raises import LateralRaiseCoach
            return LateralRaiseCoach()
        if args.exercise == "deadlift":
            from .pose_coach.exercises.deadlift import DeadliftCoach
            return DeadliftCoach()
        if args.exercise == "box_jump":
            from .pose_coach.exercises.box_jumps import BoxJumpCoach
            return BoxJumpCoach()
        if args.exercise == "cable_woodchopper":
            from .pose_coach.exercises.cable_woodchoppers import CableWoodchopperCoach
            return CableWoodchopperCoach()
        raise ValueError("Unknown exercise")
    else:
        if args.pose == "double_biceps":
            return DoubleBicepsRater()
        if args.pose == "arnold":
            return ArnoldPoseRater()
        if args.pose == "quarter_turns":
            from .pose_coach.poses.quarter_turns import QuarterTurnsRater
            return QuarterTurnsRater()
        if args.pose == "front_lat_spread":
            from .pose_coach.poses.front_lat_spread import FrontLatSpreadRater
            return FrontLatSpreadRater()
        if args.pose == "back_double_biceps":
            from .pose_coach.poses.double_biceps import DoubleBicepsRater
            return DoubleBicepsRater()
        if args.pose == "rear_lat_spread":
            from .pose_coach.poses.back_lat_spread import RearLatSpreadRater
            return RearLatSpreadRater()
        if args.pose == "side_chest":
            from .pose_coach.poses.side_chest import SideChestRater
            return SideChestRater()
        if args.pose == "side_triceps":
            from .pose_coach.poses.side_triceps import SideTricepsRater
            return SideTricepsRater()
        if args.pose == "ab_thigh":
            from .pose_coach.poses.ab_thigh import AbdominalsAndThighsRater
            return AbdominalsAndThighsRater()
        if args.pose == "most_muscular":
            from .pose_coach.poses.most_muscular import MostMuscularRater
            return MostMuscularRater()
        if args.pose == "vacuum":
            from .pose_coach.poses.vacuum import VacuumPoseRater
            return VacuumPoseRater()
        if args.pose == "moon_pose":
            from .pose_coach.poses.moon_pose import MoonPoseRater
            return MoonPoseRater()
        raise ValueError("Unknown pose")


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Unable to open camera. Try a different --camera-index")

    pose = init_pose_estimator(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    analyzer = get_coach_or_rater(args)

    fps_smoothing = 0.9
    fps = 0.0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        landmarks_px, visibility = extract_landmarks(results, frame.shape)

        overlay_text = analyzer.update(landmarks_px, visibility, frame.shape)

        # FPS
        now = time.time()
        inst_fps = 1.0 / max(1e-6, (now - last_time))
        last_time = now
        fps = fps_smoothing * fps + (1 - fps_smoothing) * inst_fps

        output_frame = draw_landmarks_and_info(frame, results, overlay_text, fps)

        cv2.imshow("Pose Coach", output_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()