import argparse
import time
from typing import Optional

import cv2
import gradio as gr
import numpy as np

from ..pose_coach.utils import init_pose_estimator, extract_landmarks
from ..pose_coach.drawing import draw_landmarks_and_info
from ..pose_coach.exercises.bicep_curl import BicepCurlCoach
from ..pose_coach.exercises.barbell_row import BarbellRowCoach
from ..pose_coach.exercises.squats import SquatCoach
from ..pose_coach.exercises.lunges import LungeCoach
from ..pose_coach.exercises.rdl import RDLCoach
from ..pose_coach.exercises.leg_press import LegPressCoach
from ..pose_coach.exercises.calf_raises import CalfRaiseCoach
from ..pose_coach.exercises.pushups import PushupCoach
from ..pose_coach.exercises.pull_downs import PullDownCoach
from ..pose_coach.exercises.bench_press import BenchPressCoach
from ..pose_coach.exercises.bent_over_rows import BentOverRowCoach
from ..pose_coach.exercises.lateral_raises import LateralRaiseCoach
from ..pose_coach.exercises.deadlift import DeadliftCoach
from ..pose_coach.exercises.box_jumps import BoxJumpCoach
from ..pose_coach.exercises.cable_woodchoppers import CableWoodchopperCoach
from ..pose_coach.poses.double_biceps import DoubleBicepsRater
from ..pose_coach.poses.arnold_pose import ArnoldPoseRater
from ..pose_coach.poses.quarter_turns import QuarterTurnsRater
from ..pose_coach.poses.front_lat_spread import FrontLatSpreadRater
from ..pose_coach.poses.back_lat_spread import RearLatSpreadRater
from ..pose_coach.poses.side_chest import SideChestRater
from ..pose_coach.poses.side_triceps import SideTricepsRater
from ..pose_coach.poses.ab_thigh import AbdominalsAndThighsRater
from ..pose_coach.poses.most_muscular import MostMuscularRater
from ..pose_coach.poses.vacuum import VacuumPoseRater
from ..pose_coach.poses.moon_pose import MoonPoseRater


EXERCISES = [
    ("bicep_curl", BicepCurlCoach),
    ("barbell_row", BarbellRowCoach),
    ("squat", SquatCoach),
    ("lunge", LungeCoach),
    ("rdl", RDLCoach),
    ("leg_press", LegPressCoach),
    ("calf_raise", CalfRaiseCoach),
    ("pushup", PushupCoach),
    ("pull_down", PullDownCoach),
    ("bench_press", BenchPressCoach),
    ("bent_over_row", BentOverRowCoach),
    ("lateral_raise", LateralRaiseCoach),
    ("deadlift", DeadliftCoach),
    ("box_jump", BoxJumpCoach),
    ("cable_woodchopper", CableWoodchopperCoach),
]
POSES = [
    ("double_biceps", DoubleBicepsRater),
    ("arnold", ArnoldPoseRater),
    ("quarter_turns", QuarterTurnsRater),
    ("front_lat_spread", FrontLatSpreadRater),
    ("rear_lat_spread", RearLatSpreadRater),
    ("side_chest", SideChestRater),
    ("side_triceps", SideTricepsRater),
    ("ab_thigh", AbdominalsAndThighsRater),
    ("most_muscular", MostMuscularRater),
    ("vacuum", VacuumPoseRater),
    ("moon_pose", MoonPoseRater),
]


def build_analyzer(mode: str, name: str):
    if mode == "exercise":
        mapping = dict(EXERCISES)
    else:
        mapping = dict(POSES)
    cls = mapping.get(name)
    if not cls:
        raise gr.Error("Unknown analyzer selection")
    return cls()


def app():
    with gr.Blocks(title="Pose Coach") as demo:
        gr.Markdown("## Ai-Gym-Trainer")
        with gr.Row():
            mode = gr.Dropdown(["exercise", "pose"], value="exercise", label="Mode")
            exercise = gr.Dropdown([name for name, _ in EXERCISES], value="bicep_curl", label="Exercise")
            pose = gr.Dropdown([name for name, _ in POSES], value="double_biceps", label="Pose")
        cam = gr.Video(streaming=True, label="Webcam", height=480)
        out = gr.Image(label="Output", type="numpy")

        state = gr.State({"analyzer": None, "pose": None})

        def init(mode_val, ex_val, pose_val):
            analyzer = build_analyzer(mode_val, ex_val if mode_val == "exercise" else pose_val)
            pose_model = init_pose_estimator(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return {"analyzer": analyzer, "pose": pose_model}

        def process_frame(frame, mode_val, ex_val, pose_val, st):
            if frame is None:
                return None, st
            if st.get("analyzer") is None:
                st = init(mode_val, ex_val, pose_val)
            analyzer = st["analyzer"]
            pose_model = st["pose"]

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = pose_model.process(frame)
            landmarks_px, visibility = extract_landmarks(results, frame_bgr.shape)
            overlay = analyzer.update(landmarks_px, visibility, frame_bgr.shape)
            out_img = draw_landmarks_and_info(frame_bgr, results, overlay, fps=0.0)
            return cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), st

        cam.change(process_frame, inputs=[cam, mode, exercise, pose, state], outputs=[out, state])
        mode.change(lambda m: None, inputs=mode, outputs=None)
        exercise.change(lambda e: None, inputs=exercise, outputs=None)
        pose.change(lambda p: None, inputs=pose, outputs=None)

    return demo


if __name__ == "__main__":
    demo = app()
    demo.launch()