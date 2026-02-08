# Pose Coach: Real-time Gym Form Assistant and Pose Rater

A real-time camera assistant that helps correct exercise form (e.g., bicep curls, barbell rows) and rates bodybuilding poses (e.g., double biceps, Arnold pose) to reduce injury risk and improve technique.

## Features
- Real-time pose tracking with MediaPipe
- Exercise feedback and rep counting (bicep curl, barbell row, squat, deadlift, more)
- Bodybuilding pose rating (double biceps, Arnold, lat spreads, more)
- On-screen guidance overlays and status messages
- Web UI powered by Gradio

## Setup
```bash
cd pose_coach
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (CLI)
- Exercise assistant (webcam index 0):
```bash
python -m src.main --mode exercise --exercise bicep_curl --camera-index 0
```
- Pose rating:
```bash
python -m src.main --mode pose --pose double_biceps --camera-index 0
```

## Run (Web UI)
```bash
python -m src.web.app
```
Then open the local URL shown in the terminal (e.g., http://127.0.0.1:7860). Select Mode and Exercise/Pose, allow webcam permissions, and you’ll see the overlay guidance.

Press `q` to quit the CLI window.

## Notes
- Ensure good lighting and full body visible for best results.
- This tool provides guidance and is not a substitute for professional coaching.
