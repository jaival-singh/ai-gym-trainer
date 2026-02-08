# Pose Coach  
Real-Time Form Checker + Bodybuilding Pose Judge

Pose Coach uses your webcam to analyze movement in real time. It counts reps, spots common form mistakes during exercises, and scores classic bodybuilding poses for symmetry, angles, and presentation — helping you train smarter and safer.

Powered by MediaPipe Pose.

## Key Features

- Live rep counting & movement phase detection  
- Instant form feedback with practical corrections  
- Visual overlays showing reps, tips, and status  
- Support for many compound & isolation exercises  
- Bodybuilding pose evaluation (front/back/side mandatory poses)  
- Simple CLI + modern Gradio web interface

## Supported Exercises

- Bicep Curl  
- Leg Press  
- Squat  
- Lunge  
- Romanian Deadlift (RDL)  
- Deadlift  
- Bench Press  
- Bent-Over Row / Barbell Row  
- Lat Pulldown  
- Push-up  
- Lateral Raise  
- Calf Raise  
- Box Jump  
- Cable Woodchopper  

## Supported Poses

- Double Biceps (front & back)  
- Arnold Pose  
- Front Lat Spread  
- Rear Lat Spread  
- Side Chest  
- Side Triceps  
- Abdominals & Thighs  
- Most Muscular  
- Vacuum  
- Moon Pose  
- Quarter Turns  

## Quick Setup

```bash
cd pose_coach
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```
## Run CLI

### Exercise Mode
``` bash
python -m src.main --mode exercise --exercise bicep_curl
python -m src.main --mode exercise --exercise squat --width 1280 --height 720
```
### Pose Mode
```
python -m src.main --mode pose --pose double_biceps

Run (Web UI)
```
python -m src.web.app
