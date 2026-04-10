# Emotion Detector

Python desktop app that opens the webcam, detects faces, and classifies emotions in real time using FER + OpenCV.

**Requires Python 3.10+**

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note (fer 25.x):** This project pins `fer==25.10.3`. At this version the public `from fer import FER` raises `ImportError`; `emotion.py` imports from `fer.fer` directly. Do not upgrade `fer` without verifying compatibility.

> First run downloads TensorFlow model weights (~500 MB). This is a one-time operation.

## Run

```bash
python3 -m venv venv
source venv/bin/activate
python3 main.py
```

| Key | Action |
|-----|--------|
| `q` | Quit |
| `ESC` | Quit |
| `Ctrl+C` | Quit |

## macOS camera permissions

If the camera doesn't open, grant access in:
**System Settings → Privacy & Security → Camera → Terminal** (or your IDE)

## Privacy

All processing is 100% local. No video frames or images are sent to any external server.
