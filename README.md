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

## How it all works together (beginner explanation)

This project combines four technologies to detect emotions in real time. Here is what each one does and how they connect:

```
Webcam → OpenCV → FER (+ MTCNN + TensorFlow) → OpenCV draws results on screen
```

### OpenCV — the eyes and hands of the program

OpenCV (Open Source Computer Vision) is a library that lets Python talk to your webcam and work with images. A video is just a sequence of images (called **frames**) shown very fast — typically 30 per second. OpenCV grabs each frame from your webcam, hands it off for analysis, and then draws the results (boxes, labels, FPS counter) back onto that frame before showing it in the window.

In this project: [capture.py](capture.py) uses OpenCV to open the camera and read frames. [overlay.py](overlay.py) uses OpenCV to draw the corner brackets and emotion labels on screen.

### FER — the emotion classifier

FER (Facial Expression Recognition) is a Python library that wraps a pre-trained machine learning model. You give it an image and it tells you where faces are and what emotion each face is showing (happy, sad, angry, etc.) along with a confidence score like `87%`.

In this project: [emotion.py](emotion.py) receives each frame from the camera, passes it to FER, and returns a list of detected faces with their dominant emotion and score.

### MTCNN — the face finder

MTCNN (Multi-Task Cascaded Convolutional Networks) is a neural network specifically trained to find faces in images. It's more accurate than older methods, especially for faces that are at an angle or partially lit. FER uses MTCNN internally before classifying emotions — first find the face, then analyze its expression.

### TensorFlow — the engine underneath

TensorFlow is a machine learning framework made by Google. It is what actually runs the math behind MTCNN and the emotion model. You don't interact with TensorFlow directly — FER uses it under the hood. This is why the first run downloads ~500 MB: those are the pre-trained model weights (the learned knowledge) that TensorFlow needs to load.

### The loop — how it all runs together

[main.py](main.py) ties everything together in a continuous loop:

1. **Capture** — read one frame from the webcam (OpenCV)
2. **Detect** — send the frame to FER, get back face locations and emotions
3. **Draw** — paint the results onto the frame (OpenCV)
4. **Show** — display the frame in a window (OpenCV)
5. **Repeat** — go back to step 1, ~30 times per second

The FPS (frames per second) counter in the top-left corner tells you how fast the loop is running. Emotion detection is the slowest step because it runs a neural network on every frame.

---

## macOS camera permissions

If the camera doesn't open, grant access in:
**System Settings → Privacy & Security → Camera → Terminal** (or your IDE)

## Privacy

All processing is 100% local. No video frames or images are sent to any external server.
