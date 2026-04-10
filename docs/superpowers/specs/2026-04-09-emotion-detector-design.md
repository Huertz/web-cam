# Emotion Detector — Design Spec
Date: 2026-04-09

## Overview

Python desktop app that opens the webcam, detects faces in real time, classifies the dominant emotion per face, and renders a HUD-style overlay on an OpenCV window. Intended as a personal learning demo for computer vision concepts.

---

## Context

- **Use case:** Personal demo / learning — single user, local machine
- **Platform:** macOS, Python 3.10+
- **Performance target:** 8–15 FPS on CPU (acceptable for demo; fluidity not critical)
- **Privacy:** 100% local processing, no frames leave the machine

---

## Architecture

Modular — 4 files with single responsibilities, orchestrated by `main.py`.

```
web-cam/
├── main.py          # entrypoint: main loop, orchestrates modules
├── capture.py       # CameraCapture: opens/releases camera, delivers frames
├── emotion.py       # EmotionDetector: receives frame, returns detection results
├── overlay.py       # draw_overlay(): draws HUD over frame
├── requirements.txt
└── README.md
```

### Data flow

```
CameraCapture.read() → frame (np.ndarray)
       ↓
EmotionDetector.detect(frame) → [{"box": (x,y,w,h), "emotion": str, "score": float}, ...]
       ↓
draw_overlay(frame, results) → frame with HUD
       ↓
cv2.imshow() → window
       ↓
waitKey → check q / ESC → exit or continue
```

---

## Module Details

### `capture.py` — `CameraCapture`

- `__init__(device_index=0, width=640, height=480)` — opens `cv2.VideoCapture`, sets resolution
- `read() → (ok: bool, frame: np.ndarray)` — reads one frame
- `release()` — releases camera resource
- Raises `RuntimeError("No se pudo abrir la cámara índice X")` if camera fails to open
- macOS permission denial surfaces as `ok=False` on first `read()`; error message suggests checking *System Settings → Privacy → Camera*

### `emotion.py` — `EmotionDetector`

- Uses `FER(mtcnn=True)` — MTCNN face detector is more robust than Haar cascades for small/tilted faces
- `detect(frame) → list[dict]` — each dict: `{"box": (x, y, w, h), "emotion": str, "score": float}`
- Returns `[]` when no faces detected — never raises exceptions
- `mtcnn=True` configurable via constructor flag for slower machines
- On first run, FER downloads model weights; logs `"Cargando modelo..."` to console

### `overlay.py` — `draw_overlay(frame, results)`

**HUD aesthetic — angular brackets style:**
- Face bounding box drawn as corner brackets (`⌐`-style), not a full rectangle — 4 L-shaped corners, 2px line
- Emotion label: `HAPPY 93%` in `cv2.FONT_HERSHEY_SIMPLEX` size 0.6, with 1px black shadow for contrast
- Header bar (top-left): `[ EMOTION DETECTOR ]  FPS: 14`

**Color palette per emotion:**
| Emotion   | Color (BGR)          |
|-----------|----------------------|
| happy     | (0, 220, 80) green   |
| sad       | (200, 100, 50) blue  |
| angry     | (0, 50, 220) red     |
| fear      | (200, 50, 200) magenta |
| surprise  | (0, 200, 220) yellow |
| disgust   | (0, 130, 220) orange |
| neutral   | (180, 180, 180) gray |

### `main.py`

- Initializes `CameraCapture` and `EmotionDetector`
- Loop: `read → detect → overlay → imshow → waitKey(1)`
- FPS calculated with `time.time()` delta between frames, passed to `draw_overlay`
- Exit on `q` or `ESC` key
- `try/finally` ensures `capture.release()` and `cv2.destroyAllWindows()` always run
- Catches `KeyboardInterrupt` cleanly

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Camera not available | `RuntimeError` → `sys.exit(1)` with clear message |
| macOS camera permission denied | Error message suggests Privacy settings path |
| Model loading (first run) | Console message `"Cargando modelo..."`, no crash |
| Corrupted/empty frame | `detect()` returns `[]`, loop continues |
| No faces in frame | `[]` returned, overlay shows header only |

---

## Dependencies (`requirements.txt`)

```
opencv-python>=4.8,<5
fer>=22.5
tensorflow>=2.13
mtcnn>=0.1.1
numpy>=1.24
```

---

## README Sections

1. Description (2 lines)
2. Setup: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
3. Run: `python main.py`
4. Controls: `q` or `ESC` to quit
5. macOS permissions: System Settings → Privacy & Security → Camera → grant access to Terminal/IDE
6. Privacy note: all processing is local, no video is sent to external servers

---

## Out of Scope

- Training a model from scratch
- Web or mobile app
- Threading / concurrent capture+inference pipeline
- GPU acceleration

---

## Success Criteria

Running `python main.py` opens the camera window. Detected faces are highlighted with angular HUD brackets and an emotion label that updates in near real time (~8–15 FPS). No unhandled exceptions under normal conditions (camera connected, permissions granted).
