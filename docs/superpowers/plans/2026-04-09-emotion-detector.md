# Emotion Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular Python desktop app that opens the webcam, detects faces with FER, classifies emotions in real time, and renders a HUD-style overlay via OpenCV.

**Architecture:** Four modules with single responsibilities (`capture.py`, `emotion.py`, `overlay.py`, `main.py`) orchestrated by a main loop. FER with MTCNN handles face detection and emotion classification. OpenCV handles capture and display.

**Tech Stack:** Python 3.10+, OpenCV 4.8, FER 22.5+, TensorFlow 2.13+, MTCNN, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `capture.py` | Open/read/release webcam via cv2.VideoCapture |
| `emotion.py` | Wrap FER detector, return normalized detection dicts |
| `overlay.py` | Draw HUD brackets and emotion labels on frame |
| `main.py` | Main loop: orchestrate modules, handle FPS and exit |
| `requirements.txt` | Pinned dependencies |
| `README.md` | Setup, run instructions, permissions, privacy note |
| `tests/conftest.py` | Add project root to sys.path |
| `tests/test_capture.py` | Unit tests for CameraCapture |
| `tests/test_emotion.py` | Unit tests for EmotionDetector |
| `tests/test_overlay.py` | Unit tests for draw_overlay |

---

## Task 1: Project setup — requirements.txt, pytest, conftest

**Files:**
- Create: `requirements.txt`
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
opencv-python>=4.8,<5
fer>=22.5
tensorflow>=2.13
mtcnn>=0.1.1
numpy>=1.24
pytest>=7.4
```

- [ ] **Step 2: Create `tests/__init__.py`** (empty file)

- [ ] **Step 3: Create `tests/conftest.py`**

```python
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 4: Create and activate virtual environment, install deps**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Expected: all packages install without errors. TensorFlow will take a minute.

- [ ] **Step 5: Verify pytest runs**

```bash
pytest tests/ -v
```

Expected: `no tests ran` (0 collected) — that's fine, no tests yet.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/conftest.py tests/__init__.py
git commit -m "chore: add requirements and pytest setup"
```

---

## Task 2: `capture.py` — CameraCapture

**Files:**
- Create: `capture.py`
- Create: `tests/test_capture.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_capture.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


def test_camera_opens_successfully():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=mock_cap):
        from capture import CameraCapture
        cam = CameraCapture()
        assert cam is not None


def test_camera_raises_if_not_opened():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    with patch("cv2.VideoCapture", return_value=mock_cap):
        from capture import CameraCapture
        with pytest.raises(RuntimeError, match="No se pudo abrir la cámara índice 0"):
            CameraCapture()


def test_read_returns_frame():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, fake_frame)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        from capture import CameraCapture
        cam = CameraCapture()
        ok, frame = cam.read()
        assert ok is True
        assert frame.shape == (480, 640, 3)


def test_release_calls_cap_release():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=mock_cap):
        from capture import CameraCapture
        cam = CameraCapture()
        cam.release()
        mock_cap.release.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_capture.py -v
```

Expected: `ModuleNotFoundError: No module named 'capture'`

- [ ] **Step 3: Implement `capture.py`**

```python
import cv2


class CameraCapture:
    def __init__(self, device_index=0, width=640, height=480):
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara índice {device_index}. "
                "Verifica permisos en Ajustes del sistema → Privacidad y seguridad → Cámara."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_capture.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add capture.py tests/test_capture.py
git commit -m "feat: add CameraCapture module"
```

---

## Task 3: `emotion.py` — EmotionDetector

**Files:**
- Create: `emotion.py`
- Create: `tests/test_emotion.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_emotion.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


def test_detect_returns_empty_list_when_no_faces():
    mock_fer = MagicMock()
    mock_fer.detect_emotions.return_value = []
    with patch("fer.FER", return_value=mock_fer):
        from emotion import EmotionDetector
        detector = EmotionDetector(use_mtcnn=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result == []


def test_detect_returns_correct_structure():
    mock_fer = MagicMock()
    mock_fer.detect_emotions.return_value = [
        {
            "box": (10, 20, 100, 100),
            "emotions": {
                "happy": 0.9, "sad": 0.05, "angry": 0.02,
                "fear": 0.01, "surprise": 0.01, "disgust": 0.005, "neutral": 0.005,
            },
        }
    ]
    with patch("fer.FER", return_value=mock_fer):
        from emotion import EmotionDetector
        detector = EmotionDetector(use_mtcnn=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert len(result) == 1
        assert result[0]["box"] == (10, 20, 100, 100)
        assert result[0]["emotion"] == "happy"
        assert result[0]["score"] == pytest.approx(0.9)


def test_detect_picks_dominant_emotion():
    mock_fer = MagicMock()
    mock_fer.detect_emotions.return_value = [
        {
            "box": (0, 0, 50, 50),
            "emotions": {
                "happy": 0.1, "sad": 0.7, "angry": 0.1,
                "fear": 0.02, "surprise": 0.03, "disgust": 0.02, "neutral": 0.03,
            },
        }
    ]
    with patch("fer.FER", return_value=mock_fer):
        from emotion import EmotionDetector
        detector = EmotionDetector(use_mtcnn=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result[0]["emotion"] == "sad"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_emotion.py -v
```

Expected: `ModuleNotFoundError: No module named 'emotion'`

- [ ] **Step 3: Implement `emotion.py`**

```python
from fer import FER


class EmotionDetector:
    def __init__(self, use_mtcnn=True):
        print("Cargando modelo...")
        self._detector = FER(mtcnn=use_mtcnn)

    def detect(self, frame):
        raw = self._detector.detect_emotions(frame)
        results = []
        for r in raw:
            emotions = r["emotions"]
            dominant = max(emotions, key=emotions.get)
            results.append({
                "box": r["box"],
                "emotion": dominant,
                "score": emotions[dominant],
            })
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_emotion.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add emotion.py tests/test_emotion.py
git commit -m "feat: add EmotionDetector module"
```

---

## Task 4: `overlay.py` — draw_overlay

**Files:**
- Create: `overlay.py`
- Create: `tests/test_overlay.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_overlay.py`:

```python
import numpy as np
from overlay import draw_overlay, EMOTION_COLORS


def test_draw_overlay_returns_same_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = draw_overlay(frame, [], fps=15.0)
    assert result is frame


def test_draw_overlay_header_modifies_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_overlay(frame, [], fps=10.0)
    assert frame.sum() > 0


def test_draw_overlay_with_detection_modifies_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = [{"box": (50, 50, 100, 100), "emotion": "happy", "score": 0.85}]
    draw_overlay(frame, results, fps=12.0)
    assert frame.sum() > 0


def test_emotion_colors_cover_all_seven_emotions():
    expected = {"happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"}
    assert set(EMOTION_COLORS.keys()) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_overlay.py -v
```

Expected: `ModuleNotFoundError: No module named 'overlay'`

- [ ] **Step 3: Implement `overlay.py`**

```python
import cv2

EMOTION_COLORS = {
    "happy":    (80, 220, 0),
    "sad":      (200, 100, 50),
    "angry":    (50, 50, 220),
    "fear":     (200, 50, 200),
    "surprise": (0, 220, 220),
    "disgust":  (0, 130, 220),
    "neutral":  (180, 180, 180),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_CORNER_LEN = 20
_THICKNESS = 2


def _draw_corners(frame, x, y, w, h, color):
    pts = [
        ((x, y), (x + _CORNER_LEN, y)),
        ((x, y), (x, y + _CORNER_LEN)),
        ((x + w, y), (x + w - _CORNER_LEN, y)),
        ((x + w, y), (x + w, y + _CORNER_LEN)),
        ((x, y + h), (x + _CORNER_LEN, y + h)),
        ((x, y + h), (x, y + h - _CORNER_LEN)),
        ((x + w, y + h), (x + w - _CORNER_LEN, y + h)),
        ((x + w, y + h), (x + w, y + h - _CORNER_LEN)),
    ]
    for p1, p2 in pts:
        cv2.line(frame, p1, p2, color, _THICKNESS)


def draw_overlay(frame, results, fps=0.0):
    header = f"[ EMOTION DETECTOR ]  FPS: {fps:.1f}"
    cv2.putText(frame, header, (11, 26), _FONT, _FONT_SCALE, (0, 0, 0), 2)
    cv2.putText(frame, header, (10, 25), _FONT, _FONT_SCALE, (180, 255, 180), 1)

    for r in results:
        x, y, w, h = r["box"]
        emotion = r["emotion"]
        score = r["score"]
        color = EMOTION_COLORS.get(emotion, (200, 200, 200))

        _draw_corners(frame, x, y, w, h, color)

        label = f"{emotion.upper()} {int(score * 100)}%"
        cv2.putText(frame, label, (x + 1, y - 8), _FONT, _FONT_SCALE, (0, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 9), _FONT, _FONT_SCALE, color, 1)

    return frame
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_overlay.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Run all tests to verify no regressions**

```bash
pytest tests/ -v
```

Expected: `11 passed`

- [ ] **Step 6: Commit**

```bash
git add overlay.py tests/test_overlay.py
git commit -m "feat: add HUD overlay module"
```

---

## Task 5: `main.py` — main loop

**Files:**
- Create: `main.py`

> Note: `main.py` is not unit-tested (it requires a real camera and display). Manual smoke test instructions are provided instead.

- [ ] **Step 1: Implement `main.py`**

```python
import sys
import time

import cv2

from capture import CameraCapture
from emotion import EmotionDetector
from overlay import draw_overlay


def main():
    try:
        camera = CameraCapture()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    detector = EmotionDetector()
    prev_time = time.time()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Error: no se pudo leer el frame.", file=sys.stderr)
                break

            results = detector.detect(frame)

            now = time.time()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
            prev_time = now

            draw_overlay(frame, results, fps)
            cv2.imshow("Emotion Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test manually**

```bash
python main.py
```

Expected: webcam window opens, face is framed with corner brackets, emotion label updates in near real time. Press `q` or `ESC` to exit cleanly.

If the camera window opens but shows a black frame → check macOS camera permissions: *System Settings → Privacy & Security → Camera → Terminal (or your IDE)*.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add main loop"
```

---

## Task 6: `README.md`

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# Emotion Detector

Python desktop app that opens the webcam, detects faces, and classifies emotions in real time using FER + OpenCV.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> First run downloads TensorFlow model weights (~500 MB). This is a one-time operation.

## Run

```bash
python main.py
```

| Key | Action |
|-----|--------|
| `q` | Quit |
| `ESC` | Quit |

## macOS camera permissions

If the camera doesn't open, grant access in:
**System Settings → Privacy & Security → Camera → Terminal** (or your IDE)

## Privacy

All processing is 100% local. No video frames or images are sent to any external server.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and run instructions"
```

---

## Self-Review Notes

- All spec requirements covered: webcam capture ✓, face detection ✓, 7-emotion classification ✓, HUD display ✓, clean exit ✓, modular structure ✓, error handling ✓, requirements.txt ✓, README ✓
- No TBDs or placeholders
- Type signatures are consistent across tasks: `detect()` returns `list[dict]` with keys `box`, `emotion`, `score` — used identically in `overlay.py`
- `draw_overlay` signature `(frame, results, fps=0.0)` consistent in tests and implementation
- `use_mtcnn` flag in `EmotionDetector.__init__` used consistently in tests (False) and production (True default)
