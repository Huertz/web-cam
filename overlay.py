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
