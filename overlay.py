import cv2
import numpy as np

EMOTION_COLORS = {
    "happy":    (80, 220, 0),
    "sad":      (200, 100, 50),
    "angry":    (50, 50, 220),
    "fear":     (200, 50, 200),
    "surprise": (0, 220, 220),
    "disgust":  (0, 130, 220),
    "neutral":  (180, 180, 180),
}

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_MED   = 0.55
_FONT_SMALL = 0.40
_CORNER_LEN = 24
_THICKNESS  = 2
_ARC_R      = 28   # radius of the confidence arc


def _fill_rect(frame, x1, y1, x2, y2, color, alpha):
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    bg  = np.full(roi.shape, color, dtype=np.uint8)
    frame[y1:y2, x1:x2] = cv2.addWeighted(bg, alpha, roi, 1 - alpha, 0)


def _draw_corners(frame, x, y, w, h, color):
    arm = min(_CORNER_LEN, w // 2, h // 2)
    if arm <= 0:
        return
    pts = [
        ((x,     y),     (x + arm, y)),
        ((x,     y),     (x,     y + arm)),
        ((x + w, y),     (x + w - arm, y)),
        ((x + w, y),     (x + w, y + arm)),
        ((x,     y + h), (x + arm, y + h)),
        ((x,     y + h), (x,     y + h - arm)),
        ((x + w, y + h), (x + w - arm, y + h)),
        ((x + w, y + h), (x + w, y + h - arm)),
    ]
    for p1, p2 in pts:
        cv2.line(frame, p1, p2, color, _THICKNESS + 1, cv2.LINE_AA)
    for cx, cy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
        cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)


def _draw_arc(frame, cx, cy, score, color):
    """Large confidence arc centered at (cx, cy), anchored inside the face box corner."""
    r = _ARC_R
    # Dark background ring
    cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, 360, (40, 40, 40), 2, cv2.LINE_AA)
    # Colored fill arc
    end_angle = int(360 * score)
    if end_angle > 0:
        cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, end_angle, color, 3, cv2.LINE_AA)
    # Percentage inside
    pct = f"{int(score * 100)}%"
    (tw, th), _ = cv2.getTextSize(pct, _FONT, _FONT_SMALL, 1)
    cv2.putText(frame, pct, (cx - tw // 2, cy + th // 2),
                _FONT, _FONT_SMALL, color, 1, cv2.LINE_AA)


def _draw_label(frame, x, y, w, h, emotion, color):
    """Emotion name badge sitting above the face box."""
    label = emotion.upper()
    (lw, lh), _ = cv2.getTextSize(label, _FONT, _FONT_MED, 1)
    pad = 5
    bx1 = x
    by1 = max(0, y - lh - pad * 2 - 2)
    bx2 = bx1 + lw + pad * 2
    by2 = by1 + lh + pad * 2

    _fill_rect(frame, bx1, by1, bx2, by2, (10, 10, 10), 0.72)
    cv2.line(frame, (bx1, by1), (bx2, by1), color, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (bx1 + pad + 1, by2 - pad + 1),
                _FONT, _FONT_MED, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (bx1 + pad, by2 - pad),
                _FONT, _FONT_MED, color, 1, cv2.LINE_AA)


def draw_overlay(frame, results, fps=0.0):
    fh, fw = frame.shape[:2]

    # ── Header ──────────────────────────────────────────────────────
    _fill_rect(frame, 0, 0, fw, 36, (5, 5, 5), 0.75)
    cv2.line(frame, (0, 36), (fw, 36), (80, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "EMOTION DETECTOR", (12, 25),
                _FONT, _FONT_MED, (80, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS  {fps:.1f}", (fw - 82, 25),
                _FONT, _FONT_SMALL, (150, 150, 150), 1, cv2.LINE_AA)

    # ── Per-face ─────────────────────────────────────────────────────
    for r in results:
        x, y, w, h = r["box"]
        emotion    = r["emotion"]
        score      = r["score"]
        color      = EMOTION_COLORS.get(emotion, (200, 200, 200))

        # Corner brackets
        _draw_corners(frame, x, y, w, h, color)

        # Emotion name above the box
        _draw_label(frame, x, y, w, h, emotion, color)

        # Confidence arc — bottom-right corner, inset so it sits inside the box
        arc_cx = min(x + w - _ARC_R - 6, fw - _ARC_R - 2)
        arc_cy = min(y + h - _ARC_R - 6, fh - _ARC_R - 2)
        arc_cx = max(arc_cx, _ARC_R + 2)
        arc_cy = max(arc_cy, _ARC_R + 2)
        _draw_arc(frame, arc_cx, arc_cy, score, color)

    return frame
