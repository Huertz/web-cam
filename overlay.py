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
_FONT_SMALL = 0.38
_CORNER_LEN = 24
_THICKNESS  = 2
_ARC_R      = 14


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
        ((x,     y),     (x,       y + arm)),
        ((x + w, y),     (x + w - arm, y)),
        ((x + w, y),     (x + w,   y + arm)),
        ((x,     y + h), (x + arm, y + h)),
        ((x,     y + h), (x,       y + h - arm)),
        ((x + w, y + h), (x + w - arm, y + h)),
        ((x + w, y + h), (x + w,   y + h - arm)),
    ]
    for p1, p2 in pts:
        cv2.line(frame, p1, p2, color, _THICKNESS + 1, cv2.LINE_AA)
    for cx, cy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
        cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)


def _draw_badge(frame, x, y, emotion, score, color):
    """
    Badge above the face box:  [ arc  HAPPY ]
    Arc and text sit in the same pill-shaped background.
    """
    r     = _ARC_R
    label = emotion.upper()
    (lw, lh), _ = cv2.getTextSize(label, _FONT, _FONT_MED, 1)

    pad      = 6
    arc_diam = r * 2
    gap      = 8                          # space between arc and text
    badge_w  = pad + arc_diam + gap + lw + pad
    badge_h  = arc_diam + pad * 2
    bx       = x
    by       = max(0, y - badge_h - 4)

    # Clamp right edge
    fw = frame.shape[1]
    if bx + badge_w > fw:
        bx = fw - badge_w - 2

    # Background pill
    _fill_rect(frame, bx, by, bx + badge_w, by + badge_h, (10, 10, 10), 0.75)
    # Colored top border
    cv2.line(frame, (bx, by), (bx + badge_w, by), color, 1, cv2.LINE_AA)

    # Arc — centered vertically in badge
    arc_cx = bx + pad + r
    arc_cy = by + badge_h // 2

    cv2.ellipse(frame, (arc_cx, arc_cy), (r, r), -90, 0, 360,
                (50, 50, 50), 1, cv2.LINE_AA)
    end_angle = int(360 * score)
    if end_angle > 0:
        cv2.ellipse(frame, (arc_cx, arc_cy), (r, r), -90, 0, end_angle,
                    color, 2, cv2.LINE_AA)

    # Percentage inside arc
    pct = f"{int(score * 100)}%"
    (pw, ph), _ = cv2.getTextSize(pct, _FONT, _FONT_SMALL - 0.05, 1)
    cv2.putText(frame, pct, (arc_cx - pw // 2, arc_cy + ph // 2),
                _FONT, _FONT_SMALL - 0.05, color, 1, cv2.LINE_AA)

    # Emotion text to the right of the arc
    tx = arc_cx + r + gap
    ty = arc_cy + lh // 2
    cv2.putText(frame, label, (tx + 1, ty + 1), _FONT, _FONT_MED,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty), _FONT, _FONT_MED,
                color, 1, cv2.LINE_AA)


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

        _draw_corners(frame, x, y, w, h, color)
        _draw_badge(frame, x, y, emotion, score, color)

    return frame
